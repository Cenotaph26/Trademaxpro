"""
Capital Brain v15 — Hierarchical Portfolio Orchestrator

Mimarinin kalbi. Tüm specialist agent'lara kapital slot'u atar,
toplam portföy riskini yönetir ve Risk Governor'a danışır.

Akış:
  1. Her agent periyodik olarak signal üretir
  2. Capital Brain sinyal kuyruğunu toplar
  3. Risk Governor onay verir / veto eder
  4. Capital Brain kabul edilen sinyale USDT allocation atar
  5. Execution Layer işlemi açar
  6. Agent PnL Scorer sonucu izler, agent'ın skoru güncellenir
  7. RL Meta-Controller gelecek allokasyon ağırlıklarını öğrenir
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    ACTIVE   = "active"
    PAUSED   = "paused"   # geçici duraklama (düşük skor)
    BANNED   = "banned"   # günlük limit aşıldı


@dataclass
class AgentSlot:
    """Her specialist agent için Capital Brain'in tuttuğu kayıt."""
    agent_id: str
    name: str
    base_allocation_pct: float   # portföyün %'si (örn. 0.25 = %25)
    current_allocation_pct: float
    status: AgentStatus = AgentStatus.ACTIVE
    open_positions: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None
    score: float = 1.0           # 0-2 arası: 1=nötr, 2=çok iyi, 0=kötü


@dataclass
class AllocationRequest:
    """Bir agent'ın kapital talep etmesi."""
    agent_id: str
    symbol: str
    side: str
    signal_strength: float       # 0-1
    suggested_usdt: float
    sl_pct: float
    tp_pct: float
    leverage: int
    strategy_tag: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AllocationResult:
    approved: bool
    allocated_usdt: float
    reason: str = ""
    leverage: int = 3


class CapitalBrain:
    """
    Portföy bütçesini yönetir. Tüm agent talepleri buradan geçer.
    
    Sorumluluklar:
    - Toplam risk bütçesi izleme
    - Agent'lara dinamik kapital ağırlığı
    - Korelasyon engeli (aynı yönde çok fazla pozisyon)
    - Agent PnL skor bazlı ağırlık ayarı
    """

    def __init__(self, data_client, risk_governor, settings):
        self.data     = data_client
        self.governor = risk_governor
        self.s        = settings

        # Agent kayıt defteri
        self._agents: Dict[str, AgentSlot] = {}

        # Toplam portföy parametreleri
        self.max_total_risk_pct  = 0.20   # Toplam açık pozisyonların maks %20'si
        self.max_single_side_pct = 0.15   # Tek yönde (long/short) maks %15
        self.min_allocation_usdt = 110.0  # Binance min notional + buffer
        self.rebalance_interval  = 300    # 5dk'da bir skor bazlı rebalance

        # Skor parametreleri
        self.score_decay         = 0.995  # Her saat skoru hafifçe düşür
        self.score_win_bonus     = 0.05
        self.score_loss_penalty  = 0.10
        self.score_consec_penalty= 0.15   # Ardışık kayıplar için ekstra ceza
        self.pause_threshold     = 0.40   # Bu skorun altında agent duraklıyor
        self.ban_consec_losses   = 4      # Ardışık 4 kayıpta günlük ban

        self._running = False
        logger.info("🧠 Capital Brain başlatıldı")

    # ─── Agent kayıt ──────────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, name: str, base_pct: float):
        """Bir specialist agent'ı kaydet. Bot başlangıcında çağrılır."""
        self._agents[agent_id] = AgentSlot(
            agent_id=agent_id,
            name=name,
            base_allocation_pct=base_pct,
            current_allocation_pct=base_pct,
        )
        logger.info(f"🤖 Agent kayıtlı: {name} ({agent_id}) — base_alloc={base_pct*100:.0f}%")

    def get_agent(self, agent_id: str) -> Optional[AgentSlot]:
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[AgentSlot]:
        return list(self._agents.values())

    # ─── Kapital tahsisi ──────────────────────────────────────────────────────

    async def request_allocation(self, req: AllocationRequest) -> AllocationResult:
        """
        Bir agent kapital talep eder. Capital Brain onaylar veya reddeder.
        Bu fonksiyon Risk Governor'a danışır, korelasyon kontrolü yapar,
        ve mevcut portföy bütçesine göre USDT miktarı hesaplar.
        """
        slot = self._agents.get(req.agent_id)
        if not slot:
            return AllocationResult(False, 0, f"Bilinmeyen agent: {req.agent_id}")

        # ── 1. Agent durumu ─────────────────────────────────────────
        if slot.status == AgentStatus.BANNED:
            return AllocationResult(False, 0, f"{slot.name} günlük ban (ardışık kayıp)")
        if slot.status == AgentStatus.PAUSED:
            return AllocationResult(False, 0, f"{slot.name} düşük skor nedeniyle duraklatıldı")

        # ── 2. Risk Governor ─────────────────────────────────────────
        gov_ok, gov_reason = await self.governor.approve(req)
        if not gov_ok:
            return AllocationResult(False, 0, f"Risk Governor veto: {gov_reason}")

        # ── 3. Korelasyon kontrolü ──────────────────────────────────
        # Aynı sembolde birden fazla ajan aynı yönde açmasın
        corr_ok, corr_reason = self._check_correlation(req)
        if not corr_ok:
            return AllocationResult(False, 0, f"Korelasyon engeli: {corr_reason}")

        # ── 4. Portföy bütçesi ──────────────────────────────────────
        balance = await self._get_balance()
        if balance <= 0:
            return AllocationResult(False, 0, "Bakiye alınamadı")

        # Agent'ın skor-ağırlıklı allokasyonu
        # Skor 1.0 → base_pct, skor 2.0 → base_pct * 1.5, skor 0.5 → base_pct * 0.5
        score_multiplier  = max(0.3, min(1.5, slot.score))
        effective_pct     = slot.current_allocation_pct * score_multiplier
        max_usdt_for_slot = balance * effective_pct

        # Portföy toplam risk limiti
        total_open_notional = await self._get_total_open_notional()
        max_total_notional  = balance * self.max_total_risk_pct * req.leverage
        available_notional  = max(0, max_total_notional - total_open_notional)

        allocated = min(req.suggested_usdt, max_usdt_for_slot, available_notional)

        if allocated < self.min_allocation_usdt:
            return AllocationResult(
                False, 0,
                f"Allokasyon çok küçük: ${allocated:.0f} < min ${self.min_allocation_usdt:.0f} "
                f"(bakiye=${balance:.0f}, budget_pct={effective_pct*100:.1f}%)"
            )

        # Leverage güvenli sınır: yüksek skorlu agent daha yüksek kaldıraç alabilir
        safe_leverage = self._calc_safe_leverage(req.leverage, slot.score)

        slot.open_positions += 1
        slot.last_trade_time = datetime.now(timezone.utc)

        logger.info(
            f"✅ Brain alloc: {slot.name} → {req.symbol} {req.side} "
            f"${allocated:.0f} @{safe_leverage}x (skor={slot.score:.2f})"
        )
        return AllocationResult(
            approved=True,
            allocated_usdt=round(allocated, 2),
            reason="OK",
            leverage=safe_leverage,
        )

    def _check_correlation(self, req: AllocationRequest) -> tuple:
        """Aynı yönde çok yoğun açık olmasın."""
        same_dir = sum(
            1 for a in self._agents.values()
            if a.open_positions > 0
        )
        if same_dir >= 4:
            return False, f"Portföyde zaten {same_dir} açık pozisyon"
        return True, "OK"

    def _calc_safe_leverage(self, requested: int, score: float) -> int:
        """Düşük skorlu agent'a daha düşük kaldıraç ver."""
        if score >= 1.5:
            return min(requested, 10)
        elif score >= 1.0:
            return min(requested, 5)
        elif score >= 0.7:
            return min(requested, 3)
        else:
            return min(requested, 2)

    async def _get_balance(self) -> float:
        try:
            bal = await self.data.get_balance()
            return float(bal.get("total", 0) or 0)
        except Exception:
            return 0.0

    async def _get_total_open_notional(self) -> float:
        try:
            positions = await self.data.exchange.fetch_positions()
            total = 0.0
            for p in positions:
                contracts = float(p.get("contracts") or 0)
                if abs(contracts) < 1e-9:
                    continue
                notional = abs(float(p.get("notional") or 0))
                total += notional
            return total
        except Exception:
            return 0.0

    # ─── PnL Geri Bildirim (Agent Scorer) ────────────────────────────────────

    def record_outcome(self, agent_id: str, pnl: float, symbol: str = ""):
        """
        İşlem kapanınca çağrılır. Agent skorunu günceller.
        RL Meta-Controller da bu veriye göre öğrenir.
        """
        slot = self._agents.get(agent_id)
        if not slot:
            return

        slot.open_positions = max(0, slot.open_positions - 1)
        slot.total_pnl += pnl
        slot.daily_pnl += pnl

        if pnl > 0:
            slot.win_count += 1
            slot.consecutive_losses = 0
            slot.score = min(2.0, slot.score + self.score_win_bonus)
            logger.info(f"📈 {slot.name} win: +${pnl:.2f} | skor={slot.score:.2f}")
        else:
            slot.loss_count += 1
            slot.consecutive_losses += 1
            penalty = self.score_loss_penalty
            if slot.consecutive_losses >= 2:
                penalty += self.score_consec_penalty * (slot.consecutive_losses - 1)
            slot.score = max(0.1, slot.score - penalty)
            logger.warning(f"📉 {slot.name} loss: ${pnl:.2f} | skor={slot.score:.2f} | consec={slot.consecutive_losses}")

        # Durum güncellemesi
        if slot.consecutive_losses >= self.ban_consec_losses:
            slot.status = AgentStatus.BANNED
            logger.error(f"🚫 {slot.name} GÜNLÜK BAN: {slot.consecutive_losses} ardışık kayıp")
        elif slot.score < self.pause_threshold:
            slot.status = AgentStatus.PAUSED
            logger.warning(f"⏸️ {slot.name} DURAKLATILDI: skor={slot.score:.2f}")
        else:
            slot.status = AgentStatus.ACTIVE

    # ─── Rebalance ────────────────────────────────────────────────────────────

    def rebalance_allocations(self):
        """
        Agent skorlarına göre allokasyon ağırlıklarını yeniden dağıt.
        Toplam her zaman 1.0'a normalize edilir.
        """
        active = [a for a in self._agents.values() if a.status == AgentStatus.ACTIVE]
        if not active:
            return

        # Skor bazlı ağırlık
        total_score = sum(a.score for a in active) or 1.0
        for slot in active:
            raw_weight = (slot.score / total_score)
            # Base allokasyonu ± %50 değiştirebilir
            adjusted = slot.base_allocation_pct * (0.5 + raw_weight * len(active) * 0.5)
            slot.current_allocation_pct = round(max(0.05, min(0.40, adjusted)), 4)

        # Paused/banned agent'lar minimum allokasyon alır
        for slot in self._agents.values():
            if slot.status != AgentStatus.ACTIVE:
                slot.current_allocation_pct = 0.02

        logger.debug(f"🔄 Rebalance: " + " | ".join(
            f"{a.name}={a.current_allocation_pct*100:.1f}% (skor={a.score:.2f})"
            for a in self._agents.values()
        ))

    def daily_reset(self):
        """Gece yarısı: günlük metrikler sıfırlanır, ban'lar kaldırılır."""
        for slot in self._agents.values():
            slot.daily_pnl = 0.0
            slot.consecutive_losses = 0
            if slot.status == AgentStatus.BANNED:
                slot.status = AgentStatus.ACTIVE
                slot.score = max(slot.score, 0.50)  # min skor ile geri dön
                logger.info(f"🌅 {slot.name} ban kaldırıldı (günlük reset)")

    # ─── Background loop ──────────────────────────────────────────────────────

    async def start(self):
        """Periyodik rebalance ve skor decay."""
        self._running = True
        logger.info("🧠 Capital Brain loop başlatıldı")
        last_daily_reset = datetime.now(timezone.utc).date()
        tick = 0

        while self._running:
            await asyncio.sleep(60)
            tick += 1

            # Skor zamanla hafifçe çürür (yakın sonuçlara daha az ağırlık vermek için)
            for slot in self._agents.values():
                slot.score = max(0.1, slot.score * self.score_decay)

            # Rebalance
            if tick % 5 == 0:   # 5dk
                self.rebalance_allocations()

            # Günlük reset
            today = datetime.now(timezone.utc).date()
            if today != last_daily_reset:
                self.daily_reset()
                last_daily_reset = today

    def get_status(self) -> dict:
        return {
            "agents": [
                {
                    "id":            a.agent_id,
                    "name":          a.name,
                    "status":        a.status.value,
                    "score":         round(a.score, 3),
                    "alloc_pct":     round(a.current_allocation_pct * 100, 1),
                    "open":          a.open_positions,
                    "daily_pnl":     round(a.daily_pnl, 2),
                    "total_pnl":     round(a.total_pnl, 2),
                    "winrate":       round(a.win_count / max(1, a.win_count + a.loss_count), 3),
                    "consec_losses": a.consecutive_losses,
                }
                for a in self._agents.values()
            ]
        }
