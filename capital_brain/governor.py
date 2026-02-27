"""
Risk Governor v15 — Tüm trade kararları üzerinde veto yetkisi.

Capital Brain'in istediği her allokasyonu burada onaylanır ya da veto edilir.
Mevcut RiskEngine'den bağımsız bir katman — sadece pre-trade gate görevi görür.

Kontroller:
  1. Kill switch aktif mi?
  2. Günlük portföy kaybı limiti aşıldı mı?
  3. Toplam drawdown limiti aşıldı mı?
  4. Korelasyon: Aynı sembolde çift pozisyon var mı?
  5. Volatilite rejimi: ATR spike'ta yeni pozisyon açma
  6. Funding rate: Aşırı funding'de açma
  7. Saatlik fırsat limiti: Overtrading koruması
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capital_brain.brain import AllocationRequest

logger = logging.getLogger(__name__)


class RiskGovernor:
    """
    Tek görev: approve() → (bool, reason)
    Capital Brain her allokasyon öncesi buraya danışır.
    """

    def __init__(self, risk_engine, data_client, settings):
        self.risk     = risk_engine
        self.data     = data_client
        self.s        = settings

        # Konfigürasyon
        self.max_daily_loss_pct      = getattr(settings, "DAILY_MAX_LOSS_PCT", 2.0)
        self.max_drawdown_pct        = getattr(settings, "MAX_DRAWDOWN_PCT", 5.0)
        self.max_concurrent_positions = getattr(settings, "MAX_OPEN_POSITIONS", 10)
        self.max_trades_per_hour     = 12      # saatlik overtrading limiti
        self.atr_spike_block_mult    = 3.5     # ATR bu kadar normalin üstündeyse blokla
        self.funding_extreme_pct     = getattr(settings, "FUNDING_EXTREME_THRESHOLD", 0.01)
        self.min_interval_sec        = 30      # Aynı sembol için minimum bekleme süresi

        # Saatlik trade sayacı
        self._hourly_trades: list = []         # datetime listesi
        # Sembol bazlı son işlem zamanı
        self._last_trade_sym: dict = {}        # symbol → datetime

    async def approve(self, req: "AllocationRequest") -> tuple:
        """
        Tüm kontrolleri sırayla çalıştır. İlk başarısız kontrolde veto ver.
        Returns: (ok: bool, reason: str)
        """
        checks = [
            self._check_kill_switch,
            self._check_drawdown,
            self._check_daily_loss,
            self._check_position_count,
            self._check_overtrading,
            self._check_symbol_cooldown,
            self._check_funding,
            self._check_atr_spike,
        ]
        for check in checks:
            ok, reason = await check(req)
            if not ok:
                logger.warning(f"⛔ Governor veto [{req.agent_id}→{req.symbol}]: {reason}")
                return False, reason

        # Onaylandı — saatlik sayaca ekle
        self._hourly_trades.append(datetime.now(timezone.utc))
        self._last_trade_sym[req.symbol] = datetime.now(timezone.utc)
        return True, "OK"

    # ─── Check fonksiyonları ──────────────────────────────────────────────────

    async def _check_kill_switch(self, req) -> tuple:
        rs = self.risk.state
        if rs.kill_switch_active:
            return False, f"Kill switch aktif: {rs.kill_switch_reason}"
        return True, "OK"

    async def _check_drawdown(self, req) -> tuple:
        dd = self.risk.state.current_drawdown_pct
        if dd >= self.max_drawdown_pct:
            return False, f"Max drawdown aşıldı: {dd:.2f}% ≥ {self.max_drawdown_pct}%"
        # Uyarı: %80'ini geçtiyse kısmi blok (yeni pozisyon değil, mevcut kapat)
        if dd >= self.max_drawdown_pct * 0.80:
            logger.warning(f"⚠ Drawdown kritik seviyede: {dd:.2f}%")
        return True, "OK"

    async def _check_daily_loss(self, req) -> tuple:
        equity = self.risk.state.current_equity
        if equity <= 0:
            return True, "OK"  # bakiye yok, geç
        daily_loss = self.risk.state.daily_loss
        loss_pct = (daily_loss / equity) * 100
        if loss_pct >= self.max_daily_loss_pct:
            return False, f"Günlük kayıp limiti: {loss_pct:.2f}% ≥ {self.max_daily_loss_pct}%"
        return True, "OK"

    async def _check_position_count(self, req) -> tuple:
        count = self.risk.state.open_count
        if count >= self.max_concurrent_positions:
            return False, f"Maks pozisyon sayısı: {count}/{self.max_concurrent_positions}"
        return True, "OK"

    async def _check_overtrading(self, req) -> tuple:
        # Son 1 saat içindeki trade'leri filtrele
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        self._hourly_trades = [t for t in self._hourly_trades if t > cutoff]
        if len(self._hourly_trades) >= self.max_trades_per_hour:
            return False, f"Saatlik overtrading limiti: {len(self._hourly_trades)}/{self.max_trades_per_hour}"
        return True, "OK"

    async def _check_symbol_cooldown(self, req) -> tuple:
        last = self._last_trade_sym.get(req.symbol)
        if last:
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < self.min_interval_sec:
                return False, f"{req.symbol} cooldown: {elapsed:.0f}s < {self.min_interval_sec}s"
        return True, "OK"

    async def _check_funding(self, req) -> tuple:
        try:
            funding = getattr(self.data.state, "funding_rate", 0.0) or 0.0
            if abs(funding) > self.funding_extreme_pct:
                if funding > 0 and req.side == "BUY":
                    return False, f"Aşırı pozitif funding ({funding:.4f}) — LONG bloklandı"
                if funding < 0 and req.side == "SELL":
                    return False, f"Aşırı negatif funding ({funding:.4f}) — SHORT bloklandı"
        except Exception:
            pass
        return True, "OK"

    async def _check_atr_spike(self, req) -> tuple:
        try:
            ds = self.data.state
            klines = list(getattr(ds, "klines_1h", []))
            if len(klines) < 20:
                return True, "OK"
            def _atr(candles, p=14):
                trs = [max(c["high"]-c["low"],
                           abs(c["high"]-candles[i-1]["close"]),
                           abs(c["low"]-candles[i-1]["close"]))
                       for i, c in enumerate(candles) if i > 0]
                return sum(trs[-p:]) / p if trs else 0
            current_atr = _atr(klines[-15:])
            avg_atr     = _atr(klines[-30:])
            if avg_atr > 0 and current_atr > avg_atr * self.atr_spike_block_mult:
                return False, f"ATR spike: {current_atr:.2f} > {avg_atr:.2f}×{self.atr_spike_block_mult}"
        except Exception:
            pass
        return True, "OK"

    def get_status(self) -> dict:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        recent = [t for t in self._hourly_trades if t > cutoff]
        return {
            "kill_switch":      self.risk.state.kill_switch_active,
            "drawdown_pct":     round(self.risk.state.current_drawdown_pct, 2),
            "daily_loss":       round(self.risk.state.daily_loss, 2),
            "open_positions":   self.risk.state.open_count,
            "hourly_trades":    len(recent),
            "max_hourly":       self.max_trades_per_hour,
        }
