"""
Risk Engine v12 — Kalıcı Trade Geçmişi.
Tüm trade kayıtları artık SQLite'e yazılıyor.
Restart sonrası istatistikler sıfırlanmıyor.
"""
import asyncio
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from persistence import save_trade, load_trades, update_daily_stats

logger = logging.getLogger(__name__)


class RiskMode(str, Enum):
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


class KillSwitchReason(str, Enum):
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSS = "consecutive_loss"
    SLIPPAGE = "slippage"
    CONNECTION = "connection"
    MANUAL = "manual"


@dataclass
class TradeRecord:
    pnl: float
    timestamp: datetime
    side: str
    strategy: str
    slippage_pct: float = 0.0
    symbol: str = ""


@dataclass
class RiskState:
    kill_switch_active: bool = False
    kill_switch_reason: Optional[KillSwitchReason] = None
    daily_loss: float = 0.0
    daily_date: date = field(default_factory=date.today)
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    trade_history: List[TradeRecord] = field(default_factory=list)
    long_count: int = 0
    short_count: int = 0
    open_count: int = 0
    current_mode: RiskMode = RiskMode.NORMAL
    open_symbols: dict = field(default_factory=dict)


class RiskEngine:
    def __init__(self, settings):
        self.s = settings
        self.state = RiskState()
        self._lock = asyncio.Lock()
        self._load_history_from_db()

    def _load_history_from_db(self):
        """Başlangıçta son 200 trade'i DB'den yükle."""
        try:
            rows = load_trades(limit=200)
            for r in reversed(rows):  # eskiden yeniye sırala
                self.state.trade_history.append(TradeRecord(
                    pnl=float(r.get("pnl", 0)),
                    timestamp=datetime.fromisoformat(r["ts"]) if r.get("ts") else datetime.utcnow(),
                    side=r.get("side", ""),
                    strategy=r.get("strategy", ""),
                    slippage_pct=float(r.get("slippage_pct", 0)),
                    symbol=r.get("symbol", ""),
                ))
            if self.state.trade_history:
                logger.info(f"📊 {len(self.state.trade_history)} trade DB'den yüklendi")
        except Exception as e:
            logger.warning(f"Trade geçmişi yüklenemedi: {e}")

    # ─── Kill switch ───────────────────────────────────────────────

    def activate_kill_switch(self, reason: KillSwitchReason):
        if not self.state.kill_switch_active:
            self.state.kill_switch_active = True
            self.state.kill_switch_reason = reason
            logger.critical(f"🚨 KILL SWITCH AKTİF: {reason.value}")

    def deactivate_kill_switch(self):
        self.state.kill_switch_active = False
        self.state.kill_switch_reason = None
        logger.warning("Kill switch devre dışı bırakıldı (manuel)")

    # ─── Pre-trade checks ──────────────────────────────────────────

    def can_trade(self, side: str, symbol: str = "") -> tuple:
        s = self.state
        if s.kill_switch_active:
            return False, f"Kill switch aktif: {s.kill_switch_reason.value}"

        if s.daily_date != date.today():
            s.daily_loss = 0.0
            s.daily_date = date.today()
            s.consecutive_losses = 0

        if s.current_equity > 0:
            daily_loss_pct = (s.daily_loss / s.current_equity) * 100
            if daily_loss_pct >= self.s.DAILY_MAX_LOSS_PCT:
                return False, f"Günlük max kayıp aşıldı: {daily_loss_pct:.2f}%"

        max_pos = getattr(self.s, "MAX_OPEN_POSITIONS", 10)
        if s.open_count >= max_pos:
            return False, f"Max açık pozisyon: {s.open_count}/{max_pos}"

        if s.current_drawdown_pct >= self.s.MAX_DRAWDOWN_PCT:
            return False, f"Max drawdown aşıldı: {s.current_drawdown_pct:.2f}%"

        return True, "OK"

    def get_leverage(self, mode: RiskMode, atr_pct: float = 0.0) -> int:
        base = {
            RiskMode.CONSERVATIVE: self.s.LEV_CONSERVATIVE,
            RiskMode.NORMAL: self.s.LEV_NORMAL,
            RiskMode.AGGRESSIVE: self.s.LEV_AGGRESSIVE,
        }[mode]
        if atr_pct > 0.03:    return max(1, base - 2)
        elif atr_pct > 0.015: return max(1, base - 1)
        return base

    def calculate_position_size(self, equity, entry, stop_loss, leverage):
        risk_amount = equity * (self.s.RISK_PER_TRADE_PCT / 100)
        sl_distance_pct = abs(entry - stop_loss) / entry if entry > 0 else 0.015
        if sl_distance_pct < 0.001:
            sl_distance_pct = 0.001
        notional = risk_amount / sl_distance_pct
        return round(notional / entry if entry > 0 else 0, 6)

    # ─── Post-trade update ─────────────────────────────────────────

    async def record_trade(self, record: TradeRecord):
        async with self._lock:
            self.state.trade_history.append(record)

            if record.pnl < 0:
                self.state.daily_loss += abs(record.pnl)
                self.state.consecutive_losses += 1
            else:
                self.state.consecutive_losses = 0

            if self.state.consecutive_losses >= self.s.KILL_SWITCH_CONSECUTIVE_LOSS:
                self.activate_kill_switch(KillSwitchReason.CONSECUTIVE_LOSS)

            if record.slippage_pct > self.s.KILL_SWITCH_SLIPPAGE_PCT:
                self.activate_kill_switch(KillSwitchReason.SLIPPAGE)

            # ── Kalıcı kayıt ──
            try:
                save_trade(
                    pnl=record.pnl,
                    side=record.side,
                    strategy=record.strategy,
                    slippage_pct=record.slippage_pct,
                    symbol=getattr(record, "symbol", ""),
                )
                update_daily_stats(
                    pnl=record.pnl,
                    is_win=record.pnl > 0,
                    drawdown_pct=self.state.current_drawdown_pct,
                    peak_equity=self.state.peak_equity,
                )
            except Exception as e:
                logger.warning(f"Trade DB kayıt hatası: {e}")

    async def update_equity(self, equity: float):
        async with self._lock:
            self.state.current_equity = equity
            if equity > self.state.peak_equity:
                self.state.peak_equity = equity
            if self.state.peak_equity > 0:
                self.state.current_drawdown_pct = (
                    (self.state.peak_equity - equity) / self.state.peak_equity * 100
                )
            if self.state.current_drawdown_pct >= self.s.MAX_DRAWDOWN_PCT:
                self.activate_kill_switch(KillSwitchReason.MAX_DRAWDOWN)

    def update_position_counts(self, long_count, short_count, open_symbols=None):
        self.state.long_count = long_count
        self.state.short_count = short_count
        self.state.open_count = long_count + short_count
        if open_symbols is not None:
            self.state.open_symbols = open_symbols

    # ─── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        h = self.state.trade_history
        if not h:
            return {"winrate": 0, "expectancy": 0, "sharpe": 0, "trade_count": 0}
        wins   = [t for t in h if t.pnl > 0]
        losses = [t for t in h if t.pnl <= 0]
        winrate  = len(wins) / len(h)
        avg_win  = sum(t.pnl for t in wins) / (len(wins) or 1)
        avg_loss = abs(sum(t.pnl for t in losses) / (len(losses) or 1))
        expectancy = winrate * avg_win - (1 - winrate) * avg_loss
        pnls = [t.pnl for t in h[-50:]]
        if len(pnls) > 1:
            mean_p = sum(pnls) / len(pnls)
            std_p  = (sum((p - mean_p) ** 2 for p in pnls) / len(pnls)) ** 0.5
            sharpe = mean_p / (std_p + 1e-9) * (252 ** 0.5)
        else:
            sharpe = 0
        return {
            "winrate": round(winrate, 3),
            "expectancy": round(expectancy, 4),
            "sharpe": round(sharpe, 3),
            "trade_count": len(h),
            "consecutive_losses": self.state.consecutive_losses,
            "daily_loss": round(self.state.daily_loss, 2),
            "drawdown_pct": round(self.state.current_drawdown_pct, 2),
        }

    async def monitor_loop(self):
        logger.info("Risk monitor başlatıldı")
        while True:
            await asyncio.sleep(30)
            try:
                stats = self.get_stats()
                if self.state.kill_switch_active:
                    logger.warning(f"⛔ Kill switch aktif: {self.state.kill_switch_reason}")
            except Exception as e:
                logger.error(f"Risk monitor hatası: {e}")
