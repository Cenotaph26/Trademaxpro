"""
SmartTrade — Tek atış disiplini.
Entry → SL → TP1 → TP2 → Trailing (opsiyon)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SmartTradeParams:
    entry_price: Optional[float] = None   # None = market
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp1_pct: float = 50.0   # pozisyonun %50'si TP1'de
    tp2_pct: float = 50.0
    trailing: bool = True
    trailing_callback_pct: float = 0.5
    side: str = "BUY"
    quantity: float = 0.001


@dataclass
class SmartTrade:
    symbol: str
    params: SmartTradeParams
    order_ids: dict = field(default_factory=dict)
    active: bool = True
    opened_at: datetime = field(default_factory=datetime.utcnow)
    realized_pnl: float = 0.0


class SmartTradeStrategy:
    def __init__(self, executor, risk_engine, data_client, settings):
        self.executor = executor
        self.risk = risk_engine
        self.data = data_client
        self.s = settings
        self.trades: List[SmartTrade] = []

    async def open(self, symbol: str, params: SmartTradeParams, equity: float) -> Optional[SmartTrade]:
        """SmartTrade aç: entry + bracket emirleri."""
        can, reason = self.risk.can_trade(params.side)
        if not can:
            logger.warning(f"SmartTrade engellendi: {reason}")
            return None

        # BTC dışı semboller için anlık fiyatı Binance'den çek
        if symbol == getattr(self.s, "SYMBOL", "BTCUSDT"):
            mark = self.data.state.mark_price
        else:
            try:
                sym_fmt = symbol[:-4] + "/USDT:USDT"
                ticker = await self.data.exchange.fetch_ticker(sym_fmt)
                mark = float(ticker.get("last") or ticker.get("close") or 0)
            except Exception:
                mark = self.data.state.mark_price

        # Auto-calculate SL/TP if not set
        atr = self.data.state.atr_14
        if params.sl_price == 0 and atr > 0:
            if params.side.upper() == "BUY":
                params.sl_price = mark - atr * 1.5
                params.tp1_price = mark + atr * 1.5
                params.tp2_price = mark + atr * 3.0
            else:
                params.sl_price = mark + atr * 1.5
                params.tp1_price = mark - atr * 1.5
                params.tp2_price = mark - atr * 3.0

        # Miktar hesapla
        if params.quantity == 0.001:  # default, risk-based kullan
            params.quantity = self.risk.calculate_position_size(
                equity, mark, params.sl_price, self.s.BASE_LEVERAGE
            )

        trade = SmartTrade(symbol=symbol, params=params)

        # TP pairs
        tp_pairs = [
            (params.tp1_price, params.tp1_pct),
            (params.tp2_price, params.tp2_pct),
        ]

        entry = params.entry_price or mark
        results = await self.executor.place_bracket(
            symbol=symbol,
            side=params.side.upper(),
            quantity=params.quantity,
            entry_price=entry,
            sl_price=params.sl_price,
            tp_prices=tp_pairs,
            trailing=params.trailing,
            trailing_callback_pct=params.trailing_callback_pct,
            strategy_tag="smarttrade",
        )

        trade.order_ids = {
            "entry": results.get("entry", {}) and results["entry"].order_id,
            "sl": results.get("sl", {}) and results["sl"].order_id,
            "tps": [r.order_id for r in results.get("tps", []) if r],
        }

        self.trades.append(trade)
        logger.info(
            f"SmartTrade açıldı: {params.side} {params.quantity} @ {entry:.4f} "
            f"SL={params.sl_price:.4f} TP1={params.tp1_price:.4f}"
        )
        return trade

    def suggest_sl_tp(self, side: str, entry: float, atr: float, risk_mode: str) -> dict:
        """RL agent için SL/TP öneri helper."""
        multipliers = {
            "conservative": (1.0, 1.5, 3.0),
            "normal": (1.5, 2.0, 4.0),
            "aggressive": (2.0, 3.0, 6.0),
        }
        sl_m, tp1_m, tp2_m = multipliers.get(risk_mode, multipliers["normal"])

        if side.upper() == "BUY":
            return {
                "sl": round(entry - atr * sl_m, 4),
                "tp1": round(entry + atr * tp1_m, 4),
                "tp2": round(entry + atr * tp2_m, 4),
            }
        else:
            return {
                "sl": round(entry + atr * sl_m, 4),
                "tp1": round(entry - atr * tp1_m, 4),
                "tp2": round(entry - atr * tp2_m, 4),
            }
