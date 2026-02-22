"""
DCA Strategy — Futures için güvenli DCA.
Max step, ATR-tabanlı aralık, equity-based stop-out.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DCAParams:
    step_count: int = 5           # 2–8
    step_spacing_atr: float = 1.0 # 0.4–1.8
    size_multiplier: float = 1.2  # 1.0–1.6
    stopout_r: float = 1.5        # -1.5R'de sıfırla
    base_qty: float = 0.001


@dataclass
class DCAPosition:
    symbol: str
    side: str
    params: DCAParams
    entries: List[dict] = field(default_factory=list)  # [{price, qty, order_id}]
    total_qty: float = 0.0
    avg_entry: float = 0.0
    step: int = 0
    active: bool = True
    opened_at: datetime = field(default_factory=datetime.utcnow)
    sl_price: float = 0.0
    initial_risk: float = 0.0  # 1R değeri


class DCAStrategy:
    def __init__(self, executor, risk_engine, data_client, settings):
        self.executor = executor
        self.risk = risk_engine
        self.data = data_client
        self.s = settings
        self.positions: List[DCAPosition] = []
        self._running = False

    async def open(self, symbol: str, side: str, params: DCAParams, equity: float) -> Optional[DCAPosition]:
        """İlk DCA girişi."""
        state = self.data.state
        mark = state.mark_price
        atr = state.atr_14

        if mark == 0 or atr == 0:
            logger.warning("DCA: mark/ATR yok, açılmıyor")
            return None

        can, reason = self.risk.can_trade(side)
        if not can:
            logger.warning(f"DCA engellendi: {reason}")
            return None

        # SL ilk adım için
        sl_dist = params.step_spacing_atr * atr
        if side.upper() == "BUY":
            sl_price = mark - sl_dist * params.step_count
        else:
            sl_price = mark + sl_dist * params.step_count

        qty = self.risk.calculate_position_size(equity, mark, sl_price, self.s.BASE_LEVERAGE)

        pos = DCAPosition(
            symbol=symbol,
            side=side.upper(),
            params=params,
            sl_price=sl_price,
            initial_risk=abs(mark - sl_price) * qty,
        )

        # İlk giriş
        from execution.executor import OrderRequest
        req = OrderRequest(
            symbol=symbol,
            side=side.upper(),
            order_type="market",
            quantity=qty,
            strategy_tag="dca_step_0",
        )
        result = await self.executor.place_order(req, expected_price=mark)
        if result and result.avg_price > 0:
            pos.entries.append({"price": result.avg_price, "qty": qty, "step": 0})
            pos.total_qty = qty
            pos.avg_entry = result.avg_price
            pos.step = 1
            self.positions.append(pos)
            logger.info(f"DCA açıldı: {side} {qty} @ {result.avg_price:.4f}")
            return pos
        return None

    async def _maybe_add_step(self, pos: DCAPosition):
        """Fiyat ATR kadar ters giderse yeni step ekle."""
        if not pos.active or pos.step >= pos.params.step_count:
            return
        state = self.data.state
        mark = state.mark_price
        atr = state.atr_14
        step_dist = pos.params.step_spacing_atr * atr

        last_entry = pos.entries[-1]["price"]
        if pos.side == "BUY":
            target = last_entry - step_dist
            if mark > target:
                return
        else:
            target = last_entry + step_dist
            if mark < target:
                return

        can, reason = self.risk.can_trade(pos.side)
        if not can:
            return

        # Büyüyen miktar
        new_qty = round(pos.entries[-1]["qty"] * pos.params.size_multiplier, 4)

        from execution.executor import OrderRequest
        req = OrderRequest(
            symbol=pos.symbol,
            side=pos.side,
            order_type="market",
            quantity=new_qty,
            strategy_tag=f"dca_step_{pos.step}",
        )
        result = await self.executor.place_order(req, expected_price=mark)
        if result and result.avg_price > 0:
            pos.entries.append({"price": result.avg_price, "qty": new_qty, "step": pos.step})
            pos.total_qty += new_qty
            # Yeni avg entry
            total_cost = sum(e["price"] * e["qty"] for e in pos.entries)
            pos.avg_entry = total_cost / pos.total_qty
            pos.step += 1
            logger.info(f"DCA step {pos.step}: +{new_qty} @ {result.avg_price:.4f} | avg={pos.avg_entry:.4f}")

    async def _check_stopout(self, pos: DCAPosition):
        """Equity-based stop-out: unrealized kayıp > N * initial_risk."""
        if not pos.active:
            return
        mark = self.data.state.mark_price
        if pos.side == "BUY":
            unrealized = (mark - pos.avg_entry) * pos.total_qty
        else:
            unrealized = (pos.avg_entry - mark) * pos.total_qty

        # Stop-out: kayıp initial_risk * stopout_r geçerse
        if unrealized < -(pos.initial_risk * pos.params.stopout_r):
            logger.warning(f"DCA STOP-OUT: PnL={unrealized:.2f}, limit={-pos.initial_risk * pos.params.stopout_r:.2f}")
            await self.executor.close_position(pos.symbol, pos.side, pos.total_qty)
            pos.active = False
            from risk.engine import TradeRecord
            await self.risk.record_trade(TradeRecord(
                pnl=unrealized,
                timestamp=datetime.utcnow(),
                side=pos.side,
                strategy="dca",
            ))

    async def monitor_all(self):
        """Her 5 saniyede tüm DCA pozisyonlarını kontrol et."""
        self._running = True
        while self._running:
            for pos in [p for p in self.positions if p.active]:
                await self._check_stopout(pos)
                await self._maybe_add_step(pos)
            # Kapalıları temizle
            self.positions = [p for p in self.positions if p.active]
            await asyncio.sleep(5)

    def stop(self):
        self._running = False
