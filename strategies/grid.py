"""
Grid Strategy — Futures için range-only grid.
Trend algılanırsa otomatik kapanır.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GridParams:
    grid_levels: int = 20          # 10–40
    grid_width_atr: float = 3.0    # 1.5–6
    take_profit_style: str = "mean_revert"  # mean_revert / partial


@dataclass
class GridLevel:
    price: float
    side: str         # BUY or SELL
    quantity: float
    order_id: str = ""
    filled: bool = False
    fill_price: float = 0.0


@dataclass
class GridPosition:
    symbol: str
    params: GridParams
    center_price: float
    upper: float
    lower: float
    levels: List[GridLevel] = field(default_factory=list)
    active: bool = True
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)


class GridStrategy:
    def __init__(self, executor, risk_engine, data_client, settings):
        self.executor = executor
        self.risk = risk_engine
        self.data = data_client
        self.s = settings
        self.grids: List[GridPosition] = []
        self._running = False

    def _is_range_regime(self) -> bool:
        return self.data.state.regime == "range"

    async def open(self, symbol: str, params: GridParams, base_qty: float) -> Optional[GridPosition]:
        """Grid grid aç (sadece range rejimde)."""
        if not self._is_range_regime():
            logger.info("Grid: trend/volatile rejim, grid açılmıyor")
            return None

        state = self.data.state
        mark = state.mark_price
        atr = state.atr_14
        if mark == 0 or atr == 0:
            return None

        half_width = (params.grid_width_atr * atr) / 2
        upper = mark + half_width
        lower = mark - half_width
        step = (upper - lower) / params.grid_levels

        grid = GridPosition(
            symbol=symbol,
            params=params,
            center_price=mark,
            upper=upper,
            lower=lower,
        )

        # Seviyeleri oluştur: merkezin altı BUY, üstü SELL
        levels_placed = 0
        for i in range(params.grid_levels):
            price = lower + i * step
            side = "BUY" if price < mark else "SELL"
            can, _ = self.risk.can_trade(side)
            if not can:
                continue

            from execution.executor import OrderRequest
            req = OrderRequest(
                symbol=symbol,
                side=side,
                order_type="limit",
                quantity=base_qty,
                price=round(price, 2),
                strategy_tag=f"grid_level_{i}",
            )
            result = await self.executor.place_order(req, expected_price=price)
            if result:
                grid.levels.append(GridLevel(
                    price=round(price, 2),
                    side=side,
                    quantity=base_qty,
                    order_id=result.order_id,
                ))
                levels_placed += 1

        if levels_placed > 0:
            self.grids.append(grid)
            logger.info(f"Grid açıldı: {levels_placed} seviye | {lower:.2f}–{upper:.2f}")
            return grid
        return None

    async def _check_trend_exit(self, grid: GridPosition):
        """Trend başlarsa grid kapat."""
        if not self._is_range_regime():
            logger.warning("Grid: Trend başladı, grid kapatılıyor!")
            await self._close_grid(grid)

    async def _close_grid(self, grid: GridPosition):
        """Tüm açık grid emirlerini iptal et."""
        grid.active = False
        await self.executor.cancel_all_orders(grid.symbol)
        logger.info(f"Grid kapatıldı. Realized PnL: {grid.realized_pnl:.4f}")

    async def _check_fills(self, grid: GridPosition):
        """Dolmuş levelleri tespit et, karşı emir koy."""
        # Gerçek uygulamada exchange'den order status çek
        # Burada mark price bazlı simulate
        mark = self.data.state.mark_price
        for level in grid.levels:
            if level.filled:
                continue
            # BUY level dolduysa (mark geçti)
            if level.side == "BUY" and mark <= level.price:
                level.filled = True
                level.fill_price = level.price
                # Take profit emri
                tp_price = round(level.price * (1 + 0.003), 2)  # %0.3 yukarı
                from execution.executor import OrderRequest
                tp_req = OrderRequest(
                    symbol=grid.symbol,
                    side="SELL",
                    order_type="limit",
                    quantity=level.quantity,
                    price=tp_price,
                    reduce_only=True,
                    strategy_tag="grid_tp",
                )
                await self.executor.place_order(tp_req, expected_price=tp_price)
                logger.debug(f"Grid fill: BUY @ {level.price:.2f}, TP @ {tp_price:.2f}")

            elif level.side == "SELL" and mark >= level.price:
                level.filled = True
                level.fill_price = level.price
                tp_price = round(level.price * (1 - 0.003), 2)
                from execution.executor import OrderRequest
                tp_req = OrderRequest(
                    symbol=grid.symbol,
                    side="BUY",
                    order_type="limit",
                    quantity=level.quantity,
                    price=tp_price,
                    reduce_only=True,
                    strategy_tag="grid_tp",
                )
                await self.executor.place_order(tp_req, expected_price=tp_price)

    async def monitor_all(self):
        self._running = True
        while self._running:
            for grid in [g for g in self.grids if g.active]:
                await self._check_trend_exit(grid)
                if grid.active:
                    await self._check_fills(grid)
            self.grids = [g for g in self.grids if g.active]
            await asyncio.sleep(5)

    def stop(self):
        self._running = False
