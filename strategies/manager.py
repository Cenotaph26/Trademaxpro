"""
Strategy Manager — RL agent kararını alır, doğru stratejiyi çalıştırır.
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from execution.executor import OrderExecutor
from strategies.dca import DCAStrategy, DCAParams
from strategies.grid import GridStrategy, GridParams
from strategies.smarttrade import SmartTradeStrategy, SmartTradeParams
from risk.engine import RiskMode, TradeRecord

logger = logging.getLogger(__name__)


class StrategyManager:
    def __init__(self, data_client, risk_engine, settings):
        self.data = data_client
        self.risk = risk_engine
        self.s = settings
        self.rl_agent = None

        # Executor başlatmayı data_client connect sonrasına bırak
        self.executor: Optional[OrderExecutor] = None
        self.dca: Optional[DCAStrategy] = None
        self.grid: Optional[GridStrategy] = None
        self.smart: Optional[SmartTradeStrategy] = None
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized and self.data.exchange:
            self.executor = OrderExecutor(self.data.exchange, self.s)
            self.dca = DCAStrategy(self.executor, self.risk, self.data, self.s)
            self.grid = GridStrategy(self.executor, self.risk, self.data, self.s)
            self.smart = SmartTradeStrategy(self.executor, self.risk, self.data, self.s)
            self._initialized = True
            asyncio.create_task(self.dca.monitor_all())
            asyncio.create_task(self.grid.monitor_all())

    def set_rl_agent(self, agent):
        self.rl_agent = agent
        if agent:
            agent.set_dependencies(self.data, self.risk)

    async def handle_signal(self, signal: dict) -> dict:
        """
        TradingView webhook sinyalini işle.
        signal = {symbol, side, timeframe, strategy_tag, entry_hint}
        """
        self._ensure_init()
        symbol = signal.get("symbol", self.s.SYMBOL)
        side = signal.get("side", "BUY").upper()
        entry_hint = signal.get("entry_hint")

        logger.info(f"📡 Sinyal alındı: {symbol} {side} [{signal.get('strategy_tag')}]")

        # ── 1. Signal filters ─────────────────────────────────────────
        state = self.data.state
        issues = []

        # Spread filtresi
        if state.spread_pct > 0.05:
            issues.append(f"Spread yüksek: {state.spread_pct:.3f}%")

        # Funding rate filtresi (aşırı funding varsa o yönde girme)
        if abs(state.funding_rate) > self.s.FUNDING_EXTREME_THRESHOLD:
            # Yüksek pozitif funding → long'lar zarar eder → long'u engelle
            if state.funding_rate > 0 and side == "BUY":
                issues.append(f"Aşırı pozitif funding ({state.funding_rate:.4f}), BUY engellendi")
            elif state.funding_rate < 0 and side == "SELL":
                issues.append(f"Aşırı negatif funding ({state.funding_rate:.4f}), SELL engellendi")

        if issues:
            logger.warning(f"Sinyal filtreden geçemedi: {issues}")
            return {"ok": False, "reason": issues}

        # ── 2. Risk engine check ──────────────────────────────────────
        can, reason = self.risk.can_trade(side)
        if not can:
            return {"ok": False, "reason": reason}

        # ── 3. RL agent kararı ────────────────────────────────────────
        decision = None
        if self.rl_agent:
            decision = self.rl_agent.decide()
            logger.info(
                f"🤖 RL Kararı: {decision.strategy} | {decision.risk_mode} | "
                f"leverage≤{decision.leverage_cap}x | trade={decision.trade_allowed}"
            )
            if not decision.trade_allowed:
                return {"ok": False, "reason": "RL agent: trade_allowed=0"}

        strategy = decision.strategy if decision else "SMART"
        risk_mode_str = decision.risk_mode if decision else "normal"
        leverage = decision.leverage_cap if decision else self.s.BASE_LEVERAGE

        risk_mode = RiskMode(risk_mode_str)

        # Kaldıraç ayarla
        atr_pct = state.atr_14 / (state.mark_price + 1e-9)
        actual_leverage = self.risk.get_leverage(risk_mode, atr_pct)
        actual_leverage = min(actual_leverage, leverage)
        await self.executor.set_leverage(symbol, actual_leverage)

        # Bakiye
        balance = await self.data.get_balance()
        equity = balance["total"]

        # ── 4. Execution ──────────────────────────────────────────────
        if strategy == "DCA":
            params = DCAParams(
                step_count=self.s.DCA_MAX_STEPS,
                step_spacing_atr=self.s.DCA_STEP_SPACING_ATR,
                size_multiplier=self.s.DCA_SIZE_MULTIPLIER,
                stopout_r=self.s.DCA_STOPOUT_R,
            )
            result = await self.dca.open(symbol, side, params, equity)

        elif strategy == "GRID":
            params = GridParams(
                grid_levels=self.s.GRID_LEVELS,
                grid_width_atr=self.s.GRID_WIDTH_ATR,
            )
            result = await self.grid.open(symbol, params, base_qty=0.001)

        else:  # SMART
            atr = state.atr_14
            mark = state.mark_price
            st_params = SmartTradeParams(
                side=side,
                entry_price=entry_hint or mark,
            )
            # SL/TP önerisi
            suggestion = self.smart.suggest_sl_tp(side, entry_hint or mark, atr, risk_mode_str)
            st_params.sl_price = suggestion["sl"]
            st_params.tp1_price = suggestion["tp1"]
            st_params.tp2_price = suggestion["tp2"]
            st_params.tp1_pct = self.s.ST_TP1_PCT
            st_params.tp2_pct = self.s.ST_TP2_PCT
            st_params.trailing = self.s.ST_TRAILING
            st_params.trailing_callback_pct = self.s.ST_TRAILING_CALLBACK_PCT
            result = await self.smart.open(symbol, st_params, equity)

        # Pozisyon sayılarını güncelle
        await self._update_position_counts(symbol)

        return {
            "ok": result is not None,
            "strategy": strategy,
            "risk_mode": risk_mode_str,
            "leverage": actual_leverage,
        }

    async def _update_position_counts(self, symbol: str):
        try:
            positions = await self.data.get_positions()
            longs = sum(1 for p in positions if (p.get("side") or "").upper() == "LONG")
            shorts = sum(1 for p in positions if (p.get("side") or "").upper() == "SHORT")
            self.risk.update_position_counts(longs, shorts)
        except Exception as e:
            logger.warning(f"Pozisyon sayısı güncellenemedi: {e}")

    async def close_all_positions(self):
        """Bot kapatılırken tüm pozisyonları temizle."""
        self._ensure_init()
        try:
            positions = await self.data.get_positions()
            for p in positions:
                symbol = p.get("symbol", self.s.SYMBOL)
                side = p.get("side", "LONG")
                qty = abs(float(p.get("contracts") or 0))
                if qty > 0:
                    close_side = "SELL" if side.upper() == "LONG" else "BUY"
                    await self.executor.close_position(symbol, close_side, qty)
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {e}")
