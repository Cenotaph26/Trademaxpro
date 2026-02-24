"""
Strategy Manager — RL agent kararını alır, doğru stratejiyi çalıştırır.

DÜZELTMELER v9:
- Non-BTC sembol için doğru sym_fmt (_fmt_symbol ile)
- handle_signal: quantity USDT'den lot'a doğru çevriliyor
- handle_signal: sl_pct/tp_pct dashboard'dan geliyor ve kullanılıyor
- Manuel işlem: open_market() ile direkt MARKET emir (testnet uyumlu)
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from execution.executor import OrderExecutor, _fmt_symbol
from strategies.dca import DCAStrategy, DCAParams
from strategies.grid import GridStrategy, GridParams
from strategies.smarttrade import SmartTradeStrategy, SmartTradeParams
from risk.engine import RiskMode, TradeRecord
from utils import telegram as tg

logger = logging.getLogger(__name__)


async def _get_symbol_price(exchange, symbol: str, fallback: float = 0.0) -> float:
    """Herhangi bir sembol için anlık fiyatı ccxt üzerinden çeker."""
    try:
        sym_fmt = _fmt_symbol(symbol)
        ticker = await exchange.fetch_ticker(sym_fmt)
        price = float(ticker.get("last") or ticker.get("close") or 0)
        return price if price > 0 else fallback
    except Exception as e:
        logger.warning(f"[{symbol}] Fiyat alınamadı: {e}")
        return fallback


class StrategyManager:
    def __init__(self, data_client, risk_engine, settings):
        self.data = data_client
        self.risk = risk_engine
        self.s = settings
        self.rl_agent = None

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
        İşlem sinyalini işle.
        signal = {symbol, side, quantity(USDT), leverage, sl_pct, tp_pct,
                  order_type, strategy_tag, entry_hint}
        """
        self._ensure_init()
        symbol = signal.get("symbol", self.s.SYMBOL)
        side   = signal.get("side", "BUY").upper()

        # Dashboard'dan gelen parametreler
        usdt_qty    = float(signal.get("quantity", 100))
        leverage    = int(signal.get("leverage", self.s.BASE_LEVERAGE))
        sl_pct      = float(signal.get("sl_pct", self.s.RISK_PER_TRADE_PCT))
        tp_pct      = float(signal.get("tp_pct", sl_pct * 2))
        order_type  = signal.get("order_type", "MARKET").upper()
        entry_hint  = signal.get("entry_hint")

        logger.info(f"📡 Sinyal: {symbol} {side} ${usdt_qty} {leverage}x SL={sl_pct}% TP={tp_pct}% [{signal.get('strategy_tag')}]")

        # ── 1. Signal filters ──────────────────────────────────────
        state = self.data.state
        funding_ok = True
        if abs(state.funding_rate) > self.s.FUNDING_EXTREME_THRESHOLD:
            if state.funding_rate > 0 and side == "BUY":
                funding_ok = False
            elif state.funding_rate < 0 and side == "SELL":
                funding_ok = False
        if not funding_ok:
            msg = f"Aşırı funding rate ({state.funding_rate:.4f}), {side} engellendi"
            logger.warning(msg)
            return {"ok": False, "reason": msg}

        # ── 2. Risk engine check ───────────────────────────────────
        # Önce pozisyon sayılarını güncelle (güncel veri ile)
        await self._update_position_counts(symbol)
        try:
            bal = await self.data.get_balance()
            await self.risk.update_equity(bal.get("total", 0))
        except Exception:
            pass
        can, reason = self.risk.can_trade(side, symbol)
        if not can:
            return {"ok": False, "reason": reason}

        # ── 3. RL agent (manuel bypass) ───────────────────────────
        is_manual = signal.get("strategy_tag", "").startswith("manual")
        decision = None
        if self.rl_agent and not is_manual:
            decision = self.rl_agent.decide()
            if not decision.trade_allowed and self.rl_agent.epsilon < 0.5:
                logger.info(f"🤖 RL engelledi (ε={self.rl_agent.epsilon:.3f})")
                return {"ok": False, "reason": "RL agent: trade_allowed=0"}
            elif not decision.trade_allowed:
                logger.info(f"🤖 RL eğitim aşaması (ε={self.rl_agent.epsilon:.3f}) — bypass")
            logger.info(f"🤖 RL: {decision.strategy} | {decision.risk_mode} | trade={decision.trade_allowed}")

        risk_mode_str = decision.risk_mode if decision else "normal"
        actual_leverage = leverage  # dashboard'dan gelen kaldıracı kullan

        # ── 4. Kaldıraç ayarla ─────────────────────────────────────
        await self.executor.set_leverage(symbol, actual_leverage)

        # ── 5. Sembol fiyatını al ──────────────────────────────────
        # Önce BTC mi kontrol et, değilse anlık fiyat çek
        if symbol.upper() == self.s.SYMBOL.upper():
            mark = state.mark_price
        else:
            mark = await _get_symbol_price(self.data.exchange, symbol, fallback=state.mark_price)
            logger.info(f"[{symbol}] Anlık fiyat: {mark}")

        if mark <= 0:
            return {"ok": False, "reason": f"[{symbol}] Geçerli fiyat alınamadı"}

        # ── 6. Lot miktarını hesapla (USDT → kontrat) ─────────────
        # notional = usdt_qty * leverage
        # quantity (kontrat) = notional / mark_price
        notional = usdt_qty * actual_leverage
        quantity = round(notional / mark, 6)
        if quantity <= 0:
            return {"ok": False, "reason": "Hesaplanan miktar sıfır veya negatif"}
        logger.info(f"[{symbol}] mark={mark:.6g} notional={notional:.2f} qty={quantity:.6g}")

        # ── 7. Manuel işlem: direkt MARKET emir ───────────────────
        if is_manual:
            result = await self.executor.open_market(
                symbol=symbol, side=side, quantity=quantity,
                sl_pct=sl_pct, tp_pct=tp_pct, mark_price=mark,
                strategy_tag=signal.get("strategy_tag", "manual"),
            )
        else:
            # Otomasyon: SmartTrade / DCA / Grid
            strategy = decision.strategy if decision else "SMART"
            if strategy == "DCA":
                params = DCAParams(
                    step_count=self.s.DCA_MAX_STEPS,
                    step_spacing_atr=self.s.DCA_STEP_SPACING_ATR,
                    size_multiplier=self.s.DCA_SIZE_MULTIPLIER,
                    stopout_r=self.s.DCA_STOPOUT_R,
                )
                balance = await self.data.get_balance()
                result = await self.dca.open(symbol, side, params, balance["total"])
            elif strategy == "GRID":
                params = GridParams(
                    grid_levels=self.s.GRID_LEVELS,
                    grid_width_atr=self.s.GRID_WIDTH_ATR,
                )
                result = await self.grid.open(symbol, params, base_qty=quantity)
            else:  # SMART
                atr = state.atr_14
                sl_price = mark * (1 - sl_pct / 100) if side == "BUY" else mark * (1 + sl_pct / 100)
                tp1_price = mark * (1 + tp_pct / 100) if side == "BUY" else mark * (1 - tp_pct / 100)
                tp2_price = mark * (1 + tp_pct * 1.5 / 100) if side == "BUY" else mark * (1 - tp_pct * 1.5 / 100)
                st_params = SmartTradeParams(
                    side=side,
                    entry_price=entry_hint or mark,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    tp1_pct=self.s.ST_TP1_PCT,
                    tp2_pct=self.s.ST_TP2_PCT,
                    trailing=self.s.ST_TRAILING,
                    trailing_callback_pct=self.s.ST_TRAILING_CALLBACK_PCT,
                    quantity=quantity,
                )
                balance = await self.data.get_balance()
                result = await self.smart.open(symbol, st_params, balance["total"])

        # Pozisyon sayılarını güncelle
        await self._update_position_counts(symbol)

        # Telegram bildirimi
        if result is not None:
            try:
                asyncio.create_task(tg.notify_trade_open(
                    symbol=symbol, side=side, qty=usdt_qty,
                    leverage=actual_leverage, entry=mark,
                    sl=mark * (1 - sl_pct / 100) if side == "BUY" else mark * (1 + sl_pct / 100),
                    tp=mark * (1 + tp_pct / 100) if side == "BUY" else mark * (1 - tp_pct / 100),
                    strategy=signal.get("strategy_tag", "manual"),
                ))
            except Exception as e:
                logger.debug(f"Telegram bildirim hatası: {e}")

        ok = result is not None
        logger.info(f"{'✅' if ok else '❌'} İşlem {'açıldı' if ok else 'başarısız'}: {symbol} {side}")
        return {"ok": ok, "symbol": symbol, "side": side, "leverage": actual_leverage}

    async def _update_position_counts(self, symbol: str = ""):
        try:
            all_pos = await self.data.exchange.fetch_positions()
            longs = 0
            shorts = 0
            sym_map = {}
            for p in all_pos:
                contracts = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if abs(contracts) < 1e-9:
                    continue
                psym = p.get("symbol", "")
                direction = "LONG" if contracts > 0 else "SHORT"
                if contracts > 0:
                    longs += 1
                else:
                    shorts += 1
                sym_map[psym] = direction
            self.risk.update_position_counts(longs, shorts, sym_map)
            logger.debug(f"Pozisyon sayıları güncellendi: LONG={longs} SHORT={shorts} toplam={longs+shorts}")
        except Exception as e:
            logger.warning(f"Pozisyon sayısı güncellenemedi: {e}")

    async def close_position(self, symbol: str, side: str = None) -> dict:
        """Tek pozisyonu kapat — Binance'den canlı miktar çeker."""
        self._ensure_init()
        try:
            all_positions = await self.data.exchange.fetch_positions()
            target = None
            for p in all_positions:
                contracts = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
                if abs(contracts) < 1e-9:
                    continue
                psym = p.get("symbol", "")
                psym_clean = psym.replace("/", "").replace(":USDT", "").upper()
                sym_clean  = symbol.replace("/", "").replace(":USDT", "").upper()
                if psym_clean == sym_clean or psym == symbol:
                    target = p
                    break

            if not target:
                return {"ok": False, "reason": f"{symbol} açık pozisyon bulunamadı"}

            contracts = float(target.get("contracts") or 0)
            qty = abs(contracts)
            pos_side = "LONG" if contracts > 0 else "SHORT"

            result = await self.executor.close_position(symbol, pos_side, qty)

            try:
                upnl  = float(target.get("unrealizedPnl") or 0)
                entry = float(target.get("entryPrice") or 0)
                mmark = float(target.get("markPrice") or 0)
                pnl_pct = ((mmark - entry) / entry * 100) if entry > 0 else 0
                asyncio.create_task(tg.notify_trade_close(symbol, pos_side, upnl, pnl_pct))
                # RL: gerçek PnL'e göre reward ver
                if self.rl_agent:
                    try:
                        reward = upnl / 10.0  # normalize: $10 kazanç = +1 reward
                        asyncio.create_task(self.rl_agent.record_outcome(reward, done=True))
                        logger.info(f"🧠 RL reward: {reward:+.3f} (PnL=${upnl:+.2f})")
                    except Exception:
                        pass
            except Exception:
                pass

            return {"ok": result is not None, "symbol": symbol, "qty": qty, "side": pos_side}
        except Exception as e:
            logger.error(f"close_position hatası [{symbol}]: {e}", exc_info=True)
            return {"ok": False, "reason": str(e)}
        finally:
            # Kapatma sonrası sayaçları güncelle
            try:
                await self._update_position_counts()
            except Exception:
                pass

    async def close_all_positions(self):
        """Bot kapatılırken tüm pozisyonları temizle."""
        self._ensure_init()
        try:
            all_pos = await self.data.exchange.fetch_positions()
            for p in all_pos:
                contracts = float(p.get("contracts") or 0)
                if abs(contracts) < 1e-9:
                    continue
                sym = p.get("symbol", self.s.SYMBOL)
                pos_side = "LONG" if contracts > 0 else "SHORT"
                await self.executor.close_position(sym, pos_side, abs(contracts))
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {e}")
