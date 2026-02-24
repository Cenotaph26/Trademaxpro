"""
Execution Layer v12 — Lot boyutu ve minimum notional düzeltmeleri.

DÜZELTMELER:
- round_quantity(): Binance lot step_size filtresi (LOT_SIZE)
- Min notional kontrolü (MIN_NOTIONAL): genellikle 5 USDT
- place_order: hata loguna detaylı reason eklendi
"""
import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

OrderType = Literal["market", "limit", "stop_market", "take_profit_market", "trailing_stop_market"]
Side = Literal["BUY", "SELL"]


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    callback_rate: Optional[float] = None
    client_order_id: Optional[str] = None
    strategy_tag: str = ""


@dataclass
class OrderResult:
    order_id: str
    client_order_id: str
    status: str
    filled_qty: float
    avg_price: float
    fee: float
    timestamp: datetime
    slippage_pct: float = 0.0
    raw: dict = None


def _fmt_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "/" in symbol:
        return symbol
    symbol = symbol.replace(":USDT", "")
    for quote in ("USDT", "BUSD", "USDC"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[:-len(quote)]
            if base:
                return f"{base}/{quote}:USDT"
    return symbol


def _round_to_step(value: float, step: float) -> float:
    """Binance lot step_size'a göre yuvarla."""
    if step <= 0:
        return value
    precision = max(0, round(-math.log10(step)))
    rounded = math.floor(value / step) * step
    return round(rounded, precision)


class OrderExecutor:
    def __init__(self, exchange, settings):
        self.exchange = exchange
        self.settings = settings
        self._order_log: list = []
        self._market_info_cache: dict = {}   # symbol → {step_size, min_qty, min_notional}

    async def _get_market_info(self, symbol_ccxt: str) -> dict:
        """Market filtrelerini çek ve cache'le (LOT_SIZE, MIN_NOTIONAL)."""
        if symbol_ccxt in self._market_info_cache:
            return self._market_info_cache[symbol_ccxt]
        try:
            market = self.exchange.market(symbol_ccxt)
            limits  = market.get("limits", {})
            precision = market.get("precision", {})
            amount_prec = precision.get("amount", None)

            step_size = limits.get("amount", {}).get("step", None)
            if step_size is None and amount_prec is not None:
                try:
                    step_size = 10 ** (-int(amount_prec))
                except Exception:
                    step_size = 0.001

            info = {
                "step_size":    float(step_size or 0.001),
                "min_qty":      float(limits.get("amount", {}).get("min") or 0),
                "min_notional": float(limits.get("cost", {}).get("min") or 5.0),
                "max_qty":      float(limits.get("amount", {}).get("max") or 1e9),
            }
        except Exception as e:
            logger.debug(f"Market info alınamadı [{symbol_ccxt}]: {e}, varsayılan kullanılıyor")
            info = {"step_size": 0.001, "min_qty": 0, "min_notional": 5.0, "max_qty": 1e9}

        self._market_info_cache[symbol_ccxt] = info
        return info

    async def adjust_quantity(self, symbol_ccxt: str, quantity: float, price: float) -> tuple:
        """
        Binance filtrelerine göre lot miktarını düzelt.
        Döner: (adjusted_qty, error_msg_or_None)
        """
        info = await self._get_market_info(symbol_ccxt)

        # Step size'a yuvarla
        qty = _round_to_step(quantity, info["step_size"])

        # Min qty
        if info["min_qty"] > 0 and qty < info["min_qty"]:
            qty = info["min_qty"]

        # Min notional kontrolü
        notional = qty * price
        if notional < info["min_notional"] and price > 0:
            # Minimum notional'ı sağlayacak qty hesapla
            min_qty_for_notional = info["min_notional"] / price
            qty = _round_to_step(min_qty_for_notional * 1.01, info["step_size"])  # +%1 buffer
            qty = max(qty, info["min_qty"] if info["min_qty"] > 0 else qty)

        # Max qty
        if info["max_qty"] > 0 and qty > info["max_qty"]:
            return None, f"Miktar max limitini aşıyor: {qty} > {info['max_qty']}"

        if qty <= 0:
            return None, "Hesaplanan miktar sıfır veya negatif"

        final_notional = qty * price
        logger.debug(f"[{symbol_ccxt}] qty={quantity:.6g}→{qty:.6g} notional={final_notional:.2f} step={info['step_size']}")
        return qty, None

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            sym = _fmt_symbol(symbol)
            await self.exchange.set_leverage(leverage, sym)
            logger.info(f"Kaldıraç {leverage}x → {sym}")
        except Exception as e:
            logger.warning(f"Kaldıraç ayarlanamadı [{symbol}]: {e}")

    async def place_order(self, req: OrderRequest, expected_price: Optional[float] = None) -> Optional[OrderResult]:
        req.symbol = _fmt_symbol(req.symbol)
        try:
            params = {"reduceOnly": req.reduce_only}
            if req.stop_price:
                params["stopPrice"] = req.stop_price
            if req.callback_rate is not None:
                params["callbackRate"] = req.callback_rate
            if req.client_order_id:
                params["newClientOrderId"] = req.client_order_id

            raw = await self.exchange.create_order(
                symbol=req.symbol,
                type=req.order_type,
                side=req.side,
                amount=req.quantity,
                price=req.price,
                params=params,
            )

            avg_price = float(raw.get("average") or raw.get("price") or 0)
            filled    = float(raw.get("filled") or raw.get("amount") or 0)
            fee       = float((raw.get("fee") or {}).get("cost", 0))
            slippage  = abs(avg_price - expected_price) / expected_price * 100 if expected_price and avg_price else 0.0

            result = OrderResult(
                order_id=str(raw.get("id", "")),
                client_order_id=str(raw.get("clientOrderId", "")),
                status=raw.get("status", ""),
                filled_qty=filled,
                avg_price=avg_price,
                fee=fee,
                timestamp=datetime.utcnow(),
                slippage_pct=slippage,
                raw=raw,
            )
            self._order_log.append(result)
            logger.info(f"✅ Emir: {req.side} {req.quantity} {req.symbol} @ {avg_price:.6g} (slip={slippage:.3f}%)")
            return result

        except Exception as e:
            logger.error(f"❌ Emir hatası [{req.symbol} {req.side} qty={req.quantity}]: {type(e).__name__}: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str):
        try:
            return await self.exchange.cancel_order(order_id, _fmt_symbol(symbol))
        except Exception as e:
            logger.warning(f"İptal hatası {order_id}: {e}")

    async def cancel_all_orders(self, symbol: str):
        try:
            await self.exchange.cancel_all_orders(_fmt_symbol(symbol))
            logger.info(f"Tüm emirler iptal: {symbol}")
        except Exception as e:
            logger.error(f"Toplu iptal hatası: {e}")

    async def close_position(self, symbol: str, side: str, quantity: float) -> Optional[OrderResult]:
        side_up = side.upper()
        close_side: Side = "SELL" if side_up in ("BUY", "LONG") else "BUY"
        req = OrderRequest(
            symbol=symbol, side=close_side, order_type="market",
            quantity=quantity, reduce_only=True, strategy_tag="close_position",
        )
        return await self.place_order(req)

    async def open_market(self, symbol: str, side: Side, quantity: float,
                          sl_pct: float = 0.0, tp_pct: float = 0.0,
                          mark_price: float = 0.0,
                          strategy_tag: str = "manual") -> dict:
        sym_ccxt = _fmt_symbol(symbol)

        # Lot düzeltme
        adj_qty, err = await self.adjust_quantity(sym_ccxt, quantity, mark_price)
        if err:
            logger.error(f"[{symbol}] Lot hatası: {err}")
            return {"ok": False, "reason": err}

        req = OrderRequest(symbol=symbol, side=side, order_type="market",
                           quantity=adj_qty, strategy_tag=strategy_tag)
        result = await self.place_order(req, expected_price=mark_price)
        if not result:
            return {"ok": False, "reason": "Emir gönderilemedi"}

        sl_side: Side = "SELL" if side == "BUY" else "BUY"
        if mark_price > 0:
            if sl_pct > 0:
                sl_price = mark_price * (1 - sl_pct/100) if side == "BUY" else mark_price * (1 + sl_pct/100)
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side, order_type="stop_market",
                    quantity=adj_qty, stop_price=round(sl_price, 6),
                    reduce_only=True, strategy_tag=f"{strategy_tag}_sl",
                ))
            if tp_pct > 0:
                tp_price = mark_price * (1 + tp_pct/100) if side == "BUY" else mark_price * (1 - tp_pct/100)
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side, order_type="take_profit_market",
                    quantity=adj_qty, stop_price=round(tp_price, 6),
                    reduce_only=True, strategy_tag=f"{strategy_tag}_tp",
                ))

        return {
            "ok": True,
            "order_id": result.order_id,
            "avg_price": result.avg_price,
            "quantity": adj_qty,
            "strategy": strategy_tag,
        }

    async def place_bracket(self, symbol, side, quantity, entry_price,
                             sl_price, tp_prices, trailing=False,
                             trailing_callback_pct=0.5, strategy_tag="smarttrade") -> dict:
        sym_ccxt = _fmt_symbol(symbol)
        adj_qty, err = await self.adjust_quantity(sym_ccxt, quantity, entry_price)
        if err:
            logger.error(f"[{symbol}] Bracket lot hatası: {err}")
            return {"entry": None, "error": err}

        results = {}
        entry_req = OrderRequest(symbol=symbol, side=side, order_type="market",
                                 quantity=adj_qty, strategy_tag=strategy_tag)
        results["entry"] = await self.place_order(entry_req, expected_price=entry_price)

        if not results["entry"]:
            logger.error(f"Entry emri başarısız [{symbol}]")
            return results

        sl_side: Side = "SELL" if side == "BUY" else "BUY"

        results["sl"] = await self.place_order(OrderRequest(
            symbol=symbol, side=sl_side, order_type="stop_market",
            quantity=adj_qty, stop_price=sl_price, reduce_only=True,
            strategy_tag=f"{strategy_tag}_sl",
        ))

        results["tps"] = []
        for tp_price, tp_pct in tp_prices:
            tp_qty, _ = await self.adjust_quantity(sym_ccxt, round(adj_qty * tp_pct / 100, 6), tp_price)
            if not tp_qty:
                continue
            r = await self.place_order(OrderRequest(
                symbol=symbol, side=sl_side, order_type="take_profit_market",
                quantity=tp_qty, stop_price=tp_price, reduce_only=True,
                strategy_tag=f"{strategy_tag}_tp",
            ))
            results["tps"].append(r)

        if trailing:
            rem_qty, _ = await self.adjust_quantity(sym_ccxt, round(adj_qty * 0.5, 6), entry_price)
            if rem_qty:
                results["trailing"] = await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side, order_type="trailing_stop_market",
                    quantity=rem_qty, callback_rate=trailing_callback_pct,
                    reduce_only=True, strategy_tag=f"{strategy_tag}_trail",
                ))

        return results
