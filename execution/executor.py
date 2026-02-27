"""
Execution Layer v12.2 — Binance Futures SL/TP Düzeltmesi

DÜZELTMELER:
- STOP_MARKET, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET büyük harf (Binance zorunlu)
- workingType: "CONTRACT_PRICE" parametresi eklendi
- SL/TP hataları ana işlemi bloklamıyor
- Lot step_size ve min_notional otomatik düzeltme
"""
import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    order_type: str          # "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"
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
        self._market_info_cache: dict = {}

    async def _get_market_info(self, symbol_ccxt: str) -> dict:
        """Market filtrelerini çek ve cache'le (LOT_SIZE, MIN_NOTIONAL)."""
        if symbol_ccxt in self._market_info_cache:
            return self._market_info_cache[symbol_ccxt]
        try:
            market = self.exchange.market(symbol_ccxt)
            limits    = market.get("limits", {})
            precision = market.get("precision", {})
            amount_prec = precision.get("amount", None)
            step_size   = limits.get("amount", {}).get("step", None)
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
            logger.debug(f"Market info alınamadı [{symbol_ccxt}]: {e}")
            info = {"step_size": 0.001, "min_qty": 0, "min_notional": 5.0, "max_qty": 1e9}
        self._market_info_cache[symbol_ccxt] = info
        return info

    async def adjust_quantity(self, symbol_ccxt: str, quantity: float, price: float) -> tuple:
        """Binance filtrelerine göre lot miktarını düzelt. Döner: (qty, error_or_None)"""
        info = await self._get_market_info(symbol_ccxt)
        qty  = _round_to_step(quantity, info["step_size"])
        if info["min_qty"] > 0 and qty < info["min_qty"]:
            qty = info["min_qty"]
        if price > 0 and qty * price < info["min_notional"]:
            qty = _round_to_step((info["min_notional"] / price) * 1.02, info["step_size"])
            qty = max(qty, info["min_qty"] if info["min_qty"] > 0 else qty)
        if info["max_qty"] > 0 and qty > info["max_qty"]:
            return None, f"Miktar max limiti aşıyor: {qty:.6g} > {info['max_qty']}"
        if qty <= 0:
            return None, "Hesaplanan miktar sıfır"
        logger.debug(f"[{symbol_ccxt}] qty {quantity:.6g}→{qty:.6g} (step={info['step_size']}, notional={qty*price:.2f})")
        return qty, None

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            sym = _fmt_symbol(symbol)
            await self.exchange.set_leverage(leverage, sym)
            logger.info(f"Kaldıraç {leverage}x → {sym}")
        except Exception as e:
            logger.warning(f"Kaldıraç ayarlanamadı [{symbol}]: {e}")

    async def place_order(self, req: OrderRequest, expected_price: Optional[float] = None) -> Optional[OrderResult]:
        """Ana emir gönderme — Binance Futures uyumlu parametreler."""
        req.symbol = _fmt_symbol(req.symbol)

        # Binance Futures parametreleri
        params = {}
        if req.reduce_only:
            params["reduceOnly"] = True

        # STOP_MARKET / TAKE_PROFIT_MARKET için zorunlu parametreler
        order_type_upper = req.order_type.upper()
        if order_type_upper in ("STOP_MARKET", "TAKE_PROFIT_MARKET",
                                 "STOP", "TAKE_PROFIT"):
            if req.stop_price:
                params["stopPrice"] = req.stop_price
                # ccxt bazı versiyonlarda price olarak da isteyebilir
                params["price"] = req.stop_price
            params["workingType"] = "CONTRACT_PRICE"
            params["priceProtect"]  = False   # Binance zorunlu

        elif order_type_upper == "TRAILING_STOP_MARKET":
            if req.callback_rate is not None:
                params["callbackRate"] = req.callback_rate
            params["workingType"] = "CONTRACT_PRICE"

        if req.client_order_id:
            params["newClientOrderId"] = req.client_order_id

        try:
            raw = await self.exchange.create_order(
                symbol=req.symbol,
                type=order_type_upper,      # Büyük harf zorunlu
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
                slippage_pct=round(slippage, 4),
                raw=raw,
            )
            self._order_log.append(result)
            sp_info = f" stopPrice={req.stop_price:.6g}" if req.stop_price else ""
            logger.info(
                f"✅ Emir: {req.side} {req.quantity:.6g} {req.symbol} "
                f"[{order_type_upper}]{sp_info} @ {avg_price:.6g} (slip={slippage:.3f}%) [{req.strategy_tag}]"
            )
            return result

        except Exception as e:
            err_str = str(e)

            # -2022: ReduceOnly rejected — pozisyon zaten kapalı
            if "-2022" in err_str or ("ReduceOnly" in err_str and "rejected" in err_str):
                logger.warning(f"⚠ ReduceOnly rejected [{req.symbol}] — pozisyon kapalı")
                return None

            # -4120: TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET testnet Algo endpoint gerekiyor
            if "-4120" in err_str or "Algo Order" in err_str or "not supported for this endpoint" in err_str:
                fallback_type = None
                fallback_params = {}
                fallback_price = None
                if order_type_upper in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                    fallback_type  = "LIMIT"
                    fallback_price = req.stop_price or req.price
                    fallback_params = {"timeInForce": "GTC", "reduceOnly": True}
                elif order_type_upper == "TRAILING_STOP_MARKET":
                    fallback_type  = "STOP_MARKET"
                    fallback_params = {
                        "stopPrice": req.stop_price or req.price or 0,
                        "workingType": "CONTRACT_PRICE",
                        "reduceOnly": True,
                    }
                if fallback_type and (fallback_price or fallback_params.get("stopPrice")):
                    try:
                        logger.warning(
                            f"⚠ {order_type_upper} desteklenmiyor, {fallback_type} fallback [{req.symbol}]"
                        )
                        raw2 = await self.exchange.create_order(
                            symbol=req.symbol, type=fallback_type,
                            side=req.side, amount=req.quantity,
                            price=fallback_price, params=fallback_params,
                        )
                        logger.info(f"✅ Fallback: {fallback_type} [{req.symbol}]")
                        return OrderResult(
                            order_id=str(raw2.get("id", "")),
                            client_order_id=str(raw2.get("clientOrderId", "")),
                            status=raw2.get("status", ""),
                            filled_qty=float(raw2.get("filled") or 0),
                            avg_price=float(raw2.get("average") or raw2.get("price") or 0),
                            fee=0.0, timestamp=datetime.utcnow(),
                            slippage_pct=0.0, raw=raw2,
                        )
                    except Exception as e2:
                        logger.warning(f"⚠ Fallback başarısız [{req.symbol}]: {e2}")
                        return None
                logger.warning(f"⚠ Desteklenmeyen emir [{order_type_upper}] atlandı")
                return None

            logger.error(
                f"❌ Emir hatası [{req.symbol} {req.side} {order_type_upper} "
                f"qty={req.quantity}]: {type(e).__name__}: {e}"
            )
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
        side_up    = side.upper()
        close_side = "SELL" if side_up in ("BUY", "LONG") else "BUY"
        req = OrderRequest(
            symbol=symbol, side=close_side, order_type="MARKET",
            quantity=quantity, reduce_only=True, strategy_tag="close_position",
        )
        return await self.place_order(req)

    async def open_market(self, symbol: str, side: Side, quantity: float,
                          sl_pct: float = 0.0, tp_pct: float = 0.0,
                          mark_price: float = 0.0,
                          strategy_tag: str = "manual") -> dict:
        """
        MARKET entry + opsiyonel SL/TP.
        SL/TP hataları ana işlemi bloklamaz.
        """
        sym_ccxt = _fmt_symbol(symbol)

        # Lot düzeltme
        adj_qty, err = await self.adjust_quantity(sym_ccxt, quantity, mark_price)
        if err:
            logger.error(f"[{symbol}] Lot hatası: {err}")
            return {"ok": False, "reason": err}

        # Ana MARKET emir
        req    = OrderRequest(symbol=symbol, side=side, order_type="MARKET",
                              quantity=adj_qty, strategy_tag=strategy_tag)
        result = await self.place_order(req, expected_price=mark_price)
        if not result:
            return {"ok": False, "reason": "MARKET emir gönderilemedi"}

        # SL / TP (hata olursa loglayıp devam et)
        sl_side: Side = "SELL" if side == "BUY" else "BUY"

        if sl_pct > 0 and mark_price > 0:
            sl_price = mark_price * (1 - sl_pct / 100) if side == "BUY" else mark_price * (1 + sl_pct / 100)
            try:
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="STOP_MARKET",
                    quantity=adj_qty,
                    stop_price=round(sl_price, 8),  # hassas yuvarlama
                    reduce_only=True,
                    strategy_tag=f"{strategy_tag}_sl",
                ))
            except Exception as e:
                logger.warning(f"[{symbol}] SL emri hatası (işlem açık kaldı): {e}")

        if tp_pct > 0 and mark_price > 0:
            tp_price = mark_price * (1 + tp_pct / 100) if side == "BUY" else mark_price * (1 - tp_pct / 100)
            try:
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=adj_qty,
                    stop_price=round(tp_price, 8),
                    reduce_only=True,
                    strategy_tag=f"{strategy_tag}_tp",
                ))
            except Exception as e:
                logger.warning(f"[{symbol}] TP emri hatası (işlem açık kaldı): {e}")

        return {
            "ok":       True,
            "order_id": result.order_id,
            "avg_price": result.avg_price,
            "quantity":  adj_qty,
            "strategy":  strategy_tag,
        }

    async def place_bracket(self, symbol, side, quantity, entry_price,
                             sl_price, tp_prices, trailing=False,
                             trailing_callback_pct=0.5, strategy_tag="smarttrade") -> dict:
        """SmartTrade bracket: MARKET entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP"""
        sym_ccxt  = _fmt_symbol(symbol)
        adj_qty, err = await self.adjust_quantity(sym_ccxt, quantity, entry_price)
        if err:
            logger.error(f"[{symbol}] Bracket lot hatası: {err}")
            return {"entry": None, "error": err}

        results = {}

        # Entry
        entry_req = OrderRequest(symbol=symbol, side=side, order_type="MARKET",
                                 quantity=adj_qty, strategy_tag=strategy_tag)
        results["entry"] = await self.place_order(entry_req, expected_price=entry_price)
        if not results["entry"]:
            return results

        sl_side: Side = "SELL" if side == "BUY" else "BUY"

        # SL
        try:
            results["sl"] = await self.place_order(OrderRequest(
                symbol=symbol, side=sl_side,
                order_type="STOP_MARKET",
                quantity=adj_qty,
                stop_price=round(sl_price, 6),
                reduce_only=True,
                strategy_tag=f"{strategy_tag}_sl",
            ))
        except Exception as e:
            logger.warning(f"[{symbol}] Bracket SL hatası: {e}")

        # TPs
        results["tps"] = []
        for tp_price, tp_pct_share in tp_prices:
            try:
                tp_qty, _ = await self.adjust_quantity(sym_ccxt, round(adj_qty * tp_pct_share / 100, 6), tp_price)
                if not tp_qty:
                    continue
                r = await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=tp_qty,
                    stop_price=round(tp_price, 8),
                    reduce_only=True,
                    strategy_tag=f"{strategy_tag}_tp",
                ))
                results["tps"].append(r)
            except Exception as e:
                logger.warning(f"[{symbol}] Bracket TP hatası: {e}")

        # Trailing
        if trailing:
            try:
                rem_qty, _ = await self.adjust_quantity(sym_ccxt, round(adj_qty * 0.5, 6), entry_price)
                if rem_qty:
                    results["trailing"] = await self.place_order(OrderRequest(
                        symbol=symbol, side=sl_side,
                        order_type="TRAILING_STOP_MARKET",
                        quantity=rem_qty,
                        callback_rate=trailing_callback_pct,
                        reduce_only=True,
                        strategy_tag=f"{strategy_tag}_trail",
                    ))
            except Exception as e:
                logger.warning(f"[{symbol}] Trailing stop hatası: {e}")

        return results
