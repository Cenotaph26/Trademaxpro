"""
Execution Layer — Tüm emirler buradan geçer.
market/limit, reduceOnly, SL, TP, trailing stop.

DÜZELTMELER v9:
- _fmt_symbol: tüm uzunluktaki semboller için doğru çalışır (1000BONKUSDT vb.)
- place_order: docstring yeri düzeltildi (req.symbol ataması sonrasına alındı)
- place_bracket: entry artık MARKET order (testnet uyumlu)
"""
import asyncio
import logging
from dataclasses import dataclass
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
    """
    BTCUSDT       → BTC/USDT:USDT
    1000BONKUSDT  → 1000BONK/USDT:USDT
    ETH/USDT:USDT → ETH/USDT:USDT  (already ok)
    """
    symbol = symbol.strip().upper()
    if "/" in symbol:
        return symbol
    # Strip :USDT suffix if mistakenly added
    symbol = symbol.replace(":USDT", "")
    # Try quote currencies longest first to avoid partial matches
    for quote in ("USDT", "BUSD", "USDC"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if base:  # safety check
                return f"{base}/{quote}:USDT"
    return symbol


class OrderExecutor:
    def __init__(self, exchange, settings):
        self.exchange = exchange
        self.settings = settings
        self._order_log: list = []

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            sym = _fmt_symbol(symbol)
            await self.exchange.set_leverage(leverage, sym)
            logger.info(f"Kaldıraç {leverage}x → {sym}")
        except Exception as e:
            logger.warning(f"Kaldıraç ayarlanamadı [{symbol}]: {e}")

    async def place_order(self, req: OrderRequest, expected_price: Optional[float] = None) -> Optional[OrderResult]:
        """Ana emir gönderme. Slippage hesaplar, loglar."""
        # Symbol formatını ccxt futures formatına çevir
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
            filled = float(raw.get("filled") or raw.get("amount") or 0)
            fee = float((raw.get("fee") or {}).get("cost", 0))

            slippage = 0.0
            if expected_price and avg_price:
                slippage = abs(avg_price - expected_price) / expected_price * 100

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
            logger.info(
                f"✅ Emir: {req.side} {req.quantity} {req.symbol} "
                f"@ {avg_price:.6g} (slip={slippage:.3f}%) [{req.strategy_tag}]"
            )
            return result

        except Exception as e:
            logger.error(f"❌ Emir hatası [{req.symbol} {req.side}]: {e}")
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
        """
        Pozisyonu market ile kapat.
        side = mevcut pozisyon yönü: LONG / SHORT / BUY / SELL
        """
        side_up = side.upper()
        close_side: Side = "SELL" if side_up in ("BUY", "LONG") else "BUY"
        req = OrderRequest(
            symbol=symbol,
            side=close_side,
            order_type="market",
            quantity=quantity,
            reduce_only=True,
            strategy_tag="close_position",
        )
        return await self.place_order(req)

    async def open_market(self, symbol: str, side: Side, quantity: float,
                          sl_pct: float = 0.0, tp_pct: float = 0.0,
                          mark_price: float = 0.0,
                          strategy_tag: str = "manual") -> Optional[OrderResult]:
        """
        Basit MARKET emir — testnet dahil her ortamda çalışır.
        sl_pct / tp_pct > 0 ise SL/TP emirleri de gönderilir.
        """
        req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            strategy_tag=strategy_tag,
        )
        result = await self.place_order(req)
        if result and mark_price > 0:
            sl_side: Side = "SELL" if side == "BUY" else "BUY"
            if sl_pct > 0:
                sl_price = mark_price * (1 - sl_pct / 100) if side == "BUY" else mark_price * (1 + sl_pct / 100)
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="stop_market", quantity=quantity,
                    stop_price=round(sl_price, 6), reduce_only=True,
                    strategy_tag=f"{strategy_tag}_sl",
                ))
            if tp_pct > 0:
                tp_price = mark_price * (1 + tp_pct / 100) if side == "BUY" else mark_price * (1 - tp_pct / 100)
                await self.place_order(OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="take_profit_market", quantity=quantity,
                    stop_price=round(tp_price, 6), reduce_only=True,
                    strategy_tag=f"{strategy_tag}_tp",
                ))
        return result

    async def place_bracket(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        entry_price: float,
        sl_price: float,
        tp_prices: list,
        trailing: bool = False,
        trailing_callback_pct: float = 0.5,
        strategy_tag: str = "smarttrade",
    ) -> dict:
        """
        SmartTrade bracket: MARKET entry + SL + TP1 + TP2 (+ trailing)
        Entry artık MARKET — testnet uyumlu.
        """
        results = {}

        # Entry — MARKET (testnet'te LIMIT çalışmayabilir)
        entry_req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            strategy_tag=strategy_tag,
        )
        results["entry"] = await self.place_order(entry_req, expected_price=entry_price)

        if not results["entry"]:
            logger.error(f"Entry emri başarısız, SL/TP gönderilmiyor [{symbol}]")
            return results

        sl_side: Side = "SELL" if side == "BUY" else "BUY"

        # SL
        sl_req = OrderRequest(
            symbol=symbol, side=sl_side,
            order_type="stop_market", quantity=quantity,
            stop_price=sl_price, reduce_only=True,
            strategy_tag=f"{strategy_tag}_sl",
        )
        results["sl"] = await self.place_order(sl_req)

        # TPs
        results["tps"] = []
        for tp_price, tp_pct in tp_prices:
            tp_qty = round(quantity * tp_pct / 100, 6)
            if tp_qty <= 0:
                continue
            tp_req = OrderRequest(
                symbol=symbol, side=sl_side,
                order_type="take_profit_market", quantity=tp_qty,
                stop_price=tp_price, reduce_only=True,
                strategy_tag=f"{strategy_tag}_tp",
            )
            r = await self.place_order(tp_req)
            results["tps"].append(r)

        # Trailing
        if trailing:
            remaining_qty = round(quantity * 0.5, 6)
            if remaining_qty > 0:
                trail_req = OrderRequest(
                    symbol=symbol, side=sl_side,
                    order_type="trailing_stop_market", quantity=remaining_qty,
                    callback_rate=trailing_callback_pct, reduce_only=True,
                    strategy_tag=f"{strategy_tag}_trail",
                )
                results["trailing"] = await self.place_order(trail_req)

        return results
