"""
Manuel işlem execution endpoint'leri.
POST /execution/open  — pozisyon aç
POST /execution/close — pozisyon kapat
"""
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class OpenRequest(BaseModel):
    symbol: str
    side: str          # BUY veya SELL
    amount: float      # USDT cinsinden
    leverage: int = 3
    order_type: str = "market"
    sl_pct: Optional[float] = None
    tp_pct: Optional[float] = None


class CloseRequest(BaseModel):
    symbol: str


@router.post("/open")
async def open_position(req: OpenRequest, request: Request):
    """Manuel pozisyon aç."""
    app = request.app
    data = app.state.data_client
    risk = app.state.risk_engine

    if risk.state.kill_switch_active:
        raise HTTPException(status_code=403, detail="Kill switch aktif, işlem yapılamaz")

    try:
        # Kaldıraç ayarla
        await data.exchange.set_leverage(req.leverage, req.symbol)

        # Mevcut fiyatı al
        ticker = await data.exchange.fetch_ticker(req.symbol)
        mark_price = float(ticker.get("last") or ticker.get("close", 0))
        if not mark_price:
            raise HTTPException(status_code=500, detail="Fiyat alınamadı")

        # Miktar hesapla (USDT → coin)
        quantity = round(req.amount * req.leverage / mark_price, 4)

        params = {}
        raw = await data.exchange.create_order(
            symbol=req.symbol,
            type=req.order_type,
            side=req.side,
            amount=quantity,
            params=params,
        )

        avg_price = float(raw.get("average") or raw.get("price") or mark_price)

        # SL/TP emirleri
        close_side = "SELL" if req.side == "BUY" else "BUY"

        if req.sl_pct:
            sl_price = avg_price * (1 - req.sl_pct / 100) if req.side == "BUY" else avg_price * (1 + req.sl_pct / 100)
            sl_price = round(sl_price, 2)
            try:
                await data.exchange.create_order(
                    symbol=req.symbol,
                    type="stop_market",
                    side=close_side,
                    amount=quantity,
                    params={"stopPrice": sl_price, "reduceOnly": True},
                )
            except Exception as e:
                logger.warning(f"SL emri gönderilemedi: {e}")

        if req.tp_pct:
            tp_price = avg_price * (1 + req.tp_pct / 100) if req.side == "BUY" else avg_price * (1 - req.tp_pct / 100)
            tp_price = round(tp_price, 2)
            try:
                await data.exchange.create_order(
                    symbol=req.symbol,
                    type="take_profit_market",
                    side=close_side,
                    amount=quantity,
                    params={"stopPrice": tp_price, "reduceOnly": True},
                )
            except Exception as e:
                logger.warning(f"TP emri gönderilemedi: {e}")

        logger.info(f"✅ Manuel {req.side} {quantity} {req.symbol} @ {avg_price}")
        return {"ok": True, "avg_price": avg_price, "quantity": quantity, "order_id": raw.get("id")}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manuel işlem hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close")
async def close_position(req: CloseRequest, request: Request):
    """Açık pozisyonu kapat."""
    app = request.app
    data = app.state.data_client

    try:
        positions = await data.exchange.fetch_positions([req.symbol])
        pos = next((p for p in positions if p.get("symbol") == req.symbol and float(p.get("contracts", 0)) != 0), None)

        if not pos:
            raise HTTPException(status_code=404, detail=f"{req.symbol} için açık pozisyon yok")

        contracts = float(pos.get("contracts", 0))
        side = pos.get("side", "long")
        close_side = "SELL" if side == "long" else "BUY"

        raw = await data.exchange.create_order(
            symbol=req.symbol,
            type="market",
            side=close_side,
            amount=abs(contracts),
            params={"reduceOnly": True},
        )

        logger.info(f"✅ Pozisyon kapatıldı: {req.symbol}")
        return {"ok": True, "order_id": raw.get("id")}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pozisyon kapatma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))
