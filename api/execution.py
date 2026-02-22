"""
api/execution.py — İşlem açma/kapatma endpoint'leri
DÜZELTME: Hata mesajları artık string olarak döndürülüyor (frontend [object Object] hatası çözüldü)
"""
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


class TradeRequest(BaseModel):
    symbol: str
    side: str                    # "BUY" | "SELL"
    quantity: Optional[float] = None
    leverage: Optional[int] = 3
    order_type: Optional[str] = "MARKET"
    sl_pct: Optional[float] = 1.5
    tp_pct: Optional[float] = 3.0
    strategy_tag: Optional[str] = "manual"


class CloseRequest(BaseModel):
    symbol: str
    side: Optional[str] = None   # None = tümünü kapat


def _safe_error(e: Exception) -> str:
    """Exception'ı güvenli string'e çevir — [object Object] hatasını önler."""
    return str(e) if str(e) else type(e).__name__


@router.post("/open")
async def open_trade(request: Request, body: TradeRequest):
    """Manuel işlem açma endpoint'i."""
    strategy_manager = request.app.state.strategy_manager
    risk_engine = request.app.state.risk_engine

    try:
        # Risk kontrolü
        risk_ok, risk_reason = await risk_engine.check_new_trade(
            symbol=body.symbol,
            side=body.side,
            leverage=body.leverage,
        )
        if not risk_ok:
            return {
                "ok": False,
                "reason": str(risk_reason),   # ← string garantisi
                "symbol": body.symbol,
            }

        # İşlem gönder
        signal_payload = {
            "symbol": body.symbol,
            "side": body.side,
            "strategy_tag": body.strategy_tag,
            "leverage": body.leverage,
            "order_type": body.order_type,
            "sl_pct": body.sl_pct,
            "tp_pct": body.tp_pct,
        }
        if body.quantity:
            signal_payload["quantity"] = body.quantity

        result = await strategy_manager.handle_signal(signal_payload)

        # result her zaman dict dön, string değil
        if isinstance(result, dict):
            return result
        return {"ok": bool(result), "raw": str(result)}

    except Exception as e:
        logger.error(f"open_trade hatası: {e}", exc_info=True)
        return {
            "ok": False,
            "reason": _safe_error(e),   # ← [object Object] yerine string
        }


@router.post("/close")
async def close_trade(request: Request, body: CloseRequest):
    """Pozisyon kapatma endpoint'i."""
    strategy_manager = request.app.state.strategy_manager

    try:
        if body.symbol == "ALL":
            result = await strategy_manager.close_all_positions()
        else:
            result = await strategy_manager.close_position(body.symbol, body.side)

        if isinstance(result, dict):
            return result
        return {"ok": bool(result), "symbol": body.symbol}

    except Exception as e:
        logger.error(f"close_trade hatası: {e}", exc_info=True)
        return {
            "ok": False,
            "reason": _safe_error(e),
        }


@router.get("/positions")
async def get_positions(request: Request):
    """Açık pozisyonları listele."""
    try:
        risk_engine = request.app.state.risk_engine
        positions = risk_engine.get_open_positions()
        return {
            "ok": True,
            "positions": positions if isinstance(positions, list) else [],
            "count": len(positions) if isinstance(positions, list) else 0,
        }
    except Exception as e:
        logger.error(f"get_positions hatası: {e}", exc_info=True)
        return {"ok": False, "reason": _safe_error(e), "positions": []}
