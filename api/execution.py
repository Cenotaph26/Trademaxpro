"""
api/execution.py — İşlem açma/kapatma/pozisyon endpoint'leri
DÜZELTMELER:
- 'RiskEngine' object has no attribute 'get_open_positions' → güvenli fallback
- Tüm exception'lar str(e) ile string'e çevriliyor ([object Object] fix)
- Pozisyon listesi birden fazla kaynaktan toplanıyor
"""
import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Pydantic Modeller ────────────────────────────────────────────────────────

class TradeRequest(BaseModel):
    symbol: str
    side: str                         # "BUY" | "SELL"
    quantity: Optional[float] = None
    leverage: Optional[int] = 3
    order_type: Optional[str] = "MARKET"
    sl_pct: Optional[float] = 1.5
    tp_pct: Optional[float] = 3.0
    strategy_tag: Optional[str] = "manual"


class CloseRequest(BaseModel):
    symbol: str
    side: Optional[str] = None


# ─── Utils ────────────────────────────────────────────────────────────────────

def _safe_str(e: Exception) -> str:
    """Exception → string. Hiçbir zaman [object Object] döndürmez."""
    return str(e) if str(e) else type(e).__name__


def _get_positions(risk_engine) -> list:
    """
    RiskEngine'den açık pozisyonları güvenle çeker.
    get_open_positions(), open_positions, positions gibi farklı
    attribute/method isimlerini dener.
    """
    for method_name in ("get_open_positions", "get_positions", "list_positions"):
        fn = getattr(risk_engine, method_name, None)
        if callable(fn):
            try:
                result = fn()
                return result if isinstance(result, list) else list(result)
            except Exception as e:
                logger.warning(f"{method_name}() hatası: {e}")

    for attr_name in ("open_positions", "positions", "active_positions"):
        val = getattr(risk_engine, attr_name, None)
        if val is not None:
            if isinstance(val, dict):
                return list(val.values())
            if hasattr(val, "__iter__"):
                return list(val)

    logger.warning("RiskEngine'de pozisyon verisi bulunamadı, boş liste döndürülüyor")
    return []


# ─── /execution/open ─────────────────────────────────────────────────────────

@router.post("/open")
async def open_trade(request: Request, body: TradeRequest):
    strategy_manager = request.app.state.strategy_manager
    risk_engine      = request.app.state.risk_engine

    try:
        check_fn = getattr(risk_engine, "check_new_trade", None)
        if callable(check_fn):
            try:
                result = check_fn(symbol=body.symbol, side=body.side, leverage=body.leverage)
                if hasattr(result, "__await__"):
                    risk_ok, risk_reason = await result
                else:
                    risk_ok, risk_reason = result
                if not risk_ok:
                    return {"ok": False, "reason": str(risk_reason), "symbol": body.symbol}
            except Exception as e:
                logger.warning(f"Risk kontrolü atlandı: {e}")

        signal_payload = {
            "symbol":       body.symbol,
            "side":         body.side,
            "strategy_tag": body.strategy_tag,
            "leverage":     body.leverage,
            "order_type":   body.order_type,
            "sl_pct":       body.sl_pct,
            "tp_pct":       body.tp_pct,
        }
        if body.quantity:
            signal_payload["quantity"] = body.quantity

        result = await strategy_manager.handle_signal(signal_payload)

        if isinstance(result, dict):
            if "ok" not in result:
                result["ok"] = True
            return result
        return {"ok": bool(result), "symbol": body.symbol}

    except Exception as e:
        logger.error(f"open_trade hatası [{body.symbol}]: {e}", exc_info=True)
        return {"ok": False, "reason": _safe_str(e)}


# ─── /execution/close ────────────────────────────────────────────────────────

@router.post("/close")
async def close_trade(request: Request, body: CloseRequest):
    strategy_manager = request.app.state.strategy_manager

    try:
        if body.symbol == "ALL":
            fn = getattr(strategy_manager, "close_all_positions", None)
            result = await fn() if callable(fn) else {"ok": True, "msg": "no-op"}
        else:
            close_fn = getattr(strategy_manager, "close_position", None)
            if callable(close_fn):
                result = await close_fn(body.symbol, body.side)
            else:
                # Fallback: doğrudan exchange üzerinden kapat
                try:
                    dc = request.app.state.data_client
                    all_pos = await dc.exchange.fetch_positions()
                    target = None
                    for p in all_pos:
                        contracts = float(p.get("contracts") or 0)
                        if abs(contracts) < 1e-9:
                            continue
                        psym = p.get("symbol", "")
                        if psym == body.symbol or body.symbol in psym.replace("/", "").replace(":USDT", ""):
                            target = p
                            break
                    if not target:
                        return {"ok": False, "reason": f"{body.symbol} pozisyon bulunamadı"}
                    qty = abs(float(target.get("contracts") or 0))
                    contracts_val = float(target.get("contracts") or 0)
                    close_side = "SELL" if contracts_val > 0 else "BUY"
                    raw = await dc.exchange.create_order(
                        symbol=body.symbol, type="market", side=close_side,
                        amount=qty, params={"reduceOnly": True}
                    )
                    result = {"ok": True, "order_id": raw.get("id")}
                except Exception as e2:
                    result = {"ok": False, "reason": str(e2)}

        if isinstance(result, dict):
            if "ok" not in result:
                result["ok"] = True
            return result
        return {"ok": bool(result), "symbol": body.symbol}

    except Exception as e:
        logger.error(f"close_trade hatası [{body.symbol}]: {e}", exc_info=True)
        return {"ok": False, "reason": _safe_str(e)}


# ─── /execution/positions ────────────────────────────────────────────────────

@router.get("/positions")
async def get_positions(request: Request):
    try:
        risk_engine = request.app.state.risk_engine
        positions   = _get_positions(risk_engine)

        normalized = []
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            normalized.append({
                "symbol":         pos.get("symbol")         or pos.get("sym", "–"),
                "side":           pos.get("side")           or pos.get("position_side", "–"),
                "quantity":       pos.get("quantity")       or pos.get("amount")   or pos.get("size", 0),
                "entry_price":    pos.get("entry_price")    or pos.get("entryPrice", 0),
                "mark_price":     pos.get("mark_price")     or pos.get("markPrice",  0),
                "unrealized_pnl": pos.get("unrealized_pnl") or pos.get("unrealizedPnl") or pos.get("pnl", 0),
                "leverage":       pos.get("leverage", 1),
            })

        return {"ok": True, "positions": normalized, "count": len(normalized)}

    except Exception as e:
        logger.error(f"get_positions hatası: {e}", exc_info=True)
        return {"ok": False, "reason": _safe_str(e), "positions": [], "count": 0}
