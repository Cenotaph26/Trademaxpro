"""
Status & monitoring endpoints + WebSocket canlı veri.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Set

from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)

_ws_clients: Set[WebSocket] = set()


async def broadcast_loop(app):
    """Her saniye tüm bağlı istemcilere veri yayınla."""
    while True:
        try:
            if _ws_clients:
                data = await _build_status(app)
                msg = json.dumps(data)
                dead = set()
                for ws in list(_ws_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                _ws_clients -= dead
        except Exception as e:
            logger.warning(f"Broadcast hatası: {e}")
        await asyncio.sleep(1)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    app = websocket.app
    logger.info(f"WebSocket bağlandı. Toplam: {len(_ws_clients)}")
    try:
        data = await _build_status(app)
        await websocket.send_text(json.dumps(data))
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS hatası: {e}")
    finally:
        _ws_clients.discard(websocket)


async def _build_status(app) -> dict:
    risk = app.state.risk_engine
    data = app.state.data_client
    rl = app.state.rl_agent

    stats = risk.get_stats()
    rs = risk.state
    ds = data.state

    positions = []
    try:
        if data._auth_ok:
            positions = await data.get_positions()
    except Exception:
        pass

    return {
        "ts": datetime.utcnow().isoformat(),
        "kill_switch": {
            "active": rs.kill_switch_active,
            "reason": rs.kill_switch_reason.value if rs.kill_switch_reason else None,
        },
        "market": {
            "symbol": data.settings.SYMBOL,
            "mark_price": ds.mark_price,
            "last_price": ds.last_price,
            "bid": ds.bid,
            "ask": ds.ask,
            "regime": ds.regime,
            "atr_14": round(ds.atr_14, 4),
            "funding_rate": ds.funding_rate,
            "spread_pct": round(ds.spread_pct, 4),
            "updated_at": ds.updated_at.isoformat() if ds.updated_at else None,
        },
        "risk": {
            "open_positions": rs.open_count,
            "longs": rs.long_count,
            "shorts": rs.short_count,
            "daily_loss": round(rs.daily_loss, 2),
            "drawdown_pct": round(rs.current_drawdown_pct, 2),
            "consecutive_losses": rs.consecutive_losses,
        },
        "positions": positions,
        "performance": stats,
        "rl_agent": rl.get_status() if rl else None,
    }


@router.get("/")
async def get_status(request: Request):
    return await _build_status(request.app)


@router.get("/markets")
async def get_markets(request: Request):
    data = request.app.state.data_client
    try:
        if not data.exchange:
            raise HTTPException(status_code=503, detail="Exchange bağlı değil")
        markets = data.exchange.markets or {}
        result = []
        for symbol, m in markets.items():
            if m.get("type") == "swap" or m.get("future"):
                result.append({
                    "symbol": symbol,
                    "base": m.get("base"),
                    "quote": m.get("quote"),
                    "active": m.get("active", True),
                    "contractSize": m.get("contractSize"),
                    "precision": m.get("precision", {}),
                    "limits": m.get("limits", {}),
                })
        return {"markets": result, "count": len(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(request: Request):
    data = request.app.state.data_client
    try:
        positions = await data.get_positions()
        return {"positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balance")
async def get_balance(request: Request):
    data = request.app.state.data_client
    try:
        balance = await data.get_balance()
        return balance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill-switch/activate")
async def activate_kill_switch(request: Request):
    from risk.engine import KillSwitchReason
    risk = request.app.state.risk_engine
    risk.activate_kill_switch(KillSwitchReason.MANUAL)
    return {"ok": True, "message": "Kill switch aktif edildi"}


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch(request: Request):
    risk = request.app.state.risk_engine
    risk.deactivate_kill_switch()
    return {"ok": True, "message": "Kill switch devre dışı"}


@router.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
