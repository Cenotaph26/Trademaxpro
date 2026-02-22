"""
Status & monitoring endpoints.
"""
from fastapi import APIRouter, Request, HTTPException
from datetime import datetime

router = APIRouter()


@router.get("/")
async def get_status(request: Request):
    """Bot sağlık durumu ve özet istatistikler."""
    app = request.app
    risk = app.state.risk_engine
    data = app.state.data_client
    rl = app.state.rl_agent

    stats = risk.get_stats()
    rs = risk.state
    ds = data.state

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "kill_switch": {
            "active": rs.kill_switch_active,
            "reason": rs.kill_switch_reason.value if rs.kill_switch_reason else None,
        },
        "market": {
            "symbol": data.settings.SYMBOL,
            "mark_price": ds.mark_price,
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
        "performance": stats,
        "rl_agent": rl.get_status() if rl else None,
    }


@router.get("/positions")
async def get_positions(request: Request):
    """Açık pozisyonları listele."""
    data = request.app.state.data_client
    try:
        positions = await data.get_positions()
        return {"positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balance")
async def get_balance(request: Request):
    """Bakiye bilgisi."""
    data = request.app.state.data_client
    try:
        balance = await data.get_balance()
        return balance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill-switch/activate")
async def activate_kill_switch(request: Request):
    """Manuel kill switch aktif et."""
    from risk.engine import KillSwitchReason
    risk = request.app.state.risk_engine
    risk.activate_kill_switch(KillSwitchReason.MANUAL)
    return {"ok": True, "message": "Kill switch aktif edildi"}


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch(request: Request):
    """Kill switch'i kapat (dikkatli kullan!)."""
    risk = request.app.state.risk_engine
    risk.deactivate_kill_switch()
    return {"ok": True, "message": "Kill switch devre dışı"}


@router.get("/health")
async def health():
    """Railway health check endpoint."""
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
