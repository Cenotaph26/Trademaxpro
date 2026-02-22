"""
TradingView Webhook endpoint.
POST /webhook/tradingview
"""
import hmac
import hashlib
import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class TVSignal(BaseModel):
    symbol: str = Field(..., description="BTCUSDT, ETHUSDT vb.")
    side: str = Field(..., description="BUY veya SELL")
    timeframe: str = Field(default="1h")
    strategy_tag: str = Field(default="tv")
    entry_hint: Optional[float] = Field(default=None, description="TV'nin önerdiği giriş fiyatı")
    secret: str = Field(default="", description="Webhook doğrulama anahtarı")

    class Config:
        extra = "allow"  # TV'nin ek alanlarını geçir


def verify_secret(signal: TVSignal, request: Request):
    """Basit secret key doğrulama."""
    from config.settings import settings
    if signal.secret != settings.WEBHOOK_SECRET:
        logger.warning(f"Geçersiz webhook secret IP={request.client.host}")
        raise HTTPException(status_code=403, detail="Geçersiz secret")
    return True


@router.post("/tradingview")
async def tradingview_webhook(signal: TVSignal, request: Request):
    """
    TradingView webhook alıcısı.
    
    Örnek TV alert mesajı:
    {"symbol": "BTCUSDT", "side": "BUY", "timeframe": "1h", 
     "strategy_tag": "ema_cross", "entry_hint": 45000, "secret": "tv-secret"}
    """
    verify_secret(signal, request)

    app = request.app
    strategy_manager = app.state.strategy_manager
    risk_engine = app.state.risk_engine

    # Kill switch kontrol
    if risk_engine.state.kill_switch_active:
        return {
            "status": "blocked",
            "reason": f"Kill switch aktif: {risk_engine.state.kill_switch_reason}",
        }

    logger.info(f"📩 TV Webhook: {signal.symbol} {signal.side} [{signal.strategy_tag}]")

    result = await strategy_manager.handle_signal(signal.dict())

    # Telegram bildirimi (opsiyonel)
    await _notify(request, signal, result)

    return {
        "status": "executed" if result.get("ok") else "rejected",
        "strategy": result.get("strategy"),
        "risk_mode": result.get("risk_mode"),
        "leverage": result.get("leverage"),
        "reason": result.get("reason"),
    }


async def _notify(request: Request, signal: TVSignal, result: dict):
    """Telegram bildirimi gönder (bot token varsa)."""
    try:
        from config.settings import settings
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            return
        import httpx
        status = "✅" if result.get("ok") else "❌"
        text = (
            f"{status} {signal.symbol} {signal.side}\n"
            f"Strateji: {result.get('strategy', '-')}\n"
            f"Risk: {result.get('risk_mode', '-')}\n"
            f"Kaldıraç: {result.get('leverage', '-')}x\n"
            f"Sebep: {result.get('reason', 'OK')}"
        )
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text},
                timeout=5,
            )
    except Exception:
        pass  # Bildirim başarısız olsa bot devam etmeli
