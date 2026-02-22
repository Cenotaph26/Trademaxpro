"""
Telegram Bildirim Servisi
Tüm önemli olaylar için bildirim gönderir.
"""
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

_bot_token: Optional[str] = None
_chat_id: Optional[str] = None


def init_telegram(bot_token: str, chat_id: str):
    global _bot_token, _chat_id
    _bot_token = bot_token
    _chat_id = chat_id
    logger.info("✅ Telegram servisi başlatıldı")


async def send(text: str, parse_mode: str = "HTML") -> bool:
    """Telegram mesajı gönder."""
    if not _bot_token or not _chat_id:
        return False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{_bot_token}/sendMessage",
                json={"chat_id": _chat_id, "text": text, "parse_mode": parse_mode},
            )
            return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram gönderilemedi: {e}")
        return False


async def notify_trade_open(symbol: str, side: str, qty: float, leverage: int,
                             entry: float, sl: float, tp: float, strategy: str):
    side_emoji = "🟢" if side.upper() in ("BUY", "LONG") else "🔴"
    side_label = "LONG" if side.upper() in ("BUY", "LONG") else "SHORT"
    text = (
        f"{side_emoji} <b>İşlem Açıldı</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 Sembol: <code>{symbol}</code>\n"
        f"📊 Yön: <b>{side_label}</b>\n"
        f"💰 Miktar: <code>${qty:.2f}</code>\n"
        f"⚡ Kaldıraç: <code>{leverage}x</code>\n"
        f"🎯 Giriş: <code>{entry:.4f}</code>\n"
        f"🛡 SL: <code>{sl:.4f}</code>\n"
        f"✅ TP: <code>{tp:.4f}</code>\n"
        f"🤖 Strateji: <code>{strategy}</code>"
    )
    await send(text)


async def notify_trade_close(symbol: str, side: str, pnl: float, pnl_pct: float):
    emoji = "✅" if pnl >= 0 else "❌"
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    text = (
        f"{emoji} <b>Pozisyon Kapatıldı</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📌 Sembol: <code>{symbol}</code>\n"
        f"📊 Yön: <code>{side}</code>\n"
        f"💵 PNL: <b>{pnl_str}</b> ({pnl_pct:+.2f}%)"
    )
    await send(text)


async def notify_kill_switch(reason: str):
    text = (
        f"🚨 <b>KILL SWITCH AKTİF</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⛔ Sebep: <code>{reason}</code>\n"
        f"Bot tüm işlemleri durdurdu!"
    )
    await send(text)


async def notify_signal(symbol: str, side: str, score: float, strategy: str):
    side_emoji = "📈" if side.upper() in ("BUY", "LONG") else "📉"
    text = (
        f"{side_emoji} <b>Yeni Sinyal</b>\n"
        f"📌 <code>{symbol}</code> — {side}\n"
        f"📊 Skor: <code>{score:.3f}</code>\n"
        f"🤖 Strateji: <code>{strategy}</code>"
    )
    await send(text)


async def notify_startup(balance: float):
    text = (
        f"🚀 <b>TrademaXPRO Başlatıldı</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 Bakiye: <code>${balance:.2f} USDT</code>\n"
        f"🤖 Otomatik ajan aktif\n"
        f"✅ Sistem hazır"
    )
    await send(text)
