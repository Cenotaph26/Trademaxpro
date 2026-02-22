"""
api/status.py — Durum, fiyat, sembol ve log endpoint'leri
DÜZELTME:
- /status/overview  → bot genel durumu
- /status/price     → sembol fiyatı
- /status/symbols   → futures sembol listesi
- /status/logs      → son loglar
- /status/agent     → ajan aç/kapat
"""
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _safe(val, default=0):
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


# ─── /status (health check) ──────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def health():
    return {"status": "ok", "service": "trademaxpro"}


# ─── /status/overview ────────────────────────────────────────────────────────

@router.get("/overview")
async def overview(request: Request):
    try:
        data_client      = request.app.state.data_client
        risk_engine      = request.app.state.risk_engine
        strategy_manager = request.app.state.strategy_manager
        rl_agent         = request.app.state.rl_agent
        signal_engine    = request.app.state.signal_engine

        ds = data_client.state

        # Bakiye / PNL
        balance      = _safe(getattr(ds, "wallet_balance",   None))
        unrealized   = _safe(getattr(ds, "unrealized_pnl",   None))
        daily_pnl    = _safe(getattr(ds, "daily_pnl",        None))
        daily_trades = int(getattr(ds,   "daily_trades",     0))

        # Pozisyon sayısı — birden fazla yerde saklıyor olabilir
        open_positions = 0
        if hasattr(risk_engine, "open_positions"):
            op = risk_engine.open_positions
            open_positions = len(op) if hasattr(op, "__len__") else int(op)
        elif hasattr(risk_engine, "positions"):
            open_positions = len(risk_engine.positions)

        # Piyasa verisi
        mark_price = _safe(getattr(ds, "mark_price",   None))
        atr_14     = _safe(getattr(ds, "atr_14",       None))
        regime     = getattr(ds, "regime", "unknown")
        funding    = _safe(getattr(ds, "funding_rate", None))

        # RL
        rl_info = {}
        if rl_agent:
            rl_info = {
                "epsilon":     round(_safe(getattr(rl_agent, "epsilon", 0)), 4),
                "episodes":    int(getattr(rl_agent, "episode_count", 0)),
                "last_reward": round(_safe(getattr(rl_agent, "last_reward", 0)), 4),
            }

        # Signal engine
        se_status = {}
        if signal_engine:
            try:
                se_status = signal_engine.get_status()
            except Exception:
                se_status = {"running": False, "signal_count": 0}

        # Risk
        max_positions = getattr(getattr(risk_engine, "settings", None), "MAX_POSITIONS", 5)
        margin_usage  = _safe(getattr(ds, "margin_usage_pct", None))

        return {
            "ok":              True,
            "balance":         balance,
            "wallet_balance":  balance,
            "unrealized_pnl":  unrealized,
            "daily_pnl":       daily_pnl,
            "daily_trades":    daily_trades,
            "open_positions":  open_positions,
            "mark_price":      mark_price,
            "last_price":      mark_price,
            "atr_14":          atr_14,
            "regime":          regime,
            "funding_rate":    funding,
            "signal_engine":   se_status,
            "rl_agent":        rl_info,
            "risk": {
                "max_positions":    max_positions,
                "margin_usage_pct": margin_usage,
            },
        }
    except Exception as e:
        logger.error(f"/status/overview hatası: {e}", exc_info=True)
        return JSONResponse(status_code=200, content={
            "ok": True,
            "balance": 0, "wallet_balance": 0, "unrealized_pnl": 0,
            "daily_pnl": 0, "daily_trades": 0, "open_positions": 0,
            "mark_price": 0, "atr_14": 0, "regime": "unknown",
            "signal_engine": {}, "rl_agent": {}, "risk": {},
            "_error": str(e),
        })


# ─── /status/price ───────────────────────────────────────────────────────────

@router.get("/price")
async def get_price(request: Request, symbol: Optional[str] = None):
    try:
        data_client = request.app.state.data_client
        ds = data_client.state

        # Eğer symbol parametresi varsa ve farklı bir sembolse ccxt'ten çek
        target = (symbol or "BTCUSDT").upper()
        current_symbol = getattr(getattr(data_client, "settings", None), "SYMBOL", "BTCUSDT")

        if target == current_symbol or not symbol:
            price = _safe(getattr(ds, "mark_price", None))
            return {"ok": True, "symbol": target, "price": price, "mark_price": price}

        # Farklı sembol için ccxt'ten anlık fiyat çek
        try:
            ticker = await data_client.fetch_ticker(target)
            price = _safe(ticker.get("last") or ticker.get("close"))
            return {"ok": True, "symbol": target, "price": price, "mark_price": price}
        except Exception:
            # ccxt çalışmazsa mark_price dön
            price = _safe(getattr(ds, "mark_price", None))
            return {"ok": True, "symbol": target, "price": price, "mark_price": price}

    except Exception as e:
        logger.error(f"/status/price hatası: {e}", exc_info=True)
        return {"ok": True, "symbol": symbol or "BTCUSDT", "price": 0, "mark_price": 0}


# ─── /status/symbols ─────────────────────────────────────────────────────────

@router.get("/symbols")
async def get_symbols(request: Request):
    try:
        data_client = request.app.state.data_client

        # Önce data_client'ta hazır sembol listesi var mı?
        if hasattr(data_client, "symbols") and data_client.symbols:
            return {"ok": True, "symbols": list(data_client.symbols), "count": len(data_client.symbols)}

        if hasattr(data_client, "exchange") and data_client.exchange:
            try:
                markets = await data_client.exchange.fetch_markets()
                symbols = [
                    m["symbol"].replace("/", "")
                    for m in markets
                    if m.get("type") in ("future", "swap", "futures")
                    and str(m.get("quote", "")).upper() in ("USDT", "BUSD")
                    and m.get("active", True)
                ]
                symbols.sort()
                return {"ok": True, "symbols": symbols, "count": len(symbols)}
            except Exception as e:
                logger.warning(f"fetch_markets hatası: {e}")

        # Fallback: bilinen popüler semboller
        fallback = [
            "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
            "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","MATICUSDT","LTCUSDT",
            "ATOMUSDT","UNIUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT",
            "SHIBUSDT","TRXUSDT","ETCUSDT","XLMUSDT","VETUSDT","FILUSDT",
            "SANDUSDT","MANAUSDT","AAVEUSDT","SNXUSDT","CRVUSDT","MKRUSDT",
            "COMPUSDT","YFIUSDT","SUSHIUSDT","1INCHUSDT","ENJUSDT","CHZUSDT",
        ]
        return {"ok": True, "symbols": fallback, "count": len(fallback)}

    except Exception as e:
        logger.error(f"/status/symbols hatası: {e}", exc_info=True)
        return {"ok": False, "symbols": [], "count": 0}


# ─── /status/logs ────────────────────────────────────────────────────────────

# In-memory log buffer (main.py'de setup_logger ile doldurulur)
_log_buffer: list = []
MAX_LOGS = 500


class BufferHandler(logging.Handler):
    """Tüm log satırlarını _log_buffer listesine yazar."""
    def emit(self, record):
        try:
            _log_buffer.append({
                "time":    self.formatTime(record, "%H:%M:%S"),
                "level":   record.levelname,
                "message": record.getMessage(),
                "name":    record.name,
            })
            if len(_log_buffer) > MAX_LOGS:
                _log_buffer.pop(0)
        except Exception:
            pass


# Root logger'a handler ekle (import sırasında bir kez çalışır)
_buffer_handler = BufferHandler()
_buffer_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_buffer_handler)


@router.get("/logs")
async def get_logs(limit: int = 100):
    logs = _log_buffer[-limit:] if limit else _log_buffer
    return {"ok": True, "logs": list(reversed(logs)), "total": len(_log_buffer)}


# ─── /status/agent ───────────────────────────────────────────────────────────

@router.post("/agent")
async def set_agent(request: Request):
    try:
        body = await request.json()
        enabled = bool(body.get("enabled", True))
        signal_engine = request.app.state.signal_engine

        if enabled:
            if not signal_engine._running:
                signal_engine._running = True
                import asyncio
                asyncio.create_task(signal_engine.start())
            msg = "Ajan başlatıldı"
        else:
            signal_engine._running = False
            msg = "Ajan durduruldu"

        logger.info(f"Agent toggle: {msg}")
        return {"ok": True, "running": enabled, "message": msg}

    except Exception as e:
        logger.error(f"/status/agent hatası: {e}", exc_info=True)
        return {"ok": False, "reason": str(e)}
