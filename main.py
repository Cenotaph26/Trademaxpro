"""
Binance Futures Trading Bot - main.py  v2
DCA + Grid + SmartTrade + TradingView + RL Agent — Railway ready

DÜZELTME v2:
  /status/overview, /status/price, /status/symbols,
  /status/logs, /status/agent  →  doğrudan burada tanımlandı.
  api/status.py artık gerekli değil.
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from config.settings import settings
from data.binance_client import BinanceDataClient
from risk.engine import RiskEngine
from strategies.manager import StrategyManager
from rl_agent.agent import RLAgent
from api.webhook import router as webhook_router
from api.execution import router as execution_router
from signal_engine import AutoSignalEngine
from utils.logger import setup_logger
from utils import telegram as tg

logger = setup_logger(__name__)

# ── In-memory log buffer ──────────────────────────────────────────────────────
_log_buffer: list = []
MAX_LOGS = 500


class _BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            _log_buffer.append({
                "time":    self.formatTime(record, "%H:%M:%S"),
                "level":   record.levelname,
                "message": record.getMessage(),
            })
            if len(_log_buffer) > MAX_LOGS:
                _log_buffer.pop(0)
        except Exception:
            pass


# Root logger'a buffer handler ekle — tüm alt loggerlar propagate=True ile buraya ulaşır
_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
if not root_logger.handlers:
    # Varsayılan stdout handler yoksa ekle
    _stdout_handler = logging.StreamHandler()
    _stdout_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(_stdout_handler)
root_logger.addHandler(_buf_handler)
root_logger.setLevel(logging.INFO)

# uvicorn gibi kütüphanelerin loglarını da yakala
for _lib_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
    _lib = logging.getLogger(_lib_logger_name)
    _lib.propagate = True  # root'a ilet

# ── Global state ──────────────────────────────────────────────────────────────
data_client: BinanceDataClient = None
risk_engine: RiskEngine = None
strategy_manager: StrategyManager = None
rl_agent: RLAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_client, risk_engine, strategy_manager, rl_agent

    logger.info("🚀 Trading Bot başlatılıyor...")

    data_client = BinanceDataClient()
    try:
        await data_client.connect()
    except Exception as e:
        logger.error(f"Binance connect hatası: {e} — kısmi mod")

    risk_engine      = RiskEngine(settings)
    strategy_manager = StrategyManager(data_client, risk_engine, settings)
    rl_agent         = RLAgent(settings)

    try:
        await rl_agent.load_model()
    except Exception as e:
        logger.warning(f"RL model yüklenemedi: {e}")

    strategy_manager.set_rl_agent(rl_agent)

    signal_engine = AutoSignalEngine(
        data_client, strategy_manager, risk_engine, rl_agent, settings
    )

    asyncio.create_task(data_client.stream_market_data())
    asyncio.create_task(risk_engine.monitor_loop())
    asyncio.create_task(rl_agent.learning_loop())
    asyncio.create_task(signal_engine.start())

    app.state.data_client      = data_client
    app.state.risk_engine      = risk_engine
    app.state.strategy_manager = strategy_manager
    app.state.rl_agent         = rl_agent
    app.state.signal_engine    = signal_engine

    logger.info("✅ Bot hazır!")

    # Telegram başlat
    if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
        tg.init_telegram(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)
        try:
            bal = await data_client.get_balance()
            asyncio.create_task(tg.notify_startup(bal.get("total", 0)))
        except Exception:
            asyncio.create_task(tg.notify_startup(0))

    yield

    logger.info("🛑 Bot kapatılıyor...")
    for fn, label in [
        (strategy_manager.close_all_positions, "Pozisyon kapatma"),
        (data_client.disconnect,               "Disconnect"),
    ]:
        try:
            await fn()
        except Exception as e:
            logger.error(f"{label} hatası: {e}")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Binance Futures Bot", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(webhook_router,   prefix="/webhook")
app.include_router(execution_router, prefix="/execution")
# NOT: status_router artık kullanılmıyor, endpoint'ler aşağıda inline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val, default=0):
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def _positions_safe(re) -> list:
    """RiskEngine'den pozisyon listesini güvenle çeker."""
    for m in ("get_open_positions", "get_positions", "list_positions"):
        fn = getattr(re, m, None)
        if callable(fn):
            try:
                r = fn()
                return r if isinstance(r, list) else list(r)
            except Exception:
                pass
    for a in ("open_positions", "positions", "active_positions"):
        v = getattr(re, a, None)
        if v is not None:
            return list(v.values()) if isinstance(v, dict) else list(v)
    return []


# ════════════════════════════════════════════════════════════════════════════════
#  STATUS ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/status")
@app.get("/status/")
@app.get("/status/health")
@app.get("/health")
async def status_health():
    return {"status": "ok", "service": "trademaxpro"}


@app.get("/status/overview")
async def status_overview(request: Request):
    try:
        ds  = request.app.state.data_client.state
        re  = request.app.state.risk_engine
        se  = request.app.state.signal_engine
        rl  = request.app.state.rl_agent

        positions  = _positions_safe(re)
        open_count = len(positions)

        se_status = {}
        try:
            se_status = se.get_status()
        except Exception:
            pass

        rl_info = {}
        if rl:
            rl_info = {
                "epsilon":     round(_f(getattr(rl, "epsilon",      0)), 4),
                "episodes":    int(getattr(rl,   "episode_count",   0)),
                "last_reward": round(_f(getattr(rl, "last_reward",  0)), 4),
            }

        balance    = _f(getattr(ds, "wallet_balance",  None))
        unrealized = _f(getattr(ds, "unrealized_pnl",  None))
        daily_pnl  = _f(getattr(ds, "daily_pnl",       None))
        mark_price = _f(getattr(ds, "mark_price",       None))
        atr_14     = _f(getattr(ds, "atr_14",           None))
        funding    = _f(getattr(ds, "funding_rate",     None))
        regime     = getattr(ds, "regime", "unknown")
        margin_pct = _f(getattr(ds, "margin_usage_pct", None))
        max_pos    = getattr(getattr(re, "settings", None), "MAX_POSITIONS", 5)

        return {
            "ok": True,
            "balance": balance,        "wallet_balance": balance,
            "unrealized_pnl": unrealized, "daily_pnl": daily_pnl,
            "daily_trades": int(getattr(ds, "daily_trades", 0)),
            "open_positions": open_count,
            "mark_price": mark_price,  "last_price": mark_price,
            "atr_14": atr_14,          "regime": regime,
            "funding_rate": funding,
            "signal_engine": se_status,
            "rl_agent": rl_info,
            "risk": {"max_positions": max_pos, "margin_usage_pct": margin_pct},
        }
    except Exception as e:
        logger.error(f"/status/overview: {e}", exc_info=True)
        return {
            "ok": True, "balance": 0, "wallet_balance": 0,
            "unrealized_pnl": 0, "daily_pnl": 0, "daily_trades": 0,
            "open_positions": 0, "mark_price": 0, "atr_14": 0,
            "regime": "unknown", "signal_engine": {}, "rl_agent": {},
            "risk": {}, "_error": str(e),
        }


@app.get("/status/price")
async def status_price(request: Request, symbol: Optional[str] = Query(None)):
    try:
        dc     = request.app.state.data_client
        ds     = dc.state
        target = (symbol or "BTCUSDT").upper()
        cur    = getattr(getattr(dc, "settings", None), "SYMBOL", "BTCUSDT")

        if not symbol or target == cur:
            price = _f(getattr(ds, "mark_price", None))
            return {"ok": True, "symbol": target, "price": price, "mark_price": price}

        # Farklı sembol → ccxt ticker
        try:
            sym_fmt = target[:-4] + "/USDT"
            ticker  = await dc.exchange.fetch_ticker(sym_fmt)
            price   = _f(ticker.get("last") or ticker.get("close"))
        except Exception:
            price = _f(getattr(ds, "mark_price", None))

        return {"ok": True, "symbol": target, "price": price, "mark_price": price}
    except Exception as e:
        return {"ok": True, "symbol": symbol or "BTCUSDT", "price": 0, "_error": str(e)}


@app.get("/status/symbols")
async def status_symbols(request: Request):
    try:
        dc = request.app.state.data_client
        if hasattr(dc, "symbols") and dc.symbols:
            syms = list(dc.symbols)
            return {"ok": True, "symbols": syms, "count": len(syms)}
        if hasattr(dc, "exchange") and dc.exchange:
            try:
                markets = await dc.exchange.fetch_markets()
                syms = sorted([
                    m["symbol"].replace("/", "")
                    for m in markets
                    if m.get("type") in ("future", "swap", "futures")
                    and str(m.get("quote", "")).upper() in ("USDT", "BUSD")
                    and m.get("active", True)
                ])
                return {"ok": True, "symbols": syms, "count": len(syms)}
            except Exception as e:
                logger.warning(f"fetch_markets: {e}")
    except Exception as e:
        logger.warning(f"/status/symbols: {e}")

    fallback = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
        "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","MATICUSDT","LTCUSDT",
        "ATOMUSDT","UNIUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT",
        "SHIBUSDT","TRXUSDT","ETCUSDT","XLMUSDT","VETUSDT","FILUSDT",
        "SANDUSDT","MANAUSDT","AAVEUSDT","SNXUSDT","CRVUSDT","MKRUSDT",
    ]
    return {"ok": True, "symbols": fallback, "count": len(fallback)}


@app.get("/status/logs")
async def status_logs(limit: int = Query(100)):
    logs = _log_buffer[-limit:] if limit else _log_buffer
    return {"ok": True, "logs": list(reversed(logs)), "total": len(_log_buffer)}


@app.post("/status/agent")
async def status_agent(request: Request):
    try:
        body    = await request.json()
        enabled = bool(body.get("enabled", True))
        se      = request.app.state.signal_engine

        # Ajan ayarlarını güncelle
        agent_settings = body.get("settings", {})
        if agent_settings:
            re = request.app.state.risk_engine
            rl = request.app.state.rl_agent
            # Risk parametrelerini güncelle
            if "daily_max_loss_pct" in agent_settings:
                re.s.DAILY_MAX_LOSS_PCT = float(agent_settings["daily_max_loss_pct"])
            if "max_drawdown_pct" in agent_settings:
                re.s.MAX_DRAWDOWN_PCT = float(agent_settings["max_drawdown_pct"])
            if "risk_per_trade_pct" in agent_settings:
                re.s.RISK_PER_TRADE_PCT = float(agent_settings["risk_per_trade_pct"])
            if "max_positions" in agent_settings:
                re.s.MAX_OPEN_POSITIONS = int(agent_settings["max_positions"])
            if "kill_switch_consecutive_loss" in agent_settings:
                re.s.KILL_SWITCH_CONSECUTIVE_LOSS = int(agent_settings["kill_switch_consecutive_loss"])
            # RL parametrelerini güncelle
            if rl and "rl_epsilon" in agent_settings:
                rl.epsilon = float(agent_settings["rl_epsilon"])
            # Sinyal motoru parametrelerini güncelle
            if "scan_interval_min" in agent_settings and hasattr(se, "scan_interval_min"):
                se.scan_interval_min = int(agent_settings["scan_interval_min"])
            if "min_signal_score" in agent_settings and hasattr(se, "min_signal_score"):
                se.min_signal_score = float(agent_settings["min_signal_score"])
            logger.info(f"⚙️ Ajan ayarları güncellendi: {agent_settings}")

        # Kill switch
        if body.get("kill_switch") is True:
            from risk.engine import KillSwitchReason
            request.app.state.risk_engine.activate_kill_switch(KillSwitchReason.MANUAL)
            asyncio.create_task(tg.notify_kill_switch("Manuel — Dashboard"))
        elif body.get("kill_switch") is False:
            request.app.state.risk_engine.deactivate_kill_switch()

        if enabled:
            # Her seferinde yeniden başlat (force restart)
            se._running = False
            await asyncio.sleep(0.1)
            se._running = True
            asyncio.create_task(se.start())
            msg = "Ajan başlatıldı"
        else:
            se._running = False
            msg = "Ajan durduruldu"

        logger.info(f"Agent toggle: {msg}")
        return {"ok": True, "running": enabled, "message": msg}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


@app.get("/execution/positions/live")
async def live_positions(request: Request):
    """Binance'den canlı pozisyonları çeker."""
    try:
        dc = request.app.state.data_client
        if not dc.exchange or not dc._auth_ok:
            return {"ok": False, "reason": "Binance bağlı değil", "positions": []}
        raw = await dc.exchange.fetch_positions()
        positions = []
        for p in raw:
            contracts = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
            if abs(contracts) < 1e-9:
                continue
            notional = float(p.get("notional") or p.get("info", {}).get("notional") or 0)
            entry  = float(p.get("entryPrice") or p.get("info", {}).get("entryPrice") or 0)
            mark   = float(p.get("markPrice")  or p.get("info", {}).get("markPrice")  or 0)
            upnl   = float(p.get("unrealizedPnl") or p.get("info", {}).get("unRealizedProfit") or 0)
            lev    = int(float(p.get("leverage") or p.get("info", {}).get("leverage") or 1))
            liq    = float(p.get("liquidationPrice") or p.get("info", {}).get("liquidationPrice") or 0)
            margin = float(p.get("initialMargin") or p.get("info", {}).get("isolatedMargin") or abs(notional / lev) if lev else 0)
            
            # Pozisyon yönünü birden fazla kaynaktan doğru belirle
            pos_side_raw = (p.get("side") or p.get("positionSide") or 
                           p.get("info", {}).get("positionSide") or "").upper()
            if pos_side_raw in ("LONG", "SHORT"):
                side = pos_side_raw
            elif contracts > 0:
                side = "LONG"
            elif contracts < 0:
                side = "SHORT"
            else:
                side = "LONG"  # fallback
            sym    = p.get("symbol") or ""
            # Sembol görüntüleme formatını normalize et: BTC/USDT:USDT → BTCUSDT
            sym_display = sym.replace("/", "").replace(":USDT", "").replace(":BUSD", "")
            if entry > 0 and mark > 0:
                pnl_pct = ((mark - entry) / entry * 100 * lev) if side == "LONG" else ((entry - mark) / entry * 100 * lev)
            else:
                pnl_pct = 0.0
            positions.append({
                "symbol": sym_display,    # görüntü için (BTCUSDT)
                "symbol_ccxt": sym,       # ccxt için (BTC/USDT:USDT)
                "side": side,
                "contracts": abs(contracts),
                "notional": abs(notional),
                "entry_price": entry,
                "mark_price": mark,
                "unrealized_pnl": upnl,
                "pnl_pct": round(pnl_pct, 2),
                "leverage": lev,
                "liquidation_price": liq,
                "margin": abs(margin),
            })
        return {"ok": True, "positions": positions, "count": len(positions)}
    except Exception as e:
        logger.error(f"live_positions hatası: {e}", exc_info=True)
        return {"ok": False, "reason": str(e), "positions": []}


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/")
async def dashboard():
    return FileResponse("dashboard.html")


# ── Shutdown ──────────────────────────────────────────────────────────────────
def _handle_shutdown(sig, frame):
    logger.warning(f"Signal {sig} alındı, kapatılıyor...")
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT,  _handle_shutdown)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        reload=False,
    )
