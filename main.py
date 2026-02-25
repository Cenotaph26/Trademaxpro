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
from persistence import init_db, save_log_entry, load_logs, get_storage_info, load_trades, get_trade_stats, load_daily_stats

logger = setup_logger(__name__)

# ── Veritabani baslat ─────────────────────────────────────────────────────────
init_db()

# ── In-memory log buffer (hiz) + SQLite (kalicilik) ──────────────────────────
_log_buffer: list = []
MAX_LOGS = 500

import os, json, threading

# Gürültülü log kaynakları — buffer'a yazılmasın
_NOISE_SOURCES = {"uvicorn.access", "uvicorn.error", "asyncio", "concurrent.futures"}
_NOISE_MSGS = ("asyncio/runners", "uvloop/loop", "until_complete", "run_until_complete",
               "GET /status", "GET /health", "GET /execution", 'HTTP/1.1" 200')

class _BufferHandler(logging.Handler):
    _fmt = logging.Formatter("%(message)s")

    def emit(self, record):
        try:
            # HTTP access logları ve asyncio internal hatalarını filtrele
            if record.name in _NOISE_SOURCES:
                return
            msg = record.getMessage()
            if any(n in msg for n in _NOISE_MSGS):
                return
            if len(msg) > 500:
                msg = msg[:497] + "..."
            t = self._fmt.formatTime(record, "%H:%M:%S")
            entry = {"time": t, "level": record.levelname, "message": msg, "name": record.name}
            _log_buffer.append(entry)
            if len(_log_buffer) > MAX_LOGS:
                _log_buffer.pop(0)
            # SQLite kalici kayit
            save_log_entry(t, record.levelname, record.name, msg)
        except Exception:
            pass

def _load_persisted_logs():
    try:
        rows = load_logs(limit=200)
        for r in reversed(rows):
            _log_buffer.append({
                "time":    r.get("time", ""),
                "level":   r.get("level", "INFO"),
                "message": r.get("message", ""),
                "name":    r.get("name", ""),
            })
    except Exception:
        pass

_load_persisted_logs()


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

# uvicorn loglarını da yakala - propagate + direkt handler
for _lib_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi",
                          "signal_engine", "strategies.manager", "rl_agent.agent",
                          "risk.engine", "data.binance_client", "execution.executor"):
    _lib = logging.getLogger(_lib_logger_name)
    _lib.propagate = True
    _lib.addHandler(_buf_handler)  # direkt de ekle (propagate bypass için)

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

    async def supervised_task(coro_fn, name: str, restart_delay: int = 30):
        """Task crash'te otomatik yeniden başlatır — 7/24 dayanıklılık."""
        consecutive = 0
        while True:
            try:
                logger.info(f"▶ Task başlatıldı: {name} (deneme #{consecutive+1})")
                await coro_fn()
                consecutive = 0
                logger.warning(f"⚠ Task normal bitti (yeniden başlatılıyor): {name}")
            except asyncio.CancelledError:
                logger.info(f"🛑 Task iptal edildi: {name}")
                break
            except Exception as e:
                consecutive += 1
                delay = min(restart_delay * consecutive, 300)  # max 5dk
                logger.error(f"💥 Task crash ({name}): {e} — {delay}sn sonra restart (#{consecutive})")
                await asyncio.sleep(delay)
                continue
            await asyncio.sleep(restart_delay)

    asyncio.create_task(supervised_task(data_client.stream_market_data, "MarketData", 10))
    asyncio.create_task(supervised_task(risk_engine.monitor_loop,        "RiskMonitor", 15))
    asyncio.create_task(supervised_task(rl_agent.learning_loop,          "RLAgent",     20))
    asyncio.create_task(supervised_task(signal_engine.start,             "SignalEngine", 30))

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
async def status_logs(limit: int = Query(200), level: Optional[str] = Query(None)):
    """Kalici SQLite + memory buffer birlesimi ile loglar doner. Restart sonrasi gecmis korunur."""
    if not _log_buffer:
        logger.info("📋 Log sistemi aktif — kalici SQLite hafiza")
    # Memory buffer yeterliyse hizlica don
    combined = list(_log_buffer)
    # Ek gecmis icin SQLite'e bak
    if len(combined) < limit:
        try:
            db_logs = load_logs(limit=limit, level_filter=level)
            seen = {(e.get("time"), e.get("message")) for e in combined}
            for r in db_logs:
                key = (r.get("time"), r.get("message"))
                if key not in seen:
                    combined.append(r)
                    seen.add(key)
        except Exception:
            pass
    if level:
        combined = [e for e in combined if e.get("level", "").upper() == level.upper()]
    logs = combined[-limit:]
    return {"ok": True, "logs": list(reversed(logs)), "total": len(combined)}


@app.get("/status/history")
async def status_history(request: Request, limit: int = Query(500)):
    """Tum zamanli trade gecmisi — SQLite'den. Sayfa yenilemede kaybolmaz."""
    try:
        trades = load_trades(limit=limit)
        stats  = get_trade_stats()
        daily  = load_daily_stats(days=30)
        return {
            "ok": True,
            "trades": trades,
            "stats":  stats,
            "daily":  daily,
            "total":  len(trades),
        }
    except Exception as e:
        return {"ok": False, "trades": [], "stats": {}, "daily": [], "reason": str(e)}


@app.get("/status/storage")
async def status_storage():
    """Depolama bilgisi — Railway Volume durumu ve boyutu."""
    try:
        info = get_storage_info()
        return {"ok": True, **info}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


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
            if not getattr(se, "_running", False):
                se._running = True
                # Supervised task içinde değilse direkt başlat
                asyncio.create_task(se.start())
                msg = "Ajan yeniden başlatıldı"
            else:
                msg = "Ajan zaten çalışıyor"
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
        try:
            raw = await asyncio.wait_for(dc.exchange.fetch_positions(), timeout=8.0)
        except asyncio.TimeoutError:
            return {"ok": False, "reason": "fetch_positions timeout (8s)", "positions": []}
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
        # Açık emirleri 3sn timeout ile çek (SL/TP bilgisi)
        try:
            raw_orders = await asyncio.wait_for(
                dc.exchange.fetch_open_orders(), timeout=3.0
            )
            sl_map, tp_map = {}, {}
            for o in raw_orders:
                osym  = (o.get("symbol") or "").replace("/","").replace(":USDT","")
                otype = (o.get("type") or "").upper()
                ostop = float(o.get("stopPrice") or o.get("price") or 0)
                reduce = o.get("reduceOnly") or o.get("info", {}).get("reduceOnly") == "true"
                if not reduce or ostop <= 0:
                    continue
                for p in positions:
                    if osym != p["symbol"]:
                        continue
                    entry_p = p.get("entry_price", 0)
                    pside   = p.get("side", "LONG")
                    if otype in ("STOP_MARKET", "STOP"):
                        sl_map[osym] = ostop
                    elif otype in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT"):
                        tp_map[osym] = ostop
                    elif otype == "LIMIT" and entry_p > 0:
                        # testnet fallback: fiyata göre SL/TP ayır
                        is_tp = (pside == "LONG" and ostop > entry_p) or (pside == "SHORT" and ostop < entry_p)
                        if is_tp:
                            tp_map[osym] = ostop
                        else:
                            sl_map[osym] = ostop
            for p in positions:
                p["sl"] = sl_map.get(p["symbol"], 0)
                p["tp"] = tp_map.get(p["symbol"], 0)
        except asyncio.TimeoutError:
            logger.debug("Açık emir fetch timeout (3s) — SL/TP gösterilmeyecek")
        except Exception as oe:
            logger.debug(f"SL/TP emir çekme hatası (önemsiz): {oe}")

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


# ════════════════════════════════════════════════════════════════════════════════
#  EKSİK ENDPOINT'LER — dashboard uyumluluğu
# ════════════════════════════════════════════════════════════════════════════════

import time as _time

# Market cache (3dk TTL — Binance rate limit koruma)
_market_cache: dict = {"prices": {}, "ts": 0}
_MARKET_TTL = 180  # saniye


@app.get("/status/market")
async def status_market(request: Request):
    """
    Tüm Binance Futures USDT çiftlerinin fiyat + değişim bilgisi.
    Dashboard'un coin grid'ini doldurur.
    Format: { ok: true, prices: { "BTCUSDT": { price, change, high, low, quoteVolume } } }
    """
    global _market_cache
    now = _time.time()

    # Cache geçerliyse direkt dön
    if _market_cache["prices"] and (now - _market_cache["ts"]) < _MARKET_TTL:
        return {"ok": True, "prices": _market_cache["prices"], "cached": True,
                "count": len(_market_cache["prices"])}

    try:
        dc = request.app.state.data_client
        if not hasattr(dc, "exchange") or not dc.exchange:
            return {"ok": False, "prices": {}, "reason": "exchange yok"}

        # Binance fapi/v1/ticker/24hr — tüm semboller tek seferde
        try:
            tickers = await dc.exchange.fetch_tickers()
        except Exception as e:
            logger.warning(f"fetch_tickers hatası: {e}")
            # Cache varsa eski veriyi dön
            if _market_cache["prices"]:
                return {"ok": True, "prices": _market_cache["prices"], "cached": True, "stale": True}
            return {"ok": False, "prices": {}, "reason": str(e)}

        prices = {}
        for sym, t in tickers.items():
            # Sadece USDT perpetual futures
            if not (sym.endswith("/USDT") or sym.endswith(":USDT")):
                continue
            # Sembol normalize: BTC/USDT:USDT → BTCUSDT
            clean = sym.replace("/", "").replace(":USDT", "").replace(":BUSD", "")
            last  = float(t.get("last")  or t.get("close") or 0)
            chg   = float(t.get("percentage") or t.get("change") or 0)
            high  = float(t.get("high")  or 0)
            low   = float(t.get("low")   or 0)
            vol   = float(t.get("quoteVolume") or t.get("baseVolume") or 0)
            if last > 0:
                prices[clean] = {
                    "price":       last,
                    "change":      round(chg, 4),
                    "high":        high,
                    "low":         low,
                    "quoteVolume": vol,
                }

        if prices:
            _market_cache = {"prices": prices, "ts": now}
            logger.info(f"Market cache güncellendi: {len(prices)} coin")

        return {"ok": True, "prices": prices, "count": len(prices)}

    except Exception as e:
        logger.error(f"/status/market hatası: {e}", exc_info=True)
        if _market_cache["prices"]:
            return {"ok": True, "prices": _market_cache["prices"], "cached": True, "stale": True}
        return {"ok": False, "prices": {}, "reason": str(e)}


@app.get("/status/signals")
async def status_signals(request: Request, limit: int = Query(100)):
    """
    Sinyal geçmişini döner. Dashboard'un SİNYALLER sekmesini doldurur.
    """
    try:
        se = request.app.state.signal_engine
        status = {}
        try:
            status = se.get_status()
        except Exception:
            pass
        signals = status.get("signal_history", [])
        # En yeni önce
        signals_sorted = sorted(signals, key=lambda x: x.get("time", ""), reverse=True)
        return {"ok": True, "signals": signals_sorted[:limit], "total": len(signals_sorted)}
    except Exception as e:
        logger.error(f"/status/signals hatası: {e}")
        return {"ok": True, "signals": [], "total": 0}


@app.get("/status/positions")
async def status_positions(request: Request):
    """
    Pozisyonlar için kısa yol fallback (dashboard'un /execution/positions/live'a alternatifi).
    """
    try:
        dc = request.app.state.data_client
        if not dc.exchange or not getattr(dc, "_auth_ok", False):
            # Auth yoksa risk engine'den çek
            re = request.app.state.risk_engine
            positions = _positions_safe(re)
            return {"ok": True, "positions": positions, "source": "risk_engine"}
        # Binance'den canlı çek
        from fastapi import Request as _Req
        return await live_positions(request)
    except Exception as e:
        return {"ok": False, "positions": [], "reason": str(e)}
