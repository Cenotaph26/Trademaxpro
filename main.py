"""
Binance Futures Trading Bot
DCA + Grid + SmartTrade + TradingView + RL Agent
Railway deployment ready
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from config.settings import settings
from data.binance_client import BinanceDataClient
from risk.engine import RiskEngine
from strategies.manager import StrategyManager
from rl_agent.agent import RLAgent
from api.webhook import router as webhook_router
from api.status import router as status_router
from api.execution import router as execution_router
from signal_engine import AutoSignalEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global state
data_client: BinanceDataClient = None
risk_engine: RiskEngine = None
strategy_manager: StrategyManager = None
rl_agent: RLAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_client, risk_engine, strategy_manager, rl_agent

    logger.info("🚀 Trading Bot başlatılıyor...")

    # Initialize components
    data_client = BinanceDataClient()

    # Binance bağlantısı — hata olursa bot yine de ayağa kalkar
    try:
        await data_client.connect()
    except Exception as e:
        logger.error(f"Binance connect hatası: {e} — bot kısmi modda çalışacak")

    risk_engine = RiskEngine(settings)
    strategy_manager = StrategyManager(data_client, risk_engine, settings)
    rl_agent = RLAgent(settings)

    try:
        await rl_agent.load_model()
    except Exception as e:
        logger.warning(f"RL model yüklenemedi: {e}")

    strategy_manager.set_rl_agent(rl_agent)

    # Otomatik sinyal motoru
    signal_engine = AutoSignalEngine(
        data_client, strategy_manager, risk_engine, rl_agent, settings
    )

    # Background tasks
    asyncio.create_task(data_client.stream_market_data())
    asyncio.create_task(risk_engine.monitor_loop())
    asyncio.create_task(rl_agent.learning_loop())
    asyncio.create_task(signal_engine.start())

    # Share with routers
    app.state.data_client = data_client
    app.state.risk_engine = risk_engine
    app.state.strategy_manager = strategy_manager
    app.state.rl_agent = rl_agent
    app.state.signal_engine = signal_engine

    logger.info("✅ Bot hazır!")
    yield

    # Cleanup
    logger.info("🛑 Bot kapatılıyor...")
    try:
        await strategy_manager.close_all_positions()
    except Exception as e:
        logger.error(f"Pozisyon kapatma hatası: {e}")
    try:
        await data_client.disconnect()
    except Exception as e:
        logger.error(f"Disconnect hatası: {e}")


app = FastAPI(
    title="Binance Futures Bot",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhook_router, prefix="/webhook")
app.include_router(status_router, prefix="/status")
app.include_router(execution_router, prefix="/execution")


@app.get("/")
async def dashboard():
    return FileResponse("dashboard.html")


def handle_shutdown(sig, frame):
    logger.warning(f"Signal {sig} alındı, kapatılıyor...")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        reload=False
    )
