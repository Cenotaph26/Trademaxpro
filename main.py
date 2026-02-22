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
import uvicorn

from config.settings import settings
from data.binance_client import BinanceDataClient
from risk.engine import RiskEngine
from strategies.manager import StrategyManager
from rl_agent.agent import RLAgent
from api.webhook import router as webhook_router
from api.status import router as status_router
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
    await data_client.connect()

    risk_engine = RiskEngine(settings)
    strategy_manager = StrategyManager(data_client, risk_engine, settings)
    rl_agent = RLAgent(settings)
    await rl_agent.load_model()

    # Inject RL agent into strategy manager
    strategy_manager.set_rl_agent(rl_agent)

    # Start background tasks
    asyncio.create_task(data_client.stream_market_data())
    asyncio.create_task(risk_engine.monitor_loop())
    asyncio.create_task(rl_agent.learning_loop())

    # Share with routers
    app.state.data_client = data_client
    app.state.risk_engine = risk_engine
    app.state.strategy_manager = strategy_manager
    app.state.rl_agent = rl_agent

    logger.info("✅ Bot hazır!")
    yield

    # Cleanup
    logger.info("🛑 Bot kapatılıyor...")
    await strategy_manager.close_all_positions()
    await data_client.disconnect()


app = FastAPI(
    title="Binance Futures Bot",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(webhook_router, prefix="/webhook")
app.include_router(status_router, prefix="/status")


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
