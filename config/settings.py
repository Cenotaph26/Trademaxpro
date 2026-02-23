"""
Tüm ayarlar — .env'den okunur, Railway environment variables ile çalışır.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # ── Railway / Server ──────────────────────────────────────────
    PORT: int = Field(default=8000, env="PORT")
    ENV: str = Field(default="production", env="ENV")
    SECRET_KEY: str = Field(default="change-me-in-prod", env="SECRET_KEY")
    WEBHOOK_SECRET: str = Field(default="tv-secret", env="WEBHOOK_SECRET")

    # ── Binance Futures ───────────────────────────────────────────
    BINANCE_API_KEY: str = Field(default="", env="BINANCE_API_KEY")
    BINANCE_API_SECRET: str = Field(default="", env="BINANCE_API_SECRET")
    BINANCE_TESTNET: bool = Field(default=True, env="BINANCE_TESTNET")
    SYMBOL: str = Field(default="BTCUSDT", env="SYMBOL")
    BASE_LEVERAGE: int = Field(default=3, env="BASE_LEVERAGE")
    POSITION_MODE: str = Field(default="ONE_WAY", env="POSITION_MODE")  # ONE_WAY veya HEDGE

    # ── Risk Engine ───────────────────────────────────────────────
    DAILY_MAX_LOSS_PCT: float = Field(default=2.0, env="DAILY_MAX_LOSS_PCT")   # % bakiyeden
    MAX_OPEN_POSITIONS: int = Field(default=10, env="MAX_OPEN_POSITIONS")
    MAX_SAME_DIRECTION: int = Field(default=10, env="MAX_SAME_DIRECTION")       # artık risk engine'de kullanılmıyor
    MAX_DRAWDOWN_PCT: float = Field(default=5.0, env="MAX_DRAWDOWN_PCT")
    KILL_SWITCH_CONSECUTIVE_LOSS: int = Field(default=5, env="KILL_SWITCH_CONSECUTIVE_LOSS")
    KILL_SWITCH_SLIPPAGE_PCT: float = Field(default=0.5, env="KILL_SWITCH_SLIPPAGE_PCT")
    RISK_PER_TRADE_PCT: float = Field(default=1.0, env="RISK_PER_TRADE_PCT")   # bakiyenin %1'i

    # ── Leverage Caps per Mode ────────────────────────────────────
    LEV_CONSERVATIVE: int = Field(default=2, env="LEV_CONSERVATIVE")
    LEV_NORMAL: int = Field(default=3, env="LEV_NORMAL")
    LEV_AGGRESSIVE: int = Field(default=5, env="LEV_AGGRESSIVE")

    # ── DCA Defaults ──────────────────────────────────────────────
    DCA_MAX_STEPS: int = Field(default=5, env="DCA_MAX_STEPS")
    DCA_STEP_SPACING_ATR: float = Field(default=1.0, env="DCA_STEP_SPACING_ATR")
    DCA_SIZE_MULTIPLIER: float = Field(default=1.2, env="DCA_SIZE_MULTIPLIER")
    DCA_STOPOUT_R: float = Field(default=1.5, env="DCA_STOPOUT_R")

    # ── Grid Defaults ─────────────────────────────────────────────
    GRID_LEVELS: int = Field(default=20, env="GRID_LEVELS")
    GRID_WIDTH_ATR: float = Field(default=3.0, env="GRID_WIDTH_ATR")

    # ── SmartTrade Defaults ───────────────────────────────────────
    ST_TP1_PCT: float = Field(default=50.0, env="ST_TP1_PCT")   # TP1'de kapat %
    ST_TP2_PCT: float = Field(default=50.0, env="ST_TP2_PCT")
    ST_TRAILING: bool = Field(default=True, env="ST_TRAILING")
    ST_TRAILING_CALLBACK_PCT: float = Field(default=0.5, env="ST_TRAILING_CALLBACK_PCT")

    # ── RL Agent ──────────────────────────────────────────────────
    RL_MODEL_PATH: str = Field(default="models/rl_agent.pkl", env="RL_MODEL_PATH")
    RL_LEARNING_RATE: float = Field(default=0.001, env="RL_LEARNING_RATE")
    RL_DISCOUNT: float = Field(default=0.95, env="RL_DISCOUNT")
    RL_EPSILON: float = Field(default=0.15, env="RL_EPSILON")       # exploration
    RL_BUFFER_SIZE: int = Field(default=10000, env="RL_BUFFER_SIZE")
    RL_BATCH_SIZE: int = Field(default=64, env="RL_BATCH_SIZE")
    RL_UPDATE_EVERY: int = Field(default=100, env="RL_UPDATE_EVERY")  # her N adımda train

    # ── Funding Filter ────────────────────────────────────────────
    FUNDING_EXTREME_THRESHOLD: float = Field(default=0.01, env="FUNDING_EXTREME_THRESHOLD")  # %1

    # ── Notification ─────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
