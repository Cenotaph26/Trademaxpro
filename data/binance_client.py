"""
Binance Futures veri katmanı.
Mark price, kline, funding rate, open interest, orderbook.
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime

import ccxt.async_support as ccxt
from collections import deque

logger = logging.getLogger(__name__)


class MarketState:
    """Anlık piyasa snapshot'ı — thread-safe değil ama asyncio single-threaded."""
    def __init__(self):
        self.mark_price: float = 0.0
        self.last_price: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.spread_pct: float = 0.0
        self.funding_rate: float = 0.0
        self.next_funding_time: Optional[datetime] = None
        self.open_interest: float = 0.0
        self.atr_14: float = 0.0
        self.volatility_1h: float = 0.0
        self.regime: str = "unknown"  # trend / range / volatile
        self.klines_1m: deque = deque(maxlen=500)
        self.klines_5m: deque = deque(maxlen=300)
        self.klines_1h: deque = deque(maxlen=200)
        self.updated_at: Optional[datetime] = None


class BinanceDataClient:
    def __init__(self):
        from config.settings import settings
        self.settings = settings
        self.exchange: Optional[ccxt.binanceusdm] = None
        self.state = MarketState()
        self._running = False
        self._auth_ok = False  # auth başarılı mı?

    async def connect(self):
        params = {
            "apiKey": self.settings.BINANCE_API_KEY,
            "secret": self.settings.BINANCE_API_SECRET,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
        if self.settings.BINANCE_TESTNET:
            params["urls"] = {
                "api": {
                    "fapiPublic":    "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivate":   "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPublicV2":  "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
                    "public":        "https://demo-fapi.binance.com/fapi/v1",
                    "private":       "https://demo-fapi.binance.com/fapi/v1",
                }
            }

        self.exchange = ccxt.binanceusdm(params)
        try:
            await self.exchange.load_markets()
            self._auth_ok = True
            logger.info(f"✅ Binance bağlandı (testnet={self.settings.BINANCE_TESTNET})")
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Binance AUTH hatası — API key geçersiz: {e}")
            logger.warning("Bot kısmi modda çalışıyor — healthcheck geçecek")
            # raise etmiyoruz
        except Exception as e:
            logger.error(f"❌ Binance load_markets hatası: {e}")
            logger.warning("Bot kısmi modda çalışıyor — healthcheck geçecek")
            # raise etmiyoruz

    async def disconnect(self):
        self._running = False
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.warning(f"Exchange close hatası: {e}")

    # ─── Public data fetchers ─────────────────────────────────────

    async def fetch_mark_price(self) -> float:
        ticker = await self.exchange.fetch_ticker(self.settings.SYMBOL)
        self.state.mark_price = ticker.get("markPrice") or ticker["last"]
        self.state.last_price = ticker["last"]
        self.state.bid = ticker["bid"] or ticker["last"]
        self.state.ask = ticker["ask"] or ticker["last"]
        if self.state.ask and self.state.bid:
            self.state.spread_pct = (self.state.ask - self.state.bid) / self.state.ask * 100
        return self.state.mark_price

    async def fetch_funding_rate(self):
        try:
            info = await self.exchange.fetch_funding_rate(self.settings.SYMBOL)
            self.state.funding_rate = info.get("fundingRate", 0.0)
            self.state.next_funding_time = info.get("fundingDatetime")
        except Exception as e:
            logger.warning(f"Funding rate alınamadı: {e}")

    async def fetch_open_interest(self):
        try:
            oi = await self.exchange.fetch_open_interest(self.settings.SYMBOL)
            self.state.open_interest = oi.get("openInterestAmount", 0.0)
        except Exception as e:
            logger.warning(f"OI alınamadı: {e}")

    async def fetch_klines(self, timeframe: str = "1h", limit: int = 200):
        ohlcv = await self.exchange.fetch_ohlcv(
            self.settings.SYMBOL, timeframe=timeframe, limit=limit
        )
        candles = [
            {
                "ts": c[0], "open": c[1], "high": c[2],
                "low": c[3], "close": c[4], "volume": c[5]
            }
            for c in ohlcv
        ]
        if timeframe == "1m":
            self.state.klines_1m.clear()
            self.state.klines_1m.extend(candles)
        elif timeframe == "5m":
            self.state.klines_5m.clear()
            self.state.klines_5m.extend(candles)
        elif timeframe == "1h":
            self.state.klines_1h.clear()
            self.state.klines_1h.extend(candles)
            self._compute_atr_and_regime()
        return candles

    # ─── Computed indicators ──────────────────────────────────────

    def _compute_atr_and_regime(self, period: int = 14):
        candles = list(self.state.klines_1h)
        if len(candles) < period + 1:
            return
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]
            l = candles[i]["low"]
            pc = candles[i - 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(trs[-period:]) / period
        self.state.atr_14 = atr

        closes = [c["close"] for c in candles[-21:]]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            self.state.volatility_1h = variance ** 0.5

        recent = candles[-20:]
        total_range = sum(c["high"] - c["low"] for c in recent) / 20
        net_move = abs(recent[-1]["close"] - recent[0]["close"])
        ratio = net_move / (total_range + 1e-9)

        if self.state.volatility_1h > 0.015:
            self.state.regime = "volatile"
        elif ratio > 0.5:
            self.state.regime = "trend"
        else:
            self.state.regime = "range"

    async def get_balance(self) -> dict:
        balance = await self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": usdt.get("total", 0.0),
            "free": usdt.get("free", 0.0),
            "used": usdt.get("used", 0.0),
        }

    async def get_positions(self) -> list:
        positions = await self.exchange.fetch_positions([self.settings.SYMBOL])
        return [p for p in positions if abs(p.get("contracts", 0) or 0) > 0]

    # ─── Continuous stream ────────────────────────────────────────

    async def stream_market_data(self):
        self._running = True
        logger.info("Market data stream başlatıldı")
        while self._running:
            try:
                # Exchange hiç init edilmediyse bekle
                if self.exchange is None:
                    await asyncio.sleep(30)
                    continue

                # Auth başarısızsa tekrar bağlanmayı dene (her 60sn)
                if not self._auth_ok:
                    logger.warning("⏳ Auth yok — 60sn sonra tekrar denenecek")
                    await asyncio.sleep(60)
                    try:
                        await self.connect()
                    except Exception:
                        pass
                    continue

                await self.fetch_mark_price()
                await self.fetch_funding_rate()
                await self.fetch_klines("1h", 200)
                await self.fetch_klines("5m", 100)
                self.state.updated_at = datetime.utcnow()

            except ccxt.AuthenticationError as e:
                logger.error(f"❌ Stream auth hatası: {e}")
                self._auth_ok = False
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Market data hatası: {e}")
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(10)
