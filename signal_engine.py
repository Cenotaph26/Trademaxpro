"""
Otomatik Sinyal Motoru — Her 15 dakikada çalışır.
EMA, RSI, MACD, Bollinger, ATR Breakout, Volume Spike, Haber Sentiment
tüm sinyalleri birleştirir → RL agent denetler → işlem açar.
"""
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Teknik İndikatörler ──────────────────────────────────────────────────────

def ema(closes: list, period: int) -> float:
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    k = 2 / (period + 1)
    val = sum(closes[:period]) / period
    for c in closes[period:]:
        val = c * k + val * (1 - k)
    return val


def rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in diffs if d > 0]
    losses = [-d for d in diffs if d < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 1e-9
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(closes: list, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # Signal line: EMA of MACD (approximate)
    macd_vals = []
    for i in range(slow, len(closes)):
        ef = ema(closes[:i+1], fast)
        es = ema(closes[:i+1], slow)
        macd_vals.append(ef - es)
    signal_line = ema(macd_vals, signal) if len(macd_vals) >= signal else macd_line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(closes: list, period=20, std_mult=2.0):
    if len(closes) < period:
        mid = closes[-1]
        return mid, mid, mid
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((c - mid) ** 2 for c in window) / period
    std = math.sqrt(variance)
    return mid + std_mult * std, mid, mid - std_mult * std


def atr(candles: list, period=14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def volume_spike(candles: list, period=20, threshold=2.0) -> bool:
    if len(candles) < period + 1:
        return False
    vols = [c["volume"] for c in candles]
    avg_vol = sum(vols[-period-1:-1]) / period
    current_vol = vols[-1]
    return current_vol > avg_vol * threshold


def stochastic_rsi(closes: list, period=14) -> float:
    """StochRSI — RSI'nın aşırı noktalarını tespit eder."""
    if len(closes) < period * 2:
        return 50.0
    rsi_vals = []
    for i in range(period, len(closes)):
        rsi_vals.append(rsi(closes[max(0, i-period):i+1], period))
    if not rsi_vals:
        return 50.0
    window = rsi_vals[-period:]
    min_rsi = min(window)
    max_rsi = max(window)
    if max_rsi == min_rsi:
        return 50.0
    return (rsi_vals[-1] - min_rsi) / (max_rsi - min_rsi) * 100


def obv(candles: list) -> float:
    """On-Balance Volume — fiyat yönünü volume ile doğrular."""
    if len(candles) < 2:
        return 0.0
    obv_val = 0.0
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i-1]["close"]:
            obv_val += candles[i]["volume"]
        elif candles[i]["close"] < candles[i-1]["close"]:
            obv_val -= candles[i]["volume"]
    return obv_val


# ─── Haber Sentiment ──────────────────────────────────────────────────────────

async def fetch_news_sentiment(symbol: str) -> float:
    """
    CryptoCompare News API'den haber çeker, sentiment skoru döner.
    -1.0 (çok negatif) → +1.0 (çok pozitif)
    """
    try:
        import aiohttp
        coin = symbol.replace("USDT", "").replace("BUSD", "")
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&lang=EN&sortOrder=latest"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
                articles = data.get("Data", [])[:10]
                if not articles:
                    return 0.0

                # Basit keyword sentiment analizi
                positive_words = [
                    "surge", "rally", "bullish", "breakout", "gain", "rise",
                    "pump", "up", "high", "growth", "adoption", "partnership",
                    "launch", "upgrade", "positive", "buy", "long", "moon"
                ]
                negative_words = [
                    "crash", "drop", "bearish", "sell", "down", "loss", "hack",
                    "ban", "regulation", "fear", "dump", "low", "negative",
                    "liquidation", "risk", "warning", "concern", "decline"
                ]

                scores = []
                for article in articles:
                    text = (article.get("title", "") + " " + article.get("body", "")[:200]).lower()
                    pos = sum(1 for w in positive_words if w in text)
                    neg = sum(1 for w in negative_words if w in text)
                    total = pos + neg
                    if total > 0:
                        scores.append((pos - neg) / total)

                return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.debug(f"Haber sentiment alınamadı: {e}")
        return 0.0  # Haber alınamazsa nötr


# ─── Ana Sinyal Motoru ────────────────────────────────────────────────────────

class SignalScore:
    def __init__(self):
        self.scores: dict = {}
        self.details: list = []

    def add(self, name: str, score: float, reason: str):
        """score: -1 (güçlü sat) → +1 (güçlü al)"""
        self.scores[name] = score
        self.details.append(f"{name}: {score:+.2f} ({reason})")

    @property
    def total(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def side(self) -> Optional[str]:
        if self.total > 0.25:
            return "BUY"
        elif self.total < -0.25:
            return "SELL"
        return None  # Sinyal yok

    @property
    def strength(self) -> str:
        t = abs(self.total)
        if t > 0.7:
            return "güçlü"
        elif t > 0.4:
            return "orta"
        return "zayıf"


class AutoSignalEngine:
    def __init__(self, data_client, strategy_manager, risk_engine, rl_agent, settings):
        self.data = data_client
        self.strategy = strategy_manager
        self.risk = risk_engine
        self.rl = rl_agent
        self.s = settings
        self._running = False
        self.last_signal: Optional[dict] = None
        self.signal_count = 0
        self.scan_interval = 15 * 60  # 15 dakika

    async def start(self):
        self._running = True
        logger.info("🤖 Otomatik Sinyal Motoru başlatıldı (15 dk tarama)")
        # İlk taramadan önce veri yüklenmesini bekle
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._scan_and_trade()
            except Exception as e:
                logger.error(f"Sinyal motoru hatası: {e}")
            await asyncio.sleep(self.scan_interval)

    async def _scan_and_trade(self):
        ds = self.data.state
        if not ds.mark_price or not ds.updated_at:
            logger.warning("Piyasa verisi henüz yok, tarama atlandı")
            return

        symbol = self.s.SYMBOL
        candles_1h = list(ds.klines_1h)
        candles_5m = list(ds.klines_5m)

        if len(candles_1h) < 50:
            logger.warning("Yetersiz kline verisi, tarama atlandı")
            return

        closes_1h = [c["close"] for c in candles_1h]
        closes_5m = [c["close"] for c in candles_5m] if candles_5m else closes_1h
        current_price = ds.mark_price

        score = SignalScore()

        # ── 1. EMA Kesişimi ────────────────────────────────────────
        ema9  = ema(closes_1h, 9)
        ema21 = ema(closes_1h, 21)
        ema50 = ema(closes_1h, 50)

        if ema9 > ema21 > ema50:
            score.add("EMA", 0.8, f"EMA9({ema9:.0f}) > EMA21({ema21:.0f}) > EMA50({ema50:.0f}) — güçlü yükseliş trendi")
        elif ema9 > ema21:
            score.add("EMA", 0.4, f"EMA9 > EMA21 — kısa vadeli yükseliş")
        elif ema9 < ema21 < ema50:
            score.add("EMA", -0.8, f"EMA9({ema9:.0f}) < EMA21({ema21:.0f}) < EMA50({ema50:.0f}) — güçlü düşüş trendi")
        elif ema9 < ema21:
            score.add("EMA", -0.4, "EMA9 < EMA21 — kısa vadeli düşüş")
        else:
            score.add("EMA", 0.0, "EMA nötr")

        # ── 2. RSI ────────────────────────────────────────────────
        rsi_val = rsi(closes_1h, 14)
        rsi_5m  = rsi(closes_5m, 14) if len(closes_5m) > 14 else rsi_val

        if rsi_val < 30:
            score.add("RSI", 0.9, f"RSI={rsi_val:.1f} — aşırı satım, güçlü alım sinyali")
        elif rsi_val < 40:
            score.add("RSI", 0.5, f"RSI={rsi_val:.1f} — satım bölgesi")
        elif rsi_val > 70:
            score.add("RSI", -0.9, f"RSI={rsi_val:.1f} — aşırı alım, güçlü satım sinyali")
        elif rsi_val > 60:
            score.add("RSI", -0.5, f"RSI={rsi_val:.1f} — alım bölgesi")
        else:
            score.add("RSI", 0.0, f"RSI={rsi_val:.1f} — nötr bölge")

        # ── 3. MACD ───────────────────────────────────────────────
        macd_line, signal_line, histogram = macd(closes_1h)

        if histogram > 0 and macd_line > signal_line:
            strength = min(abs(histogram) / (abs(macd_line) + 1e-9), 1.0)
            score.add("MACD", 0.6 * strength, f"MACD pozitif hist={histogram:.2f}")
        elif histogram < 0 and macd_line < signal_line:
            strength = min(abs(histogram) / (abs(macd_line) + 1e-9), 1.0)
            score.add("MACD", -0.6 * strength, f"MACD negatif hist={histogram:.2f}")
        else:
            score.add("MACD", 0.0, "MACD nötr")

        # ── 4. Bollinger Bands ────────────────────────────────────
        bb_upper, bb_mid, bb_lower = bollinger(closes_1h, 20, 2.0)
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)

        if current_price < bb_lower:
            score.add("BB", 0.7, f"Fiyat alt band altında ({current_price:.0f} < {bb_lower:.0f})")
        elif current_price > bb_upper:
            score.add("BB", -0.7, f"Fiyat üst band üstünde ({current_price:.0f} > {bb_upper:.0f})")
        elif current_price < bb_mid and bb_width > 0.02:
            score.add("BB", 0.2, "Bandın alt yarısı — hafif alım")
        elif current_price > bb_mid and bb_width > 0.02:
            score.add("BB", -0.2, "Bandın üst yarısı — hafif satım")
        else:
            score.add("BB", 0.0, "BB nötr")

        # ── 5. ATR Breakout ───────────────────────────────────────
        atr_val = ds.atr_14
        if atr_val > 0 and len(candles_1h) > 2:
            prev_high = max(c["high"] for c in candles_1h[-4:-1])
            prev_low  = min(c["low"] for c in candles_1h[-4:-1])
            breakout_threshold = atr_val * 0.5

            if current_price > prev_high + breakout_threshold:
                score.add("ATR_BREAKOUT", 0.85, f"Üst breakout: {current_price:.0f} > {prev_high:.0f} + {breakout_threshold:.0f}")
            elif current_price < prev_low - breakout_threshold:
                score.add("ATR_BREAKOUT", -0.85, f"Alt breakout: {current_price:.0f} < {prev_low:.0f} - {breakout_threshold:.0f}")
            else:
                score.add("ATR_BREAKOUT", 0.0, "Breakout yok — konsolidasyon")

        # ── 6. Volume Spike ───────────────────────────────────────
        if candles_1h:
            vol_spike = volume_spike(candles_1h, period=20, threshold=1.8)
            price_up = current_price > closes_1h[-2] if len(closes_1h) > 1 else True
            if vol_spike and price_up:
                score.add("VOLUME", 0.6, "Yüksek hacimli yükseliş mumu — güçlü alım baskısı")
            elif vol_spike and not price_up:
                score.add("VOLUME", -0.6, "Yüksek hacimli düşüş mumu — güçlü satış baskısı")
            else:
                score.add("VOLUME", 0.0, "Normal hacim")

        # ── 7. StochRSI ───────────────────────────────────────────
        srsi = stochastic_rsi(closes_1h, 14)
        if srsi < 20:
            score.add("STOCH_RSI", 0.7, f"StochRSI={srsi:.1f} — aşırı satım")
        elif srsi > 80:
            score.add("STOCH_RSI", -0.7, f"StochRSI={srsi:.1f} — aşırı alım")
        else:
            score.add("STOCH_RSI", 0.0, f"StochRSI={srsi:.1f} — nötr")

        # ── 8. OBV Trendi ─────────────────────────────────────────
        if len(candles_1h) >= 20:
            obv_current = obv(candles_1h[-20:])
            obv_prev    = obv(candles_1h[-40:-20]) if len(candles_1h) >= 40 else 0
            if obv_current > obv_prev * 1.1:
                score.add("OBV", 0.4, "OBV yükseliyor — alım baskısı güçlü")
            elif obv_current < obv_prev * 0.9:
                score.add("OBV", -0.4, "OBV düşüyor — satış baskısı güçlü")
            else:
                score.add("OBV", 0.0, "OBV nötr")

        # ── 9. Funding Rate Filtresi ──────────────────────────────
        funding = ds.funding_rate
        if abs(funding) > 0.005:
            if funding > 0:
                score.add("FUNDING", -0.3, f"Yüksek pozitif funding ({funding:.4f}) — longlar baskı altında")
            else:
                score.add("FUNDING", 0.3, f"Negatif funding ({funding:.4f}) — shortlar baskı altında")

        # ── 10. Haber Sentiment ───────────────────────────────────
        news_score = await fetch_news_sentiment(symbol)
        if abs(news_score) > 0.1:
            score.add("NEWS", news_score * 0.5, f"Haber sentiment: {news_score:+.2f}")

        # ── Sonuç ─────────────────────────────────────────────────
        total = score.total
        side  = score.side

        logger.info(
            f"📊 Sinyal Taraması [{symbol}] | Skor: {total:+.3f} | "
            f"Yön: {side or 'YOK'} | Güç: {score.strength} | "
            f"Rejim: {ds.regime}"
        )
        for detail in score.details:
            logger.info(f"   → {detail}")

        if side is None:
            logger.info("⏭ Yeterli sinyal yok, işlem atlandı")
            self.last_signal = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": None,
                "score": round(total, 3),
                "strength": score.strength,
                "details": score.details,
            }
            return

        # ── RL Agent Denetimi ─────────────────────────────────────
        if self.rl:
            decision = self.rl.decide()
            if not decision.trade_allowed:
                logger.info(f"🤖 RL agent işlemi engelledi (ε={self.rl.epsilon:.3f})")
                return

            # RL sinyali indikatör sinyaliyle çelişiyorsa geri çekil
            # (exploration sırasında %50 ihtimalle yine de devam et)
            import random
            if decision.strategy == "GRID" and ds.regime == "trend" and random.random() > 0.5:
                logger.info("🤖 RL: GRID seçti ama trend rejimi — SMART'a geçiliyor")
                decision_strategy = "SMART"
            else:
                decision_strategy = decision.strategy

            logger.info(
                f"🤖 RL Onayı: {decision_strategy} | {decision.risk_mode} | "
                f"kaldıraç≤{decision.leverage_cap}x | ε={self.rl.epsilon:.3f}"
            )

        # ── İşlem Aç ─────────────────────────────────────────────
        signal_payload = {
            "symbol": symbol,
            "side": side,
            "timeframe": "1h",
            "strategy_tag": "auto_signal",
            "entry_hint": current_price,
            "secret": self.s.WEBHOOK_SECRET,
        }

        logger.info(f"🚀 Otomatik işlem açılıyor: {symbol} {side} (skor={total:+.3f})")
        result = await self.strategy.handle_signal(signal_payload)

        self.signal_count += 1
        self.last_signal = {
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "score": round(total, 3),
            "strength": score.strength,
            "strategy": result.get("strategy"),
            "ok": result.get("ok"),
            "details": score.details,
        }

        if result.get("ok"):
            logger.info(f"✅ İşlem açıldı: {result}")
        else:
            logger.warning(f"❌ İşlem reddedildi: {result.get('reason')}")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "scan_interval_min": self.scan_interval // 60,
            "signal_count": self.signal_count,
            "last_signal": self.last_signal,
        }
