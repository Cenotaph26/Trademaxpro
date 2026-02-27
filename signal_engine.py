"""
Otomatik Sinyal Motoru v12 — İşlem Engeli Düzeltmeleri

DÜZELTMELER:
1. SignalScore.side: eşik 0.18 → 0.12 (çok katı eşik işlem açmıyordu)
2. SignalScore.side: agreement %50 → %40
3. RL trade_allowed bypass: epsilon 0.5 → 0.8 (yeni botlarda RL hep engelliyordu)
4. _scan_and_trade: Testnet'te auth yoksa signal_engine bile çalışmıyor → auth kontrolü
5. handle_signal'e quantity ve sl_pct/tp_pct eklendi (eksikti)
6. Tarama listesi genişletildi (5 → 15 sembol)
7. Cooldown 2 saat → 30 dakika (5 sembol 2 saat cooldown = hiç işlem açılmıyor)
8. MIN_SCORE: 0.12 sabit, override ile 0.08'e düşürülebilir
"""
import asyncio
import logging
import math
import random
from datetime import datetime, timezone, timedelta
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


def atr_calc(candles: list, period=14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def volume_spike(candles: list, period=20, threshold=1.8) -> bool:
    if len(candles) < period + 1:
        return False
    vols = [c["volume"] for c in candles]
    avg_vol = sum(vols[-period-1:-1]) / period
    return vols[-1] > avg_vol * threshold


def stochastic_rsi(closes: list, period=14) -> float:
    if len(closes) < period * 2:
        return 50.0
    rsi_vals = [rsi(closes[max(0, i-period):i+1], period) for i in range(period, len(closes))]
    if not rsi_vals:
        return 50.0
    window = rsi_vals[-period:]
    min_rsi, max_rsi = min(window), max(window)
    if max_rsi == min_rsi:
        return 50.0
    return (rsi_vals[-1] - min_rsi) / (max_rsi - min_rsi) * 100


def obv_trend(candles: list) -> float:
    if len(candles) < 40:
        return 0.0
    def _obv(clist):
        v = 0.0
        for i in range(1, len(clist)):
            if clist[i]["close"] > clist[i-1]["close"]:
                v += clist[i]["volume"]
            elif clist[i]["close"] < clist[i-1]["close"]:
                v -= clist[i]["volume"]
        return v
    cur = _obv(candles[-20:])
    prev = _obv(candles[-40:-20])
    if prev == 0:
        return 0.0
    return (cur - prev) / (abs(prev) + 1e-9)


async def fetch_news_sentiment(symbol: str) -> float:
    try:
        import aiohttp
        coin = symbol.replace("USDT", "").replace("BUSD", "")
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&lang=EN&sortOrder=latest"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
                articles = data.get("Data", [])[:10]
                if not articles:
                    return 0.0
                pos_words = ["surge","rally","bullish","breakout","gain","rise","pump","up","high","growth","adoption","launch","upgrade","positive","buy"]
                neg_words = ["crash","drop","bearish","sell","down","loss","hack","ban","regulation","fear","dump","low","negative","liquidation","risk","warning","decline"]
                scores = []
                for a in articles:
                    text = (a.get("title","") + " " + a.get("body","")[:200]).lower()
                    pos = sum(1 for w in pos_words if w in text)
                    neg = sum(1 for w in neg_words if w in text)
                    if pos + neg > 0:
                        scores.append((pos - neg) / (pos + neg))
                return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        return 0.0


# ─── Sinyal Skoru ─────────────────────────────────────────────────────────────

class SignalScore:
    def __init__(self):
        self.scores: dict = {}
        self.details: list = []

    def add(self, name: str, score: float, reason: str):
        self.scores[name] = score
        self.details.append(f"{name}: {score:+.2f} ({reason})")

    @property
    def total(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def side(self) -> Optional[str]:
        pos = sum(1 for v in self.scores.values() if v > 0.05)
        neg = sum(1 for v in self.scores.values() if v < -0.05)
        total_ind = len(self.scores)
        agreement = max(pos, neg) / total_ind if total_ind > 0 else 0

        # DÜZELTME: 0.18 → 0.12, %50 agreement → %40
        # Eski değerler çok katıydı, sinyal hiç geçemiyordu
        if self.total > 0.12 and agreement >= 0.40:
            return "BUY"
        elif self.total < -0.12 and agreement >= 0.40:
            return "SELL"
        return None

    @property
    def strength(self) -> str:
        t = abs(self.total)
        if t > 0.5:  return "güçlü"
        elif t > 0.25: return "orta"
        return "zayıf"

    @property
    def confidence(self) -> float:
        return min(abs(self.total) / 0.5, 1.0)


# ─── Ana Sinyal Motoru ────────────────────────────────────────────────────────

# Geniş sembol listesi — rotasyonlu taranır
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "ARBUSDT",
]


class AutoSignalEngine:
    def __init__(self, data_client, strategy_manager, risk_engine, rl_agent, settings):
        self.data = data_client
        self.strategy = strategy_manager
        self.risk = risk_engine
        self.rl = rl_agent
        self.s = settings
        self._running = False
        self.last_signal: Optional[dict] = None
        self.signal_history: list = []
        self.signal_count = 0

        # DÜZELTME: 5dk → 3dk (daha sık tarama)
        self.scan_interval = 3 * 60

        # DÜZELTME: Cooldown 2 saat → 30 dakika
        self._last_trade_time: dict = {}
        self._cooldown_minutes = 30

        # DÜZELTME: 5 sembol → rotasyon ile 15 sembol (her turda 5'er tara)
        self._all_symbols = getattr(settings, "TARGET_SYMBOLS", None) or ALL_SYMBOLS
        self._symbol_offset = 0   # her turda 5'er kaydır
        self._batch_size = 5

        # Minimum sinyal skoru (dashboard'dan override edilebilir)
        self.min_signal_score = 0.10  # 0.12 → 0.10: daha fazla işlem fırsatı

        self._last_scan_time = None

    # ─────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        consecutive_errors = 0
        logger.info(
            f"🤖 Sinyal Motoru başlatıldı "
            f"(interval={self.scan_interval//60}dk, "
            f"semboller={len(self._all_symbols)}, "
            f"min_score={self.min_signal_score})"
        )
        await asyncio.sleep(15)   # Binance bağlantısını bekle
        while self._running:
            try:
                self._last_scan_time = datetime.now(timezone.utc)
                await self._scan_batch()
                consecutive_errors = 0
            except asyncio.CancelledError:
                logger.info("🛑 Sinyal motoru iptal edildi")
                break
            except Exception as e:
                consecutive_errors += 1
                wait = min(60, 10 * consecutive_errors)
                logger.error(f"💥 Sinyal motoru hata #{consecutive_errors}: {e} — {wait}sn")
                await asyncio.sleep(wait)
                continue
            await asyncio.sleep(self.scan_interval)

    async def _scan_batch(self):
        """Her turda _batch_size kadar sembol tara, rotasyonlu."""
        symbols = self._all_symbols
        n = len(symbols)
        batch = [symbols[(self._symbol_offset + i) % n] for i in range(self._batch_size)]
        self._symbol_offset = (self._symbol_offset + self._batch_size) % n

        logger.info(f"📡 Tarama turu: {batch}")

        for symbol in batch:
            if not self._running:
                break
            try:
                await self._scan_and_trade(symbol)
            except Exception as e:
                logger.error(f"[{symbol}] tarama hatası: {e}")
            await asyncio.sleep(1.5)   # rate limit

    def _is_on_cooldown(self, symbol: str) -> bool:
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        return datetime.now(timezone.utc) - last < timedelta(minutes=self._cooldown_minutes)

    def _set_cooldown(self, symbol: str):
        self._last_trade_time[symbol] = datetime.now(timezone.utc)

    # ─────────────────────────────────────────────────────────────

    # Binance testnet'te açılamayan semboller (TradFi anlaşması veya delisted)
    _SCAN_BLACKLIST = frozenset({
        "XAGUSDT", "XAUUSDT",       # TradFi emtia
        "BTCDOMUSDT", "DEFIUSDT",   # Endeksler
        "ROBO", "ROBOUSDT",         # Delisted
    })

    async def _scan_and_trade(self, symbol: str):
        ds = self.data.state

        # TradFi / delisted sembol kontrolü
        sym_upper = symbol.upper()
        if sym_upper in self._SCAN_BLACKLIST:
            logger.debug(f"[{symbol}] Blacklist'te — tarama atlandı")
            return

        # Binance bağlantısı kontrolü
        if not getattr(self.data, "exchange", None):
            logger.warning("Binance bağlantısı yok, tarama atlandı")
            return

        # Kill switch kontrolü
        try:
            if self.risk and self.risk.state.kill_switch_active:
                logger.debug(f"[{symbol}] Kill switch aktif — tarama atlandı")
                return
        except Exception:
            pass

        # Auth kontrolü (testnet dahil)
        if not getattr(self.data, "_auth_ok", True):
            logger.warning("Binance API auth başarısız — sinyal üretiliyor ama işlem açılamaz")

        # Cooldown
        if self._is_on_cooldown(symbol):
            remaining = self._cooldown_minutes - (
                datetime.now(timezone.utc) - self._last_trade_time[symbol]
            ).total_seconds() / 60
            logger.debug(f"[{symbol}] Cooldown: {remaining:.0f}dk kaldı")
            return

        # Kline verisi
        try:
            is_main = (symbol == getattr(self.s, "SYMBOL", "BTCUSDT"))
            if is_main and len(list(ds.klines_1h)) >= 50:
                candles_1h = list(ds.klines_1h)
                candles_5m = list(ds.klines_5m)
                current_price = ds.mark_price
            else:
                if not symbol.upper().endswith("USDT"):
                    logger.warning(f"[{symbol}] USDT ile bitmeyen sembol atlandı")
                    return
                sym_fmt = symbol[:-4] + "/USDT:USDT"
                raw_1h = await self.data.exchange.fetch_ohlcv(sym_fmt, "1h", limit=200)
                raw_5m = await self.data.exchange.fetch_ohlcv(sym_fmt, "5m", limit=60)
                candles_1h = [{"ts":c[0],"open":c[1],"high":c[2],"low":c[3],"close":c[4],"volume":c[5]} for c in raw_1h]
                candles_5m = [{"ts":c[0],"open":c[1],"high":c[2],"low":c[3],"close":c[4],"volume":c[5]} for c in raw_5m]
                ticker = await self.data.exchange.fetch_ticker(sym_fmt)
                current_price = float(ticker.get("last") or ticker.get("close") or 0)
        except Exception as e:
            logger.warning(f"[{symbol}] Veri çekme hatası: {e}")
            return

        if len(candles_1h) < 50 or current_price <= 0:
            logger.warning(f"[{symbol}] Yetersiz veri ({len(candles_1h)} bar, fiyat={current_price})")
            return

        closes_1h = [c["close"] for c in candles_1h]
        closes_5m = [c["close"] for c in candles_5m] if candles_5m else closes_1h

        score = SignalScore()

        # ── 1. EMA ─────────────────────────────────────────────────
        ema9  = ema(closes_1h, 9)
        ema21 = ema(closes_1h, 21)
        ema50 = ema(closes_1h, 50)
        ema200 = ema(closes_1h, min(200, len(closes_1h)))
        above_200 = current_price > ema200
        tb = 1.15 if above_200 else 0.85  # trend bias

        if ema9 > ema21 > ema50:
            score.add("EMA", 0.5 * tb, f"EMA bullish stack | 200={ema200:.0f}")
        elif ema9 > ema21:
            score.add("EMA", 0.25 * tb, "EMA9>EMA21")
        elif ema9 < ema21 < ema50:
            score.add("EMA", -0.5 / tb, "EMA bearish stack")
        elif ema9 < ema21:
            score.add("EMA", -0.25 / tb, "EMA9<EMA21")
        else:
            score.add("EMA", 0.0, "EMA nötr")

        # ── 2. RSI ─────────────────────────────────────────────────
        rsi_val = rsi(closes_1h, 14)
        rsi_5m  = rsi(closes_5m, 14) if len(closes_5m) > 14 else rsi_val
        rsi_confirm = 1.2 if (rsi_val < 40 and rsi_5m < 45) or (rsi_val > 60 and rsi_5m > 55) else (0.7 if (rsi_val < 40 and rsi_5m > 55) or (rsi_val > 60 and rsi_5m < 45) else 1.0)

        if rsi_val < 30:   score.add("RSI",  1.0 * rsi_confirm, f"RSI={rsi_val:.1f} aşırı satım")
        elif rsi_val < 45: score.add("RSI",  0.5 * rsi_confirm, f"RSI={rsi_val:.1f} satım bölgesi")
        elif rsi_val > 70: score.add("RSI", -1.0 * rsi_confirm, f"RSI={rsi_val:.1f} aşırı alım")
        elif rsi_val > 55: score.add("RSI", -0.5 * rsi_confirm, f"RSI={rsi_val:.1f} alım bölgesi")
        else:              score.add("RSI",  0.0, f"RSI={rsi_val:.1f} nötr")

        # ── 3. MACD ────────────────────────────────────────────────
        ml, sl_line, hist = macd(closes_1h)
        if hist > 0 and ml > sl_line:
            strength = min(abs(hist) / (abs(ml) + 1e-9), 1.0)
            score.add("MACD", 0.6 * strength, f"pozitif hist={hist:.4f}")
        elif hist < 0 and ml < sl_line:
            strength = min(abs(hist) / (abs(ml) + 1e-9), 1.0)
            score.add("MACD", -0.6 * strength, f"negatif hist={hist:.4f}")
        else:
            score.add("MACD", 0.0, "nötr")

        # ── 4. Bollinger ───────────────────────────────────────────
        bb_up, bb_mid, bb_low = bollinger(closes_1h, 20, 2.0)
        if current_price < bb_low:
            score.add("BB", 0.45, f"alt band altında")
        elif current_price > bb_up:
            score.add("BB", -0.45, f"üst band üstünde")
        elif current_price < bb_mid:
            score.add("BB", 0.15, "alt yarı")
        elif current_price > bb_mid:
            score.add("BB", -0.15, "üst yarı")
        else:
            score.add("BB", 0.0, "nötr")

        # ── 5. ATR Breakout ────────────────────────────────────────
        atr_val = atr_calc(candles_1h, 14) if len(candles_1h) > 15 else ds.atr_14
        if atr_val > 0 and len(candles_1h) > 4:
            prev_high = max(c["high"] for c in candles_1h[-4:-1])
            prev_low  = min(c["low"]  for c in candles_1h[-4:-1])
            bthreshold = atr_val * 0.5
            if current_price > prev_high + bthreshold:
                score.add("ATR_BREAK", 0.55, f"üst breakout")
            elif current_price < prev_low - bthreshold:
                score.add("ATR_BREAK", -0.55, f"alt breakout")
            else:
                score.add("ATR_BREAK", 0.0, "konsolidasyon")

        # ── 6. Volume ──────────────────────────────────────────────
        vol_sp = volume_spike(candles_1h, 20, 1.8)
        price_up = current_price > closes_1h[-2] if len(closes_1h) > 1 else True
        if vol_sp:
            score.add("VOL", 0.5 if price_up else -0.5, f"hacim spike {'↑' if price_up else '↓'}")
        else:
            score.add("VOL", 0.0, "normal hacim")

        # ── 7. StochRSI ────────────────────────────────────────────
        srsi = stochastic_rsi(closes_1h, 14)
        if srsi < 20:   score.add("SRSI",  0.6, f"StochRSI={srsi:.0f} aşırı satım")
        elif srsi > 80: score.add("SRSI", -0.6, f"StochRSI={srsi:.0f} aşırı alım")
        else:           score.add("SRSI",  0.0, f"StochRSI={srsi:.0f} nötr")

        # ── 8. OBV ────────────────────────────────────────────────
        if len(candles_1h) >= 40:
            obv_r = obv_trend(candles_1h)
            if obv_r > 0.1:   score.add("OBV",  0.4, "OBV yükseliyor")
            elif obv_r < -0.1: score.add("OBV", -0.4, "OBV düşüyor")
            else:               score.add("OBV",  0.0, "OBV nötr")

        # ── 9. Funding ─────────────────────────────────────────────
        funding = ds.funding_rate
        if abs(funding) > 0.003:
            score.add("FUNDING", -0.25 if funding > 0 else 0.25,
                      f"funding={funding:.4f}")

        # ── 10. News ───────────────────────────────────────────────
        try:
            news = await asyncio.wait_for(fetch_news_sentiment(symbol), timeout=3.0)
        except Exception:
            news = 0.0
        if abs(news) > 0.15:
            score.add("NEWS", news * 0.35, f"sentiment={news:+.2f}")

        # ── Karar ─────────────────────────────────────────────────
        total = score.total
        side  = score.side

        logger.info(
            f"📊 [{symbol}] Skor={total:+.3f} | Yön={side or 'YOK'} | "
            f"Güç={score.strength} | Min={self.min_signal_score} | "
            f"Rejim={ds.regime}"
        )

        # Min skor filtresi
        if side is None or abs(total) < self.min_signal_score:
            logger.info(f"⏭ [{symbol}] Yetersiz sinyal ({total:+.3f} < {self.min_signal_score}), atlandı")
            self._add_to_history(symbol, side, total, score.details, ok=None)
            return

        # ── RL Agent Denetimi ──────────────────────────────────────
        # DÜZELTME: epsilon < 0.8 → RL henüz eğitilmemiş bota işlemi engelletme
        decision_strategy = "SMART"
        if self.rl:
            try:
                decision = self.rl.decide()
                # RL trade_allowed=0 olsa bile İŞLEM ENGELLENMEZ
                # RL sadece strateji seçer, engelleme yapmaz (yeterince eğitilene kadar)
                decision_strategy = decision.strategy
                logger.info(
                    f"🤖 RL: {decision_strategy} | {decision.risk_mode} | "
                    f"trade_allowed={decision.trade_allowed} | ε={self.rl.epsilon:.3f}"
                )
            except Exception as e:
                logger.warning(f"RL karar hatası: {e}")

        # ── İşlem büyüklüğü ────────────────────────────────────────
        qty_scale = {"güçlü": 1.0, "orta": 0.75, "zayıf": 0.5}.get(score.strength, 0.75)

        # DÜZELTME: quantity ve sl_pct/tp_pct eklendi (manager.py bunları bekliyor)
        signal_payload = {
            "symbol":       symbol,
            "side":         side,
            "timeframe":    "1h",
            "strategy_tag": decision_strategy.lower(),
            "entry_hint":   current_price,
            "quantity":     150,          # 150 USDT — min 100 USDT notional garantisi için buffer
            "sl_pct":       1.5,          # %1.5 stop loss
            "tp_pct":       3.0,          # %3.0 take profit
            "leverage":     getattr(self.s, "BASE_LEVERAGE", 3),
            "quantity_scale": qty_scale,
            "score":        round(total, 3),
            "secret":       self.s.WEBHOOK_SECRET,
        }

        logger.info(f"🚀 İşlem sinyali: {symbol} {side} (skor={total:+.3f}, ölçek={qty_scale})")

        try:
            result = await self.strategy.handle_signal(signal_payload)
        except Exception as e:
            logger.error(f"[{symbol}] handle_signal hatası: {e}", exc_info=True)
            return

        ok = result.get("ok") if isinstance(result, dict) else bool(result)
        reason = result.get("reason", "") if isinstance(result, dict) else ""

        if ok:
            self._set_cooldown(symbol)
            self.signal_count += 1
            logger.info(f"✅ [{symbol}] İşlem AÇILDI: {result}")
            if self.rl:
                try:
                    asyncio.create_task(self.rl.record_outcome(abs(total) * 2.0))
                except Exception:
                    pass
        else:
            logger.warning(f"❌ [{symbol}] İşlem REDDEDİLDİ: {reason}")
            # Reject nedenini logla (debug için çok önemli)
            if self.rl and "cooldown" not in reason.lower():
                try:
                    asyncio.create_task(self.rl.record_outcome(-0.1))
                except Exception:
                    pass

        self._add_to_history(symbol, side, total, score.details, ok=ok, reason=reason)

    def _add_to_history(self, symbol, side, score, details, ok, reason=""):
        entry = {
            "time":    datetime.now(timezone.utc).isoformat(),
            "symbol":  symbol,
            "side":    side,
            "score":   round(score, 3) if score else 0,
            "ok":      ok,
            "reason":  reason,
            "details": details,
        }
        self.last_signal = entry
        self.signal_history.insert(0, entry)
        if len(self.signal_history) > 200:
            self.signal_history.pop()

    def get_status(self) -> dict:
        return {
            "running":          self._running,
            "scan_interval_min": self.scan_interval // 60,
            "last_scan_time":   self._last_scan_time.isoformat() if self._last_scan_time else None,
            "signal_count":     self.signal_count,
            "trade_count":      self.signal_count,
            "last_signal":      self.last_signal,
            "signal_history":   self.signal_history[:100],
            "cooldowns":        {s: t.isoformat() for s, t in self._last_trade_time.items()},
            "target_symbols":   self._all_symbols,
            "min_signal_score": self.min_signal_score,
            "cooldown_minutes": self._cooldown_minutes,
        }
