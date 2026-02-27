"""
Specialist Agents v15 — 4 bağımsız trading ajanı.

Her ajan kendi teknik analizini yapar, Capital Brain'den kapital ister,
ve işlem açar. Birbirlerinden habersizdir — koordinasyon Capital Brain'e aittir.

Ajanlar:
  1. TrendAgent    — EMA, MACD, breakout takibi
  2. MeanRevAgent  — RSI aşırı alım/satım, BB geri dönüş
  3. ScalpAgent    — Kısa vadeli momentum, 5dk grafik
  4. MacroAgent    — Funding rate arb, büyük volatilite fırsatları
"""
import asyncio
import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─── Teknik indikatör helpers (bağımsız) ─────────────────────────────────────

def _ema(closes, p):
    if len(closes) < p:
        return closes[-1] if closes else 0.0
    k = 2 / (p + 1)
    v = sum(closes[:p]) / p
    for c in closes[p:]:
        v = c * k + v * (1 - k)
    return v

def _rsi(closes, p=14):
    if len(closes) < p + 1:
        return 50.0
    diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d for d in diffs if d > 0]
    losses = [-d for d in diffs if d < 0]
    ag = sum(gains[-p:]) / p if gains else 0
    al = sum(losses[-p:]) / p if losses else 1e-9
    return 100 - (100 / (1 + ag / al))

def _macd(closes, fast=12, slow=26, sig=9):
    if len(closes) < slow + sig:
        return 0.0, 0.0
    macd_vals = []
    for i in range(slow, len(closes)):
        ef = _ema(closes[:i+1], fast)
        es = _ema(closes[:i+1], slow)
        macd_vals.append(ef - es)
    signal_line = _ema(macd_vals, sig) if len(macd_vals) >= sig else macd_vals[-1]
    return macd_vals[-1], signal_line

def _atr(candles, p=14):
    if len(candles) < p + 1:
        return 0.0
    trs = [max(c["high"]-c["low"],
               abs(c["high"]-candles[i-1]["close"]),
               abs(c["low"]-candles[i-1]["close"]))
           for i, c in enumerate(candles) if i > 0]
    return sum(trs[-p:]) / p if trs else 0.0

def _bb(closes, p=20):
    if len(closes) < p:
        m = closes[-1]
        return m, m, m
    w = closes[-p:]
    m = sum(w) / p
    s = math.sqrt(sum((c - m)**2 for c in w) / p)
    return m + 2*s, m, m - 2*s

def _stoch_rsi(closes, p=14):
    if len(closes) < p * 2:
        return 50.0
    rsi_vals = [_rsi(closes[max(0, i-p):i+1], p) for i in range(p, len(closes))]
    w = rsi_vals[-p:]
    lo, hi = min(w), max(w)
    return (rsi_vals[-1] - lo) / (hi - lo + 1e-9) * 100


@dataclass
class AgentSignal:
    agent_id: str
    symbol: str
    side: str        # "BUY" | "SELL"
    strength: float  # 0-1
    sl_pct: float
    tp_pct: float
    leverage: int
    reason: str


class BaseSpecialistAgent:
    """Tüm specialist agent'ların ortak altyapısı."""

    AGENT_ID = "base"
    NAME     = "Base Agent"

    def __init__(self, data_client, capital_brain, strategy_manager, settings):
        self.data     = data_client
        self.brain    = capital_brain
        self.manager  = strategy_manager
        self.s        = settings
        self._running = False
        self.scan_interval = 120    # saniye (varsayılan)
        self.symbols: list = []     # Bu agent'ın taradığı semboller

    async def analyze(self, symbol: str, candles_1h: list, candles_5m: list,
                      current_price: float) -> Optional[AgentSignal]:
        """Her agent bu metodu override eder."""
        raise NotImplementedError

    async def _get_candles(self, symbol: str):
        """CCXT ile 1h ve 5m mum verisi çek."""
        from execution.executor import _fmt_symbol
        sym_fmt = _fmt_symbol(symbol)
        try:
            raw_1h = await self.data.exchange.fetch_ohlcv(sym_fmt, "1h", limit=100)
            raw_5m = await self.data.exchange.fetch_ohlcv(sym_fmt, "5m", limit=60)
            c1h = [{"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in raw_1h]
            c5m = [{"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in raw_5m]
            ticker = await self.data.exchange.fetch_ticker(sym_fmt)
            price = float(ticker.get("last") or ticker.get("close") or 0)
            return c1h, c5m, price
        except Exception as e:
            logger.debug(f"[{self.NAME}] Candle hatası {symbol}: {e}")
            return [], [], 0.0

    async def _submit_signal(self, signal: AgentSignal):
        """Capital Brain'e allokasyon iste, onaylanırsa işlem aç."""
        from capital_brain.brain import AllocationRequest
        balance = 0.0
        try:
            bal = await self.data.get_balance()
            balance = float(bal.get("total", 0) or 0)
        except Exception:
            pass

        if balance <= 0:
            return

        # Varsayılan USDT miktarı: bakiyenin %10'u (Brain bunu kırpacak)
        suggested_usdt = balance * 0.10

        req = AllocationRequest(
            agent_id=self.AGENT_ID,
            symbol=signal.symbol,
            side=signal.side,
            signal_strength=signal.strength,
            suggested_usdt=suggested_usdt,
            sl_pct=signal.sl_pct,
            tp_pct=signal.tp_pct,
            leverage=signal.leverage,
            strategy_tag=f"{self.AGENT_ID}_auto",
        )

        result = await self.brain.request_allocation(req)
        if not result.approved:
            logger.debug(f"[{self.NAME}] Allokasyon reddedildi: {result.reason}")
            return

        # İşlem aç
        try:
            outcome = await self.manager.handle_signal({
                "symbol":       signal.symbol,
                "side":         signal.side,
                "quantity":     result.allocated_usdt,
                "leverage":     result.leverage,
                "sl_pct":       signal.sl_pct,
                "tp_pct":       signal.tp_pct,
                "strategy_tag": f"{self.AGENT_ID}_auto",
                "order_type":   "MARKET",
            })
            if outcome and outcome.get("ok"):
                logger.info(
                    f"✅ [{self.NAME}] işlem açıldı: {signal.symbol} {signal.side} "
                    f"${result.allocated_usdt:.0f} @{result.leverage}x | {signal.reason}"
                )
            else:
                logger.warning(f"[{self.NAME}] İşlem başarısız: {outcome}")
                # Allokasyonu geri iade et
                self.brain.record_outcome(self.AGENT_ID, 0, signal.symbol)
        except Exception as e:
            logger.error(f"[{self.NAME}] submit_signal hatası: {e}")

    async def start(self):
        """Ana tarama döngüsü."""
        self._running = True
        logger.info(f"▶ {self.NAME} başlatıldı (interval={self.scan_interval}s)")
        await asyncio.sleep(random.uniform(5, 30))  # Staggered start

        while self._running:
            try:
                for sym in self.symbols:
                    if not self._running:
                        break
                    c1h, c5m, price = await self._get_candles(sym)
                    if not c1h or price <= 0:
                        continue
                    signal = await self.analyze(sym, c1h, c5m, price)
                    if signal:
                        await self._submit_signal(signal)
                    await asyncio.sleep(1)   # Semboller arası kısa bekleme
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.NAME}] döngü hatası: {e}")
            await asyncio.sleep(self.scan_interval)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TREND AGENT — EMA crossover + MACD momentum
# ══════════════════════════════════════════════════════════════════════════════

class TrendAgent(BaseSpecialistAgent):
    """
    Güçlü trend yakalama. Uzun vadeli EMA crossover'lar + MACD histogram.
    Trend yönünde işlem açar, karşı trende girmez.
    """
    AGENT_ID = "trend"
    NAME     = "Trend Agent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_interval = 180   # 3dk — trend değişimi sık değil
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
            "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT",
            "ARBUSDT", "OPUSDT",
        ]

    async def analyze(self, symbol, c1h, c5m, price) -> Optional[AgentSignal]:
        closes = [c["close"] for c in c1h]
        if len(closes) < 50:
            return None

        ema9   = _ema(closes, 9)
        ema21  = _ema(closes, 21)
        ema50  = _ema(closes, 50)
        macd_l, macd_sig = _macd(closes)
        rsi_v  = _rsi(closes)

        # --- LONG sinyal ---
        if (ema9 > ema21 * 1.001 and          # EMA9 > EMA21 (trend yukarı)
                ema21 > ema50 * 0.999 and      # EMA21 > EMA50 (uzun trend yukarı)
                macd_l > macd_sig and          # MACD bullish
                macd_l > 0 and                 # Pozitif MACD
                30 < rsi_v < 70):              # Aşırı alım değil
            strength = min(1.0, (ema9/ema21 - 1) * 50 + 0.5)
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="BUY",
                strength=strength, sl_pct=1.5, tp_pct=3.0, leverage=3,
                reason=f"EMA9>{ema9:.0f}>EMA21>{ema21:.0f} MACD bullish RSI={rsi_v:.0f}"
            )

        # --- SHORT sinyal ---
        if (ema9 < ema21 * 0.999 and
                ema21 < ema50 * 1.001 and
                macd_l < macd_sig and
                macd_l < 0 and
                30 < rsi_v < 70):
            strength = min(1.0, (ema21/ema9 - 1) * 50 + 0.5)
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="SELL",
                strength=strength, sl_pct=1.5, tp_pct=3.0, leverage=3,
                reason=f"EMA9<{ema9:.0f}<EMA21<{ema21:.0f} MACD bearish RSI={rsi_v:.0f}"
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEAN REVERSION AGENT — RSI aşırı seviyeleri + BB dokunuşu
# ══════════════════════════════════════════════════════════════════════════════

class MeanRevAgent(BaseSpecialistAgent):
    """
    Aşırı alınmış/satılmış seviyelerden dönüş oynar.
    RSI < 25 → LONG, RSI > 75 → SHORT. BB bantlarına dokunuşla konfirmasyon.
    """
    AGENT_ID = "meanrev"
    NAME     = "Mean Reversion Agent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_interval = 120
        self.symbols = [
            "ADAUSDT", "XRPUSDT", "DOGEUSDT", "SHIBUSDT",
            "LTCUSDT", "ETCUSDT", "AAVEUSDT", "UNIUSDT",
            "ATOMUSDT", "XLMUSDT",
        ]

    async def analyze(self, symbol, c1h, c5m, price) -> Optional[AgentSignal]:
        closes = [c["close"] for c in c1h]
        if len(closes) < 30:
            return None

        rsi_v      = _rsi(closes)
        rsi_5m     = _rsi([c["close"] for c in c5m], 14) if c5m else rsi_v
        bb_up, bb_mid, bb_low = _bb(closes)
        stoch      = _stoch_rsi(closes)

        # Güçlü aşırı satım → LONG
        if (rsi_v < 28 and rsi_5m < 35 and
                price <= bb_low * 1.01 and
                stoch < 20):
            strength = (30 - rsi_v) / 30
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="BUY",
                strength=strength, sl_pct=1.2, tp_pct=2.5, leverage=2,
                reason=f"RSI aşırı satım {rsi_v:.0f} BB alt bant temas stoch={stoch:.0f}"
            )

        # Güçlü aşırı alım → SHORT
        if (rsi_v > 72 and rsi_5m > 65 and
                price >= bb_up * 0.99 and
                stoch > 80):
            strength = (rsi_v - 70) / 30
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="SELL",
                strength=strength, sl_pct=1.2, tp_pct=2.5, leverage=2,
                reason=f"RSI aşırı alım {rsi_v:.0f} BB üst bant temas stoch={stoch:.0f}"
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCALP AGENT — Kısa vadeli momentum + hacim spike
# ══════════════════════════════════════════════════════════════════════════════

class ScalpAgent(BaseSpecialistAgent):
    """
    Yüksek hacim + momentum kombinasyonu.
    5 dakika grafiğinde ani hareket yakalar.
    Küçük TP/SL, düşük kaldıraç, hızlı giriş-çıkış.
    """
    AGENT_ID = "scalp"
    NAME     = "Scalp Agent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_interval = 90    # 1.5dk — daha sık tarama
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
            "DOGE", "PEPEUSDT", "SHIBUSDT",
        ]

    async def analyze(self, symbol, c1h, c5m, price) -> Optional[AgentSignal]:
        if not c5m or len(c5m) < 20:
            return None

        closes_5m = [c["close"] for c in c5m]
        vols_5m   = [c["volume"] for c in c5m]
        rsi_5m    = _rsi(closes_5m)
        ema5      = _ema(closes_5m, 5)
        ema10     = _ema(closes_5m, 10)

        # Hacim spike
        avg_vol  = sum(vols_5m[-20:-1]) / 19 if len(vols_5m) > 20 else 1
        vol_mult = vols_5m[-1] / (avg_vol + 1e-9)

        # Fiyat değişimi
        price_chg = (closes_5m[-1] - closes_5m[-4]) / (closes_5m[-4] + 1e-9) * 100

        # LONG scalp: hacim spike + yukarı momentum + EMA5 > EMA10
        if (vol_mult > 2.0 and
                price_chg > 0.3 and
                ema5 > ema10 and
                40 < rsi_5m < 75):
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="BUY",
                strength=min(1.0, vol_mult / 4),
                sl_pct=0.8, tp_pct=1.5, leverage=2,
                reason=f"Scalp LONG: vol×{vol_mult:.1f} chg={price_chg:.2f}%"
            )

        # SHORT scalp: hacim spike + aşağı momentum
        if (vol_mult > 2.0 and
                price_chg < -0.3 and
                ema5 < ema10 and
                25 < rsi_5m < 60):
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="SELL",
                strength=min(1.0, vol_mult / 4),
                sl_pct=0.8, tp_pct=1.5, leverage=2,
                reason=f"Scalp SHORT: vol×{vol_mult:.1f} chg={price_chg:.2f}%"
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. MACRO AGENT — Funding arb + volatilite rejimi fırsatları
# ══════════════════════════════════════════════════════════════════════════════

class MacroAgent(BaseSpecialistAgent):
    """
    Makro fırsatlar: Yüksek negatif/pozitif funding, BB sıkışma sonrası patlama,
    güçlü hacim anomalileri, ve multi-timeframe trend konfirmasyonu.
    Daha az ama daha büyük işlemler.
    """
    AGENT_ID = "macro"
    NAME     = "Macro Agent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_interval = 300   # 5dk — sabırlı
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT",
            "AVAXUSDT", "BNBUSDT", "DOTUSDT",
        ]

    async def analyze(self, symbol, c1h, c5m, price) -> Optional[AgentSignal]:
        closes = [c["close"] for c in c1h]
        if len(closes) < 50:
            return None

        # BB sıkışma sonrası patlama tespiti
        bb_up, bb_mid, bb_low = _bb(closes)
        bb_width = (bb_up - bb_low) / (bb_mid + 1e-9)

        # Son 20 mumun BB genişliği — sıkışma var mıydı?
        recent_closes = closes[-25:]
        prev_bb_up, prev_bb_mid, prev_bb_low = _bb(recent_closes[:-5])
        prev_width = (prev_bb_up - prev_bb_low) / (prev_bb_mid + 1e-9)

        bb_expansion = bb_width > prev_width * 1.5  # BB genişledi
        bb_was_tight = prev_width < 0.03              # Önce sıkıştı

        ema21  = _ema(closes, 21)
        ema50  = _ema(closes, 50)
        rsi_v  = _rsi(closes)
        atr_v  = _atr(c1h)

        # BB sıkışma → patlama LONG
        if (bb_expansion and bb_was_tight and
                price > bb_mid and
                ema21 > ema50 and
                45 < rsi_v < 70):
            strength = min(1.0, bb_width / 0.05)
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="BUY",
                strength=strength,
                sl_pct=2.0, tp_pct=5.0, leverage=3,
                reason=f"BB sıkışma patlaması LONG: width={bb_width:.3f} RSI={rsi_v:.0f}"
            )

        # BB sıkışma → patlama SHORT
        if (bb_expansion and bb_was_tight and
                price < bb_mid and
                ema21 < ema50 and
                30 < rsi_v < 55):
            strength = min(1.0, bb_width / 0.05)
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="SELL",
                strength=strength,
                sl_pct=2.0, tp_pct=5.0, leverage=3,
                reason=f"BB sıkışma patlaması SHORT: width={bb_width:.3f} RSI={rsi_v:.0f}"
            )

        # Funding arbitraj: aşırı negatif funding + yukarı trend = LONG
        funding = getattr(getattr(self.data, "state", object()), "funding_rate", 0) or 0
        if (funding < -0.005 and ema21 > ema50 and rsi_v < 60):
            return AgentSignal(
                agent_id=self.AGENT_ID, symbol=symbol, side="BUY",
                strength=0.7,
                sl_pct=1.8, tp_pct=4.0, leverage=2,
                reason=f"Funding arb LONG: funding={funding:.4f} trend up"
            )

        return None
