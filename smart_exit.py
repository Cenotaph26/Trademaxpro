"""
Smart Exit Engine — Volatiliteye göre akıllı pozisyon kapatma.

Çıkış koşulları:
1. ATR Spike: Volatilite aniden 2x artarsa → koruyucu çıkış
2. Trend Reversal: EMA9 < EMA21 (LONG'da) → trend dönüşü
3. Profit Lock: %profit > threshold'da trailing stop sıkıştır
4. Time Decay: Pozisyon çok uzun süredir açık ve karda → kapat
5. Partial TP: Hedefin %50'sine ulaşınca yarı pozisyonu kapat
6. Volatility Squeeze: BB daralıyor + pozisyon zararda → çıkış
"""
import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _ema(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    k = 2 / (period + 1)
    v = sum(closes[:period]) / period
    for c in closes[period:]:
        v = c * k + v * (1 - k)
    return v


def _atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    trs = [max(c["high"] - c["low"],
               abs(c["high"] - candles[i-1]["close"]),
               abs(c["low"]  - candles[i-1]["close"]))
           for i, c in enumerate(candles) if i > 0]
    return sum(trs[-period:]) / period if trs else 0.0


def _bollinger_width(closes, period=20):
    if len(closes) < period:
        return 0.0
    w = closes[-period:]
    mid = sum(w) / period
    std = math.sqrt(sum((c - mid)**2 for c in w) / period)
    return (2 * std) / mid if mid > 0 else 0.0


class SmartExitEngine:
    """
    Her açık pozisyonu izler, çıkış koşulu oluşunca kapatır.
    strategy_manager.close_position() çağırır.
    """

    def __init__(self, data_client, strategy_manager, settings):
        self.data     = data_client
        self.strategy = strategy_manager
        self.s        = settings
        self._running = False
        self._partial_done: set = set()  # partial TP yapılmış pozisyonlar

        # Konfigürasyon
        self.CHECK_INTERVAL     = 60     # saniye (30→60: çok hızlı tetiklenmesin)
        self.ATR_SPIKE_MULT     = 3.0    # ATR bu kadar artarsa spike (2.2→3.0: false positive azalt)
        self.PROFIT_LOCK_PCT    = 1.5    # % karda trailing stop sıkıştır (kaldıraçlı)
        self.PROFIT_LOCK_TRAIL  = 0.5    # % trailing mesafesi
        self.PARTIAL_TP_PCT     = 0.8    # hedefin %80'ine ulaşınca partial
        self.MAX_HOLD_HOURS     = 48     # max açık kalma süresi (saat)
        self.TREND_REVERSAL_EMA = True   # EMA trend dönüşünde kapat
        self.BB_SQUEEZE_EXIT    = True   # BB daralmasında zararlı pozisyonu kapat
        self.MIN_HOLD_MINUTES   = 10     # Pozisyon en az bu kadar açık kalmadan kapanmaz

        # Pozisyon açılış zamanı takibi
        self._open_since: dict = {}  # symbol → datetime

    async def start(self):
        self._running = True
        logger.info("🛡️ Smart Exit Engine başlatıldı")
        await asyncio.sleep(15)  # sistem yüklensin
        while self._running:
            try:
                await self._check_all_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Smart Exit hata: {e}", exc_info=True)
            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_all_positions(self):
        """Tüm açık pozisyonları tek tek değerlendir."""
        try:
            raw = await self.data.exchange.fetch_positions()
        except Exception as e:
            logger.warning(f"Smart Exit: pozisyon çekme hatası: {e}")
            return

        # Artık açık olmayan pozisyonları _open_since'den temizle
        active_syms = set()
        for p in raw:
            c = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
            if abs(c) > 1e-9:
                sym = p.get("symbol", "").replace("/", "").replace(":USDT", "")
                active_syms.add(sym)
        def _base(sym):
            return sym.replace("/", "").replace(":USDT", "").upper().removesuffix("USDT")
        active_bases = {_base(s) for s in active_syms}
        stale = [k for k in list(self._open_since.keys()) if _base(k) not in active_bases]
        for k in stale:
            self._open_since.pop(k, None)
            self._partial_done.discard(f"{k}_LONG_partial")
            self._partial_done.discard(f"{k}_SHORT_partial")

        ds = self.data.state
        candles_1h = list(ds.klines_1h)
        if len(candles_1h) < 30:
            return

        closes = [c["close"] for c in candles_1h]
        current_atr  = _atr(candles_1h, 14)
        avg_atr      = _atr(candles_1h[-30:], 14)  # son 30 mumluk ATR
        atr_spike    = current_atr > avg_atr * self.ATR_SPIKE_MULT if avg_atr > 0 else False
        ema9         = _ema(closes, 9)
        ema21        = _ema(closes, 21)
        bb_width     = _bollinger_width(closes)
        current_price = ds.mark_price

        for p in raw:
            contracts = float(p.get("contracts") or p.get("info", {}).get("positionAmt") or 0)
            if abs(contracts) < 1e-9:
                continue

            sym       = (p.get("symbol") or "").replace("/", "").replace(":USDT", "")
            entry     = float(p.get("entryPrice") or 0)
            mark      = float(p.get("markPrice") or 0) or current_price
            upnl      = float(p.get("unrealizedPnl") or 0)
            lev       = int(float(p.get("leverage") or 1))
            side      = "LONG" if contracts > 0 else "SHORT"
            notional  = abs(float(p.get("notional") or contracts * mark))

            # Açılış zamanı takibi
            if sym not in self._open_since:
                self._open_since[sym] = datetime.now(timezone.utc)
            hold_hours = (datetime.now(timezone.utc) - self._open_since[sym]).total_seconds() / 3600
            hold_minutes = hold_hours * 60

            # ── Minimum bekleme süresi — yeni pozisyonlar erken kapanmasın ──
            # Spread ve anlık fiyat dalgalanmaları nedeniyle pozisyon açılır açılmaz
            # zararda görünebilir. MIN_HOLD_MINUTES geçmeden Smart Exit tetiklenmez.
            if hold_minutes < self.MIN_HOLD_MINUTES:
                logger.debug(f"⏳ Smart Exit bekleniyor [{sym}]: {hold_minutes:.1f}dk < min {self.MIN_HOLD_MINUTES}dk")
                continue

            # PnL % (kaldıraçsız gerçek %)
            pnl_pct = 0.0
            if entry > 0 and mark > 0:
                pnl_pct = ((mark - entry) / entry * 100) if side == "LONG" else ((entry - mark) / entry * 100)

            reason = None

            # ── Koşul 1: ATR Spike — volatilite patlaması ──────────
            if atr_spike and pnl_pct < -1.0:
                reason = f"ATR spike ({current_atr:.2f} > {avg_atr:.2f}x{self.ATR_SPIKE_MULT}) + zararda"

            # ── Koşul 2: Trend dönüşü ──────────────────────────────
            # pnl_pct eşiği -0.3'ten -1.5'e yükseltildi:
            # Yeni pozisyon spread nedeniyle hemen -0.1~0.5% zararda açılır,
            # bu yüzden -0.3% eşiği çok düşük ve pozisyon 30-60sn içinde kapanıyordu.
            elif self.TREND_REVERSAL_EMA and pnl_pct < -1.5:
                if side == "LONG" and ema9 < ema21 * 0.999:
                    reason = f"Trend dönüşü: EMA9({ema9:.0f}) < EMA21({ema21:.0f}) LONG zararda"
                elif side == "SHORT" and ema9 > ema21 * 1.001:
                    reason = f"Trend dönüşü: EMA9({ema9:.0f}) > EMA21({ema21:.0f}) SHORT zararda"

            # ── Koşul 3: Maksimum holding süresi ──────────────────
            elif hold_hours > self.MAX_HOLD_HOURS and pnl_pct > 0:
                reason = f"Max hold ({hold_hours:.1f}s > {self.MAX_HOLD_HOURS}s) — karda kapat"
            elif hold_hours > self.MAX_HOLD_HOURS * 1.5:
                reason = f"Max hold x1.5 ({hold_hours:.1f}s) — zorunlu kapat"

            # ── Koşul 4: BB Sıkışma + zararda ─────────────────────
            elif self.BB_SQUEEZE_EXIT and bb_width < 0.015 and pnl_pct < -2.0:
                reason = f"BB sıkışma ({bb_width:.3f}) + zararda ({pnl_pct:.2f}%)"

            # ── Koşul 5: Profit Lock ───────────────────────────────
            # Yeterince karda → partial TP (sadece bir kez)
            key = f"{sym}_{side}_partial"
            if (pnl_pct * lev > self.PROFIT_LOCK_PCT * 0.8 and
                    key not in self._partial_done and
                    notional > 0 and
                    reason is None):
                # Yarı pozisyonu kapat
                try:
                    partial_qty = abs(contracts) * 0.5
                    from execution.executor import OrderExecutor, _fmt_symbol
                    if hasattr(self.strategy, "executor") and self.strategy.executor:
                        close_side = "SELL" if side == "LONG" else "BUY"
                        from execution.executor import OrderRequest
                        r = await self.strategy.executor.place_order(OrderRequest(
                            symbol=sym, side=close_side, order_type="MARKET",
                            quantity=partial_qty, reduce_only=True,
                            strategy_tag="smart_exit_partial"
                        ))
                        if r:
                            self._partial_done.add(key)
                            logger.info(f"💰 Partial TP: {sym} {side} {partial_qty:.4f} @ kâr={pnl_pct:.2f}%")
                except Exception as e:
                    es = str(e)
                    if "-4120" in es or "not supported for this endpoint" in es:
                        logger.warning(f"⚠ Partial TP: {sym} testnet'te TAKE_PROFIT_MARKET desteklenmiyor, atlandı")
                    elif "-2022" in es or "ReduceOnly" in es:
                        logger.warning(f"⚠ Partial TP: {sym} zaten kapalı")
                    else:
                        logger.warning(f"Partial TP hatası [{sym}]: {e}")

            # ── Tam kapatma ────────────────────────────────────────
            if reason:
                logger.warning(f"🛡️ Smart Exit [{sym} {side}]: {reason}")
                try:
                    result = await self.strategy.close_position(sym, side)
                    if result and result.get("ok"):
                        logger.info(f"✅ Smart Exit kapatıldı: {sym} {side} PnL={upnl:+.2f}")
                        self._open_since.pop(sym, None)
                        self._partial_done.discard(f"{sym}_{side}_partial")
                    elif result and result.get("reason") in ("zaten_kapali", "qty=0"):
                        logger.info(f"ℹ️ Smart Exit: {sym} zaten kapalıydı")
                        self._open_since.pop(sym, None)
                    elif result and result.get("qty", 1) == 0:
                        logger.info(f"ℹ️ Smart Exit: {sym} miktar=0, kapalı")
                        self._open_since.pop(sym, None)
                    elif result and not result.get("ok"):
                        reason_txt = result.get("reason", "")
                        if "zaten" in reason_txt or "None" in reason_txt or "gönderilemedi" in reason_txt:
                            logger.info(f"ℹ️ Smart Exit: {sym} zaten kapalı (sinyal/SL tarafından kapanmış)")
                            self._open_since.pop(sym, None)
                        else:
                            logger.warning(f"⚠ Smart Exit kapama başarısız: {sym} — {reason_txt or result}")
                    else:
                        logger.warning(f"⚠ Smart Exit kapama: {sym} — {result}")
                except Exception as e:
                    es = str(e)
                    if "-2022" in es or "ReduceOnly" in es:
                        logger.warning(f"⚠ Smart Exit: {sym} zaten kapalı (ReduceOnly)")
                    else:
                        logger.error(f"Smart Exit kapat hatası [{sym}]: {e}")

    def get_status(self) -> dict:
        return {
            "running":       self._running,
            "tracked_count": len(self._open_since),
            "partial_done":  len(self._partial_done),
            "config": {
                "atr_spike_mult":    self.ATR_SPIKE_MULT,
                "profit_lock_pct":   self.PROFIT_LOCK_PCT,
                "max_hold_hours":    self.MAX_HOLD_HOURS,
            }
        }
