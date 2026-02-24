# Trademaxpro v12 — Kalıcı Hafıza Kurulum Rehberi

## ✅ Ne Değişti?

### Önceki Sorun
- Sayfa yenilenince tüm trade geçmişi gidiyordu
- Bot restart olunca RL modeli sıfırlanıyordu  
- Loglar /tmp'de tutuluyordu (restart'ta siliniyordu)

### v12 Çözümü
- **SQLite veritabanı** → trade geçmişi, loglar, günlük istatistikler kalıcı
- **RL Q-table** → Railway Volume'a kaydediliyor, restart sonrası kaldığı yerden devam
- **Dashboard** → Sayfa yenilenince geçmiş `/status/history` API'sinden otomatik yükleniyor

---

## 🚀 Railway'de Kurulum

### 1. Volume Ekle (EN ÖNEMLİ ADIM)

Railway panelinde:
```
Servisin sayfası → Settings → Volumes → Add Volume
Mount Path: /data
Size: 1 GB (yeterli)
```

Bu yapılmazsa bot /tmp kullanır ve restart'ta veriler silinir.  
Dashboard'da log panelinin yanında **kırmızı TMP** badge görünür — Volume eklenince **yeşil VOLUME** olur.

### 2. Environment Variables

```env
PERSIST_DIR=/data          # Railway Volume mount path
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=false      # Production için
PORT=8000
```

### 3. Deploy

```bash
# Projeyi Railway'e push et
git add .
git commit -m "v12: kalici hafiza (SQLite + Railway Volume)"
git push
```

---

## 📊 Yeni API Endpoint'leri

| Endpoint | Açıklama |
|----------|----------|
| `GET /status/history?limit=500` | Tüm zamanlı trade geçmişi (SQLite) |
| `GET /status/storage` | Volume durumu, DB boyutu |
| `GET /status/logs?limit=200&level=ERROR` | Kalıcı log geçmişi (level filtreli) |

---

## 🗄️ Veritabanı Yapısı

`/data/trademaxpro.db` (SQLite):

- `trade_history` — her trade kaydı (pnl, side, strategy, symbol, timestamp)
- `daily_stats` — günlük PNL özeti
- `rl_snapshots` — RL model snapshot'ları (son 50 kayıt)
- `system_logs` — sistem logları (son 5000 kayıt)
- `bot_settings` — dinamik ayarlar

`/data/rl_agent.pkl` — RL Q-table hızlı erişim dosyası

---

## 🔍 Sorun Giderme

**"TMP" badge görünüyorsa:**
→ Railway'de Volume eklenmemiş. Settings → Volumes → Add (`/data`)

**Trade geçmişi görünmüyorsa:**
→ `/status/history` endpoint'ini test et
→ Bot yeni kurulduysa trade olmadığı için boş normal

**RL model sıfırlanıyorsa:**
→ Volume mount edildi mi? `/status/storage` endpoint'ini kontrol et
→ `is_persistent: true` olmalı
