"""
PersistenceStore — Railway'de kalıcı hafıza.

Öncelik sırası:
  1. PERSIST_DIR env var (Railway Volume mount path, örn: /data)
  2. /data  (Railway volume default)
  3. /tmp   (fallback — restart'ta silinir, uyarı verir)

Saklanan veriler:
  - Trade geçmişi (SQLite)
  - RL model Q-table + epsilon (pickle → SQLite BLOB)
  - Sistem logları (SQLite, son 5000 kayıt)
  - Bot ayarları snapshot (JSON)
  - Günlük PNL istatistikleri
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, date
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Dizin tespiti ──────────────────────────────────────────────────────────────

def _detect_persist_dir() -> str:
    """
    Kalıcı dizini tespit et.
    Railway Volume: Settings → Volume → Mount Path olarak /data ayarlanmalı.
    """
    candidates = [
        os.environ.get("PERSIST_DIR", ""),
        "/data",          # Railway Volume default mount
        "/mnt/data",      # alternatif mount
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            # Yazılabilirlik testi
            test_file = os.path.join(p, ".write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
                logger.info(f"✅ Kalıcı dizin: {p} (Railway Volume)")
                return p
            except Exception:
                continue

    # Fallback: /tmp (restart'ta silinir)
    fallback = "/tmp/trademaxpro_data"
    os.makedirs(fallback, exist_ok=True)
    logger.warning(
        "⚠️  Kalıcı dizin bulunamadı! /tmp kullanılıyor. "
        "Railway'de Volume ekle: Settings → Volumes → Mount Path: /data"
    )
    return fallback


PERSIST_DIR = _detect_persist_dir()
DB_PATH = os.path.join(PERSIST_DIR, "trademaxpro.db")
RL_MODEL_PATH = os.path.join(PERSIST_DIR, "rl_agent.pkl")
LOGS_PATH = os.path.join(PERSIST_DIR, "bot_logs.jsonl")

MAX_LOG_ROWS = 5000
MAX_TRADE_ROWS = 10000


# ── SQLite bağlantısı (thread-safe) ───────────────────────────────────────────

_db_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # yazma çakışmasını önler
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Tabloları oluştur (yoksa)."""
    with _db_lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    side        TEXT,
                    strategy    TEXT,
                    pnl         REAL,
                    slippage_pct REAL DEFAULT 0,
                    symbol      TEXT DEFAULT '',
                    extra_json  TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    stat_date   TEXT PRIMARY KEY,
                    total_pnl   REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    win_count   INTEGER DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    peak_equity REAL DEFAULT 0,
                    updated_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS rl_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_at    TEXT NOT NULL,
                    epsilon     REAL,
                    episode_count INTEGER,
                    total_reward  REAL,
                    q_table_size  INTEGER,
                    model_blob  BLOB
                );

                CREATE TABLE IF NOT EXISTS system_logs (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT NOT NULL,
                    level   TEXT,
                    name    TEXT,
                    message TEXT
                );

                CREATE TABLE IF NOT EXISTS bot_settings (
                    key     TEXT PRIMARY KEY,
                    value   TEXT,
                    updated_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_trades_ts ON trade_history(ts);
                CREATE INDEX IF NOT EXISTS idx_logs_ts   ON system_logs(ts);
            """)
            conn.commit()
            logger.info(f"🗄️  Veritabanı hazır: {DB_PATH}")
        finally:
            conn.close()


# ── Trade geçmişi ──────────────────────────────────────────────────────────────

def save_trade(pnl: float, side: str, strategy: str,
               slippage_pct: float = 0.0, symbol: str = "",
               extra: dict = None):
    """Trade kaydını SQLite'e yaz."""
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO trade_history (ts, side, strategy, pnl, slippage_pct, symbol, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    side, strategy,
                    round(pnl, 6),
                    round(slippage_pct, 6),
                    symbol,
                    json.dumps(extra or {}),
                ),
            )
            # Eski kayıtları temizle (max satır)
            conn.execute(
                f"""DELETE FROM trade_history WHERE id NOT IN (
                       SELECT id FROM trade_history ORDER BY id DESC LIMIT {MAX_TRADE_ROWS}
                   )"""
            )
            conn.commit()
        finally:
            conn.close()


def load_trades(limit: int = 1000) -> List[dict]:
    """Son N trade kaydını döner."""
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_trade_stats() -> dict:
    """Tüm zamanlı trade istatistikleri."""
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl) as avg_pnl,
                    MIN(pnl) as worst_trade,
                    MAX(pnl) as best_trade
                FROM trade_history
            """).fetchone()
            if row and row["total"]:
                total = row["total"]
                wins = row["wins"] or 0
                return {
                    "trade_count":  total,
                    "total_pnl":    round(row["total_pnl"] or 0, 4),
                    "winrate":      round(wins / total, 3),
                    "avg_pnl":      round(row["avg_pnl"] or 0, 4),
                    "worst_trade":  round(row["worst_trade"] or 0, 4),
                    "best_trade":   round(row["best_trade"] or 0, 4),
                }
            return {"trade_count": 0, "total_pnl": 0, "winrate": 0}
        finally:
            conn.close()


# ── Günlük istatistikler ───────────────────────────────────────────────────────

def update_daily_stats(pnl: float, is_win: bool, drawdown_pct: float, peak_equity: float):
    today = date.today().isoformat()
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT INTO daily_stats (stat_date, total_pnl, trade_count, win_count,
                                         max_drawdown_pct, peak_equity, updated_at)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(stat_date) DO UPDATE SET
                    total_pnl        = total_pnl + excluded.total_pnl,
                    trade_count      = trade_count + 1,
                    win_count        = win_count + excluded.win_count,
                    max_drawdown_pct = MAX(max_drawdown_pct, excluded.max_drawdown_pct),
                    peak_equity      = MAX(peak_equity, excluded.peak_equity),
                    updated_at       = excluded.updated_at
            """, (today, pnl, 1 if is_win else 0, drawdown_pct, peak_equity,
                  datetime.utcnow().isoformat()))
            conn.commit()
        finally:
            conn.close()


def load_daily_stats(days: int = 30) -> List[dict]:
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM daily_stats ORDER BY stat_date DESC LIMIT ?", (days,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ── RL Model ───────────────────────────────────────────────────────────────────

def save_rl_model(q_table: dict, epsilon: float, episode_count: int,
                  total_reward: float):
    """RL modelini hem dosyaya hem SQLite'e kaydet (çift yedek)."""
    # 1. Dosya yedek (hızlı okuma için)
    try:
        os.makedirs(os.path.dirname(RL_MODEL_PATH) if os.path.dirname(RL_MODEL_PATH) else ".", exist_ok=True)
        with open(RL_MODEL_PATH, "wb") as f:
            pickle.dump({"q_table": q_table, "epsilon": epsilon}, f)
    except Exception as e:
        logger.warning(f"RL dosya kaydı hatası: {e}")

    # 2. SQLite yedek (kalıcı)
    try:
        blob = pickle.dumps({"q_table": q_table, "epsilon": epsilon})
        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute("""
                    INSERT INTO rl_snapshots
                        (saved_at, epsilon, episode_count, total_reward, q_table_size, model_blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    round(epsilon, 6),
                    episode_count,
                    round(total_reward, 4),
                    len(q_table),
                    blob,
                ))
                # Son 50 snapshot tut
                conn.execute("""
                    DELETE FROM rl_snapshots WHERE id NOT IN (
                        SELECT id FROM rl_snapshots ORDER BY id DESC LIMIT 50
                    )
                """)
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        logger.warning(f"RL SQLite kaydı hatası: {e}")


def load_rl_model() -> Optional[dict]:
    """
    RL modelini yükle.
    Önce dosyadan dene, yoksa SQLite'teki son snapshot'ı al.
    """
    # 1. Dosyadan dene
    if os.path.exists(RL_MODEL_PATH):
        try:
            with open(RL_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            logger.info(f"✅ RL model dosyadan yüklendi: {RL_MODEL_PATH} ({len(data.get('q_table', {}))} state)")
            return data
        except Exception as e:
            logger.warning(f"RL dosya yüklenemedi: {e}, SQLite'e bakılıyor...")

    # 2. SQLite'ten al
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT model_blob, saved_at FROM rl_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row["model_blob"]:
                data = pickle.loads(row["model_blob"])
                logger.info(f"✅ RL model SQLite'ten yüklendi (kayıt: {row['saved_at']})")
                # Dosyayı da güncelle
                try:
                    with open(RL_MODEL_PATH, "wb") as f:
                        pickle.dump(data, f)
                except Exception:
                    pass
                return data
        except Exception as e:
            logger.warning(f"RL SQLite yüklenemedi: {e}")
        finally:
            conn.close()

    logger.info("RL model bulunamadı, sıfırdan başlanıyor")
    return None


# ── Sistem logları ─────────────────────────────────────────────────────────────

def save_log_entry(ts: str, level: str, name: str, message: str):
    """Log kaydını SQLite'e yaz (non-blocking — thread safe)."""
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    "INSERT INTO system_logs (ts, level, name, message) VALUES (?, ?, ?, ?)",
                    (ts, level, name, message[:1000]),
                )
                # Max satır koru
                conn.execute(f"""
                    DELETE FROM system_logs WHERE id NOT IN (
                        SELECT id FROM system_logs ORDER BY id DESC LIMIT {MAX_LOG_ROWS}
                    )
                """)
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass  # log sistemi asla çökmemeli


def load_logs(limit: int = 200, level_filter: str = None) -> List[dict]:
    """Son logları döner (en yeni önce)."""
    with _db_lock:
        conn = _get_conn()
        try:
            if level_filter:
                rows = conn.execute(
                    "SELECT ts as time, level, name, message FROM system_logs "
                    "WHERE level = ? ORDER BY id DESC LIMIT ?",
                    (level_filter.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts as time, level, name, message FROM system_logs "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ── Bot ayarları ───────────────────────────────────────────────────────────────

def save_setting(key: str, value: Any):
    """Ayarı SQLite'e kaydet."""
    with _db_lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT INTO bot_settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, json.dumps(value), datetime.utcnow().isoformat()))
            conn.commit()
        finally:
            conn.close()


def load_setting(key: str, default=None) -> Any:
    with _db_lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM bot_settings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
            return default
        finally:
            conn.close()


def load_all_settings() -> Dict[str, Any]:
    with _db_lock:
        conn = _get_conn()
        try:
            rows = conn.execute("SELECT key, value FROM bot_settings").fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}
        finally:
            conn.close()


# ── Storage istatistikleri ─────────────────────────────────────────────────────

def get_storage_info() -> dict:
    """Depolama bilgisi — dashboard için."""
    info = {
        "persist_dir": PERSIST_DIR,
        "is_persistent": PERSIST_DIR not in ("/tmp/trademaxpro_data", "/tmp"),
        "db_path": DB_PATH,
        "rl_model_path": RL_MODEL_PATH,
    }
    try:
        if os.path.exists(DB_PATH):
            info["db_size_kb"] = round(os.path.getsize(DB_PATH) / 1024, 1)
        if os.path.exists(RL_MODEL_PATH):
            info["rl_model_size_kb"] = round(os.path.getsize(RL_MODEL_PATH) / 1024, 1)
        # Disk kullanımı
        st = os.statvfs(PERSIST_DIR)
        info["disk_free_mb"] = round(st.f_bavail * st.f_frsize / 1024 / 1024, 1)
        info["disk_total_mb"] = round(st.f_blocks * st.f_frsize / 1024 / 1024, 1)
    except Exception:
        pass

    with _db_lock:
        conn = _get_conn()
        try:
            info["trade_count"]  = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0]
            info["log_count"]    = conn.execute("SELECT COUNT(*) FROM system_logs").fetchone()[0]
            info["rl_snapshots"] = conn.execute("SELECT COUNT(*) FROM rl_snapshots").fetchone()[0]
            last_save = conn.execute(
                "SELECT saved_at FROM rl_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            info["rl_last_saved"] = last_save[0] if last_save else None
        except Exception:
            pass
        finally:
            conn.close()

    return info
