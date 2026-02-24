"""
RL Agent — Meta-controller (v12 — Kalıcı Hafıza).
Artık Q-table ve epsilon Railway Volume'a kaydediliyor.
Restart sonrası öğrenme kaldığı yerden devam eder.
"""
import asyncio
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

from persistence import save_rl_model, load_rl_model

logger = logging.getLogger(__name__)

STRATEGIES = ["DCA", "GRID", "SMART"]
RISK_MODES = ["conservative", "normal", "aggressive"]
LEVERAGE_CAPS = [2, 3, 5]
TRADE_ALLOWED = [0, 1]

ACTIONS = [
    (s, r, l, t)
    for t in TRADE_ALLOWED
    for s in range(len(STRATEGIES))
    for r in range(len(RISK_MODES))
    for l in range(len(LEVERAGE_CAPS))
]
N_ACTIONS = len(ACTIONS)


@dataclass
class AgentDecision:
    strategy: str
    risk_mode: str
    leverage_cap: int
    trade_allowed: bool
    confidence: float = 1.0
    reason: str = ""


@dataclass
class Experience:
    state: tuple
    action_idx: int
    reward: float
    next_state: tuple
    done: bool


def discretize_state(regime, atr_pct, funding_rate, winrate_last10, drawdown_pct, daily_loss_pct):
    regime_d  = {"trend": 0, "range": 1, "volatile": 2, "unknown": 1}.get(regime, 1)
    atr_d     = 0 if atr_pct < 0.005 else (1 if atr_pct < 0.015 else (2 if atr_pct < 0.03 else 3))
    funding_d = 0 if abs(funding_rate) < 0.002 else (1 if funding_rate < 0 else 2)
    winrate_d = 0 if winrate_last10 < 0.4 else (1 if winrate_last10 < 0.6 else 2)
    dd_d      = 0 if drawdown_pct < 1 else (1 if drawdown_pct < 3 else 2)
    dl_d      = 0 if daily_loss_pct < 0.5 else (1 if daily_loss_pct < 1.5 else 2)
    return (regime_d, atr_d, funding_d, winrate_d, dd_d, dl_d)


class RLAgent:
    def __init__(self, settings):
        self.s = settings
        self.q_table: dict = {}
        self.memory: deque = deque(maxlen=settings.RL_BUFFER_SIZE)
        self.epsilon = settings.RL_EPSILON
        self.lr = settings.RL_LEARNING_RATE
        self.gamma = settings.RL_DISCOUNT
        self.step_count = 0
        self.episode_count = 0
        self.last_reward = 0.0
        self.total_reward = 0.0
        self._last_state: Optional[tuple] = None
        self._last_action_idx: Optional[int] = None
        self._running = False
        self._data_client = None
        self._risk_engine = None

    def set_dependencies(self, data_client, risk_engine):
        self._data_client = data_client
        self._risk_engine = risk_engine

    async def load_model(self):
        saved = load_rl_model()
        if saved:
            self.q_table = saved.get("q_table", {})
            self.epsilon  = saved.get("epsilon", self.epsilon)
            logger.info(f"🧠 RL model yüklendi: {len(self.q_table)} state | ε={self.epsilon:.4f}")
        else:
            logger.info("🧠 RL model yok, sıfırdan başlanıyor")

    async def save_model(self):
        try:
            save_rl_model(
                q_table=self.q_table,
                epsilon=self.epsilon,
                episode_count=self.episode_count,
                total_reward=self.total_reward,
            )
            logger.debug(f"RL model kaydedildi ({len(self.q_table)} state)")
        except Exception as e:
            logger.warning(f"RL kayıt hatası: {e}")

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * N_ACTIONS
        return self.q_table[state]

    def _build_state(self):
        if not self._data_client or not self._risk_engine:
            return (1, 1, 0, 1, 0, 0)
        ds = self._data_client.state
        rs = self._risk_engine.state
        atr_pct = ds.atr_14 / (ds.mark_price + 1e-9)
        daily_loss_pct = (rs.daily_loss / (rs.current_equity + 1e-9)) * 100
        hist = rs.trade_history[-10:]
        winrate10 = sum(1 for t in hist if t.pnl > 0) / len(hist) if hist else 0.5
        return discretize_state(ds.regime, atr_pct, ds.funding_rate, winrate10,
                                 rs.current_drawdown_pct, daily_loss_pct)

    def decide(self) -> AgentDecision:
        state = self._build_state()
        self._last_state = state
        if random.random() < self.epsilon:
            action_idx = random.randrange(N_ACTIONS)
        else:
            q_vals = self._get_q(state)
            action_idx = max(range(N_ACTIONS), key=lambda i: q_vals[i])
        self._last_action_idx = action_idx
        s_idx, r_idx, l_idx, t = ACTIONS[action_idx]
        return AgentDecision(
            strategy=STRATEGIES[s_idx], risk_mode=RISK_MODES[r_idx],
            leverage_cap=LEVERAGE_CAPS[l_idx], trade_allowed=bool(t),
            confidence=1.0 - self.epsilon,
        )

    def compute_reward(self, net_pnl, drawdown_increment, volatility, overtrading, sharpe_increment):
        return (net_pnl - 2.0 * max(0, drawdown_increment)
                - 0.5 * volatility * abs(net_pnl)
                - 1.0 * (0.5 if overtrading else 0)
                + 0.3 * sharpe_increment)

    async def record_outcome(self, reward: float, done: bool = False):
        if self._last_state is None or self._last_action_idx is None:
            return
        next_state = self._build_state()
        exp = Experience(self._last_state, self._last_action_idx, reward, next_state, done)
        self.memory.append(exp)
        self.step_count += 1
        self.episode_count += 1
        self.last_reward = reward
        self.total_reward += reward
        self._update_q(exp)
        self.epsilon = max(0.05, self.epsilon * (0.998 if reward > 0 else 0.9999))
        if self.step_count % 50 == 0:
            await self.save_model()

    def _update_q(self, exp: Experience):
        q = self._get_q(exp.state)
        next_q = self._get_q(exp.next_state)
        target = exp.reward + (self.gamma * max(next_q) if not exp.done else 0)
        q[exp.action_idx] += self.lr * (target - q[exp.action_idx])

    async def learning_loop(self):
        self._running = True
        logger.info("🧠 RL learning loop başlatıldı (kalıcı hafıza aktif)")
        save_counter = 0
        while self._running:
            await asyncio.sleep(60)
            if len(self.memory) >= self.s.RL_BATCH_SIZE:
                recent    = list(self.memory)[-self.s.RL_BATCH_SIZE // 2:]
                old_batch = random.sample(list(self.memory), min(self.s.RL_BATCH_SIZE // 2, len(self.memory)))
                batch = recent + old_batch
                for exp in batch:
                    self._update_q(exp)
                avg_r = sum(e.reward for e in batch) / len(batch) if batch else 0
                logger.info(f"🧠 RL batch: {len(batch)} exp | ε={self.epsilon:.4f} | "
                            f"states={len(self.q_table)} | avg_r={avg_r:.3f} | ep={self.episode_count}")
            save_counter += 1
            if save_counter >= 10:  # 10 dakikada bir zorunlu kayıt
                save_counter = 0
                await self.save_model()

    def get_status(self) -> dict:
        from persistence import get_storage_info
        try:
            storage = get_storage_info()
        except Exception:
            storage = {}
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "memory_size": len(self.memory),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "last_reward": round(self.last_reward, 4),
            "total_reward": round(self.total_reward, 3),
            "storage": storage,
        }
