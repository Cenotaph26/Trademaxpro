"""
RL Agent — Meta-controller.
"Hangi strateji + parametre + risk modu" seçer.
Emir atmaz → Strategy Engine'e karar verir.

Algoritma: Tabular Q-Learning (başlangıç için basit, production'da DQN'e geç).
State space discretize edilmiş → yönetilebilir Q tablosu.
"""
import asyncio
import logging
import pickle
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── Action Space ─────────────────────────────────────────────────────────────
STRATEGIES = ["DCA", "GRID", "SMART"]
RISK_MODES = ["conservative", "normal", "aggressive"]
LEVERAGE_CAPS = [2, 3, 5]
TRADE_ALLOWED = [0, 1]

# Flatten action: (strategy_idx, risk_mode_idx, leverage_idx, trade_allowed)
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
    strategy: str           # DCA / GRID / SMART
    risk_mode: str          # conservative / normal / aggressive
    leverage_cap: int       # 2 / 3 / 5
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


def discretize_state(
    regime: str,
    atr_pct: float,
    funding_rate: float,
    winrate_last10: float,
    drawdown_pct: float,
    daily_loss_pct: float,
) -> tuple:
    """State'i discretize et → Q tablosu için hashable key."""
    regime_d = {"trend": 0, "range": 1, "volatile": 2, "unknown": 1}.get(regime, 1)

    if atr_pct < 0.005:
        atr_d = 0
    elif atr_pct < 0.015:
        atr_d = 1
    elif atr_pct < 0.03:
        atr_d = 2
    else:
        atr_d = 3

    funding_d = 0 if abs(funding_rate) < 0.002 else (1 if funding_rate < 0 else 2)

    winrate_d = 0 if winrate_last10 < 0.4 else (1 if winrate_last10 < 0.6 else 2)

    dd_d = 0 if drawdown_pct < 1 else (1 if drawdown_pct < 3 else 2)

    dl_d = 0 if daily_loss_pct < 0.5 else (1 if daily_loss_pct < 1.5 else 2)

    return (regime_d, atr_d, funding_d, winrate_d, dd_d, dl_d)


class RLAgent:
    def __init__(self, settings):
        self.s = settings
        self.q_table: dict = {}   # state -> [Q values per action]
        self.memory: deque = deque(maxlen=settings.RL_BUFFER_SIZE)
        self.epsilon = settings.RL_EPSILON
        self.lr = settings.RL_LEARNING_RATE
        self.gamma = settings.RL_DISCOUNT
        self.step_count = 0
        self._last_state: Optional[tuple] = None
        self._last_action_idx: Optional[int] = None
        self._running = False

        # Bağımlılıklar (inject edilecek)
        self._data_client = None
        self._risk_engine = None

    def set_dependencies(self, data_client, risk_engine):
        self._data_client = data_client
        self._risk_engine = risk_engine

    async def load_model(self):
        path = self.s.RL_MODEL_PATH
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    saved = pickle.load(f)
                    self.q_table = saved.get("q_table", {})
                    self.epsilon = saved.get("epsilon", self.epsilon)
                logger.info(f"RL model yüklendi: {path} ({len(self.q_table)} state)")
            except Exception as e:
                logger.warning(f"Model yüklenemedi: {e}, sıfırdan başlanıyor")
        else:
            logger.info("RL model bulunamadı, sıfırdan başlanıyor")

    async def save_model(self):
        os.makedirs(os.path.dirname(self.s.RL_MODEL_PATH), exist_ok=True)
        with open(self.s.RL_MODEL_PATH, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        logger.debug("RL model kaydedildi")

    def _get_q(self, state: tuple) -> list:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * N_ACTIONS
        return self.q_table[state]

    def _build_state(self) -> tuple:
        if not self._data_client or not self._risk_engine:
            return (1, 1, 0, 1, 0, 0)  # default range state

        ds = self._data_client.state
        rs = self._risk_engine.state
        stats = self._risk_engine.get_stats()

        atr_pct = ds.atr_14 / (ds.mark_price + 1e-9)
        daily_loss_pct = (rs.daily_loss / (rs.current_equity + 1e-9)) * 100

        # winrate son 10 işlem
        hist = rs.trade_history[-10:]
        if hist:
            winrate10 = sum(1 for t in hist if t.pnl > 0) / len(hist)
        else:
            winrate10 = 0.5

        return discretize_state(
            regime=ds.regime,
            atr_pct=atr_pct,
            funding_rate=ds.funding_rate,
            winrate_last10=winrate10,
            drawdown_pct=rs.current_drawdown_pct,
            daily_loss_pct=daily_loss_pct,
        )

    def decide(self) -> AgentDecision:
        """ε-greedy politikası ile karar ver."""
        state = self._build_state()
        self._last_state = state

        if random.random() < self.epsilon:
            # Keşif (exploration)
            action_idx = random.randrange(N_ACTIONS)
        else:
            # Sömürü (exploitation)
            q_vals = self._get_q(state)
            action_idx = max(range(N_ACTIONS), key=lambda i: q_vals[i])

        self._last_action_idx = action_idx
        s_idx, r_idx, l_idx, t = ACTIONS[action_idx]

        return AgentDecision(
            strategy=STRATEGIES[s_idx],
            risk_mode=RISK_MODES[r_idx],
            leverage_cap=LEVERAGE_CAPS[l_idx],
            trade_allowed=bool(t),
            confidence=1.0 - self.epsilon,
        )

    def compute_reward(
        self,
        net_pnl: float,
        drawdown_increment: float,
        volatility: float,
        overtrading: bool,
        sharpe_increment: float,
    ) -> float:
        """Risk-adjusted reward."""
        reward = (
            net_pnl
            - 2.0 * max(0, drawdown_increment)
            - 0.5 * volatility * abs(net_pnl)
            - 1.0 * (0.5 if overtrading else 0)
            + 0.3 * sharpe_increment
        )
        return reward

    async def record_outcome(self, reward: float, done: bool = False):
        """Trade tamamlandı, sonucu kaydet ve Q güncelle."""
        if self._last_state is None or self._last_action_idx is None:
            return

        next_state = self._build_state()
        exp = Experience(
            state=self._last_state,
            action_idx=self._last_action_idx,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.memory.append(exp)
        self.step_count += 1

        # Online Q güncelleme (her step)
        self._update_q(exp)

        # Epsilon decay
        self.epsilon = max(0.05, self.epsilon * 0.9995)

        # Periyodik kayıt
        if self.step_count % 50 == 0:
            await self.save_model()

    def _update_q(self, exp: Experience):
        """Tek adım Q güncelleme (tabular)."""
        q = self._get_q(exp.state)
        next_q = self._get_q(exp.next_state)

        target = exp.reward
        if not exp.done:
            target += self.gamma * max(next_q)

        q[exp.action_idx] += self.lr * (target - q[exp.action_idx])

    async def learning_loop(self):
        """Arka planda periyodik batch güncelleme (replay buffer'dan)."""
        self._running = True
        logger.info("RL learning loop başlatıldı")
        while self._running:
            await asyncio.sleep(60)
            if len(self.memory) >= self.s.RL_BATCH_SIZE:
                batch = random.sample(self.memory, self.s.RL_BATCH_SIZE)
                for exp in batch:
                    self._update_q(exp)
                logger.debug(f"RL batch update: {self.s.RL_BATCH_SIZE} exp | ε={self.epsilon:.4f} | states={len(self.q_table)}")

    def get_status(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "memory_size": len(self.memory),
            "step_count": self.step_count,
        }
