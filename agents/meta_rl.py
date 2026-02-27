"""
RL Meta-Controller v15 — Capital allocation öğrenen üst katman.

v14'ten farkı: Artık bireysel trade kararı değil, agent allokasyon
ağırlıklarını öğreniyor. Hangi agent'a ne kadar kapital verilmeli?
Hangi market rejiminde hangi agent daha iyi?

State: (regime, atr_level, funding_sign, hour_of_day, portfolio_dd)
Action: (agent_weight_vector) → 4 agent için ağırlık
Reward: Sharpe-adjusted portfolio return (her 30dk'da hesaplanır)
"""
import asyncio
import logging
import random
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from persistence import save_rl_model, load_rl_model

logger = logging.getLogger(__name__)

AGENT_IDS = ["trend", "meanrev", "scalp", "macro"]
N_AGENTS  = len(AGENT_IDS)

# Ağırlık seviyeleri: 0=düşük(0.10), 1=normal(0.25), 2=yüksek(0.40), 3=tam(0.50)
WEIGHT_LEVELS = [0.10, 0.25, 0.40, 0.50]
N_LEVELS = len(WEIGHT_LEVELS)

# Action space: her agent için 4 seviye → 4^4 = 256 aksiyon
N_ACTIONS = N_LEVELS ** N_AGENTS  # 256


def _discretize_state(regime, atr_pct, funding, dd_pct, hour):
    r = {"trend": 0, "range": 1, "volatile": 2, "unknown": 1}.get(regime, 1)
    a = 0 if atr_pct < 0.005 else (1 if atr_pct < 0.015 else 2)
    f = 0 if abs(funding) < 0.002 else (1 if funding > 0 else 2)
    d = 0 if dd_pct < 1 else (1 if dd_pct < 3 else 2)
    h = hour // 6   # 4 blok: 0-5, 6-11, 12-17, 18-23
    return (r, a, f, d, h)


def _action_to_weights(action_idx: int) -> Dict[str, float]:
    """Action index → her agent için allokasyon ağırlığı."""
    weights = {}
    idx = action_idx
    for agent_id in reversed(AGENT_IDS):
        level = idx % N_LEVELS
        weights[agent_id] = WEIGHT_LEVELS[level]
        idx //= N_LEVELS
    # Normalize et (toplam 1.0 olsun)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


class RLMetaController:
    """
    Capital Brain'in allokasyon ağırlıklarını öğrenen üst-seviye kontrolcü.
    v14 RLAgent ile uyumlu — aynı Q-table ve persistence mekanizması.
    """

    def __init__(self, settings):
        self.s            = settings
        self.q_table: dict = {}
        self.memory       = deque(maxlen=5000)
        self.epsilon      = getattr(settings, "RL_EPSILON", 0.20)
        self.lr           = getattr(settings, "RL_LEARNING_RATE", 0.001)
        self.gamma        = getattr(settings, "RL_DISCOUNT", 0.95)
        self.episode_count = 0
        self.last_reward  = 0.0
        self.total_reward = 0.0

        # Durum takibi
        self._last_state: Optional[tuple]  = None
        self._last_action: Optional[int]   = None
        self._last_portfolio_value: float  = 0.0
        self._last_weights: Optional[dict] = None

        # Bağımlılıklar (sonradan set edilir)
        self._data    = None
        self._risk    = None
        self._brain   = None
        self._running = False

    def set_dependencies(self, data_client, risk_engine, capital_brain):
        self._data  = data_client
        self._risk  = risk_engine
        self._brain = capital_brain

    async def load_model(self):
        saved = load_rl_model()
        if saved:
            self.q_table      = saved.get("q_table", {})
            self.epsilon      = saved.get("epsilon", self.epsilon)
            self.episode_count = saved.get("episode_count", 0)
            self.total_reward = saved.get("total_reward", 0.0)
            logger.info(f"🧠 Meta-RL yüklendi: {len(self.q_table)} state | ε={self.epsilon:.4f} | ep={self.episode_count}")
        else:
            logger.info("🧠 Meta-RL sıfırdan başlıyor")

    async def save_model(self):
        try:
            save_rl_model(
                q_table=self.q_table,
                epsilon=self.epsilon,
                episode_count=self.episode_count,
                total_reward=self.total_reward,
            )
        except Exception as e:
            logger.warning(f"Meta-RL kayıt hatası: {e}")

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * N_ACTIONS
        return self.q_table[state]

    def _build_state(self) -> tuple:
        if not self._data or not self._risk:
            return (1, 1, 0, 0, 0)
        ds = self._data.state
        rs = self._risk.state
        atr_pct  = (ds.atr_14 or 0) / max(ds.mark_price or 1, 1)
        funding  = ds.funding_rate or 0
        dd_pct   = rs.current_drawdown_pct or 0
        hour     = datetime.now().hour
        from datetime import datetime
        return _discretize_state(ds.regime, atr_pct, funding, dd_pct, hour)

    def decide_weights(self) -> Dict[str, float]:
        """Mevcut piyasa durumuna göre agent ağırlıklarını belirle."""
        state = self._build_state()
        self._last_state = state

        if random.random() < self.epsilon:
            action = random.randrange(N_ACTIONS)
        else:
            q_vals = self._get_q(state)
            action = max(range(N_ACTIONS), key=lambda i: q_vals[i])

        self._last_action = action
        weights = _action_to_weights(action)
        self._last_weights = weights

        logger.debug(f"🤖 Meta-RL weights: {' '.join(f'{k}={v:.2f}' for k,v in weights.items())} ε={self.epsilon:.3f}")
        return weights

    async def record_portfolio_outcome(self, new_portfolio_value: float):
        """
        Periyodik olarak çağrılır. Portfolio değişimine göre reward hesapla.
        """
        if self._last_state is None or self._last_action is None:
            self._last_portfolio_value = new_portfolio_value
            return

        if self._last_portfolio_value > 0:
            ret = (new_portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
            # Sharpe benzeri reward: getiri - oynaklık cezası
            reward = ret * 100  # % → float
            # Brain'deki agent skorlarını da dahil et
            if self._brain:
                agent_scores = [a.score for a in self._brain.get_all_agents()]
                avg_score = sum(agent_scores) / len(agent_scores) if agent_scores else 1.0
                reward *= avg_score  # iyi ajanlarla açılan pozisyonlar daha yüksek reward

            reward = max(-5.0, min(5.0, reward))  # clip

            next_state = self._build_state()
            self._update_q(self._last_state, self._last_action, reward, next_state)

            self.last_reward   = reward
            self.total_reward += reward
            self.episode_count += 1

            # Epsilon decay
            self.epsilon = max(0.05, self.epsilon * (0.995 if reward > 0 else 0.9999))

            logger.info(
                f"🧠 Meta-RL reward={reward:+.3f} (ret={ret*100:+.3f}%) "
                f"ε={self.epsilon:.4f} ep={self.episode_count}"
            )

        self._last_portfolio_value = new_portfolio_value

        if self.episode_count % 20 == 0:
            await self.save_model()

    def _update_q(self, state, action, reward, next_state):
        q      = self._get_q(state)
        next_q = self._get_q(next_state)
        target = reward + self.gamma * max(next_q)
        q[action] += self.lr * (target - q[action])

    async def learning_loop(self):
        """Her 30dk'da portföy değerini ölç, reward hesapla, ağırlıkları güncelle."""
        self._running = True
        logger.info("🧠 Meta-RL learning loop başlatıldı")
        await asyncio.sleep(60)

        while self._running:
            try:
                # Portfolio değeri
                if self._data:
                    bal = await self._data.get_balance()
                    portfolio_val = float(bal.get("total", 0) or 0)
                    await self.record_portfolio_outcome(portfolio_val)

                # Ağırlıkları Capital Brain'e uygula
                if self._brain:
                    weights = self.decide_weights()
                    for agent_id, weight in weights.items():
                        slot = self._brain.get_agent(agent_id)
                        if slot:
                            slot.current_allocation_pct = weight

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Meta-RL loop hatası: {e}")

            await asyncio.sleep(1800)  # 30dk

    def get_status(self) -> dict:
        return {
            "epsilon":       round(self.epsilon, 4),
            "q_table_size":  len(self.q_table),
            "episode_count": self.episode_count,
            "last_reward":   round(self.last_reward, 4),
            "total_reward":  round(self.total_reward, 3),
            "last_weights":  {k: round(v, 3) for k, v in (self._last_weights or {}).items()},
        }


# Backward compat — v14 RLAgent arayüzü
from rl_agent.agent import RLAgent as _LegacyRLAgent

class RLAgent(_LegacyRLAgent):
    """v14 ile uyumluluk — Manager hâlâ bu sınıfı kullanıyor."""
    pass
