"""
agent_state_v3_final.py

V3 Final 版本：
---------------
- 引入 Episode 记录
- 每个 step 使用 immutable AgentState
- Episode 记录每个 step 的 snapshot
- 演示完整轨迹
"""

from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

@dataclass
class EpisodeStep:
    state: "AgentState"

@dataclass
class Episode:
    steps: List[EpisodeStep] = field(default_factory=list)

    def snapshot(self) -> "Episode":
        return deepcopy(self)

    def add_step(self, state: "AgentState") -> "Episode":
        ep = self.snapshot()
        ep.steps.append(EpisodeStep(state))
        return ep

    def show(self):
        print("\n===== Episode Trajectory =====")
        for s in self.steps:
            st = s.state
            print(f"Step {st.step} | Reward {st.reward} | Action {st.last_action} | Observation {st.last_observation}")
        print("="*50)

@dataclass
class AgentState:
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: list = field(default_factory=list)
    last_action: str = None
    last_observation: str = None

    def snapshot(self) -> "AgentState":
        return deepcopy(self)

    def add_memory(self, entry: str) -> "AgentState":
        s = self.snapshot()
        s.memory.append(entry)
        return s

    def set_action(self, action: str) -> "AgentState":
        s = self.snapshot()
        s.last_action = action
        return s

    def set_observation(self, observation: str) -> "AgentState":
        s = self.snapshot()
        s.last_observation = observation
        return s

    def add_reward(self, delta: float) -> "AgentState":
        s = self.snapshot()
        s.reward += delta
        return s

    def next_step(self) -> "AgentState":
        s = self.snapshot()
        s.step += 1
        return s


# -------------------------
# 示例演示
# -------------------------
if __name__ == "__main__":
    state0 = AgentState(agent_id="agent-001")
    state1 = (
        state0.add_memory("观察到房间内有一个箱子")
              .set_action("打开箱子")
              .set_observation("箱子里面有金币")
              .add_reward(5)
              .next_step()
    )
    state2 = (
        state1.add_memory("观察到箱子里面有金币,旁边5m内有售货机")
              .set_action("将金币投入售货机,买了一瓶可乐")
              .set_observation("可乐到手")
              .add_reward(-3)
              .next_step()
    )
    state3 = (
        state2.add_memory("可乐到手,还是冰镇的")
              .set_action("品尝可乐")
              .set_observation("可乐真美味")
              .next_step()
    )

    episode = Episode()
    episode = episode.add_step(state0)
    episode = episode.add_step(state1)
    episode = episode.add_step(state2)
    episode = episode.add_step(state3)
    episode.show()
