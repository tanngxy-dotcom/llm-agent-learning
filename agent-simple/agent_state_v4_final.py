"""
文件名: agent_state_v4_final.py
版本: v4-final
功能:
    - 将 AgentState、EpisodeStep、Episode 合并在同一文件
    - 支持状态快照、memory/action/observation/reward 修改
    - Episode 支持添加 step 并记录 state 的快照
相较于 v3-final:
    - 集成 Episode 逻辑，不再依赖外部文件
    - 支持 next_step 方法增加 step
Demo:
    - 演示多步状态修改
    - 构建 Episode 并显示完整轨迹
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from copy import deepcopy
from datetime import datetime

@dataclass
class AgentState:
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    last_observation: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

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

    def to_dict(self):
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "AgentState":
        data = json.loads(s)
        return AgentState(**data)

    def show(self):
        print(f"Step {self.step} | Reward: {self.reward}")
        print(f"Memory: {self.memory}")
        print(f"Last action: {self.last_action}")
        print(f"Last observation: {self.last_observation}")
        print("-" * 50)

@dataclass
class EpisodeStep:
    state: AgentState
    action: Optional[str] = None
    observation: Optional[str] = None
    reward: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Episode:
    steps: List[EpisodeStep] = field(default_factory=list)

    def snapshot(self) -> "Episode":
        return deepcopy(self)

    def add_step(self, state: AgentState) -> "Episode":
        ep = self.snapshot()
        ep.steps.append(EpisodeStep(
            state=state,
            action=state.last_action,
            observation=state.last_observation,
            reward=state.reward
        ))
        return ep

    def show(self):
        print("Episode trajectory:")
        for s in self.steps:
            print(f"Step {s.state.step} | Reward: {s.reward} | Action: {s.action} | Observation: {s.observation}")
        print("=" * 60)


# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    state0 = AgentState(agent_id="agent-001")
    state0.show()

    state1 = state0.add_memory("观察到房间内有一个箱子").set_action("打开箱子").set_observation("箱子里面有金币").add_reward(5).next_step()
    state1.show()

    state2 = state1.add_memory("观察到箱子里面有金币,旁边5m内有售货机").set_action("将金币投入售货机,买了一瓶可乐").set_observation("可乐到手").add_reward(-3).next_step()
    state2.show()

    state3 = state2.add_memory("可乐到手,还是冰镇的").set_action("品尝可乐").set_observation("可乐真美味").next_step()
    state3.show()

    episode = Episode()
    episode = episode.add_step(state0).add_step(state1).add_step(state2).add_step(state3)
    episode.show()
