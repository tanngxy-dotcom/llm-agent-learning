"""
agent_state_v3_raw.py

V3 Raw 版本：
-------------
- 引入 Episode 对象
- 使用 AgentState immutable 方法生成每个 step
- Episode 记录每个 step 的 snapshot
- 演示链式状态修改
"""

from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

@dataclass
class Episode:
    steps: List[int] = field(default_factory=list)

    def snapshot(self) -> "Episode":
        return deepcopy(self)

    def add_step(self, step: int) -> "Episode":
        new_episode = self.snapshot()
        new_episode.steps.append(step)
        return new_episode

@dataclass
class AgentState:
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: List[str] = field(default_factory=list)
    last_action: str = None
    last_observation: str = None

    def snapshot(self) -> "AgentState":
        return deepcopy(self)

    def add_memory(self, entry: str) -> "AgentState":
        new_state = self.snapshot()
        new_state.memory.append(entry)
        return new_state

    def set_action(self, action: str) -> "AgentState":
        new_state = self.snapshot()
        new_state.last_action = action
        return new_state

    def set_observation(self, observation: str) -> "AgentState":
        new_state = self.snapshot()
        new_state.last_observation = observation
        return new_state

    def add_reward(self, delta: float) -> "AgentState":
        new_state = self.snapshot()
        new_state.reward += delta
        return new_state


# -------------------------
# 示例演示
# -------------------------
if __name__ == "__main__":
    state0 = AgentState(agent_id="agent-001")
    state1 = state0.add_memory("观察到房间内有一个箱子").set_action("打开箱子").set_observation("箱子里面有金币")
    state1.step = 1
    state1.add_reward(5)

    state2 = state1.add_memory("观察到箱子里面有金币,旁边5m内有售货机").set_action("将金币投入售货机,买了一瓶可乐").set_observation("可乐到手")
    state2.step = 2
    state2.add_reward(-3)

    state3 = state2.add_memory("可乐到手,还是冰镇的").set_action("品尝可乐").set_observation("可乐真美味")
    state3.step = 3

    task1 = Episode()
    task1 = task1.add_step(state0.step).add_step(state1.step).add_step(state2.step).add_step(state3.step)
    for step in task1.steps:
        print(step)
