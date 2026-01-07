"""
agent_state_v2_final.py

V2 Final 版本：
---------------
- AgentState immutable 风格
- 封装方法: add_memory, set_action, set_observation, add_reward
- snapshot 深拷贝
- 演示链式操作
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AgentState:
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    last_observation: Optional[str] = None

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

    def show(self):
        print(f"Step {self.step} | Reward: {self.reward}")
        print(f"Memory: {self.memory}")
        print(f"Last action: {self.last_action}")
        print(f"Last observation: {self.last_observation}")
        print("-" * 50)


# -------------------------
# 示例演示
# -------------------------
if __name__ == "__main__":
    state0 = AgentState(agent_id="agent-001")
    state0.show()

    state1 = (
        state0.add_memory("观察到房间内有一个箱子")
              .set_action("打开箱子")
              .set_observation("箱子里面有金币")
              .add_reward(5)
              .next_step()
    )
    state1.show()

    state2 = (
        state1.add_memory("观察到箱子里面有金币,旁边5m内有售货机")
              .set_action("将金币投入售货机,买了一瓶可乐")
              .set_observation("可乐到手")
              .add_reward(-3)
              .next_step()
    )
    state2.show()

    state3 = (
        state2.add_memory("可乐到手,还是冰镇的")
              .set_action("品尝可乐")
              .set_observation("可乐真美味")
              .next_step()
    )
    state3.show()
