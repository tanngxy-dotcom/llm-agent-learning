"""
agent_state_v1_final.py

V1 Final 版本：
---------------
- 基础 AgentState
- 支持 snapshot、序列化/反序列化
- 手动更新 step/memory/action/observation/reward
- 增加 show 方法展示状态
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
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

    def to_dict(self) -> dict:
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


# -------------------------
# 示例演示
# -------------------------
if __name__ == "__main__":
    state = AgentState(agent_id="agent-001")
    state.memory.append("观察到房间内有一个箱子")
    state.last_action = "打开箱子"
    state.last_observation = "箱子里面有金币"
    state.reward = 5
    state.step += 1

    print("初始状态:")
    state.show()

    # 快照演示
    snap = state.snapshot()
    print("快照状态:")
    snap.show()

    # JSON 序列化演示
    json_str = state.to_json()
    print("JSON:", json_str)
    state_back = AgentState.from_json(json_str)
    print("还原对象:")
    state_back.show()
