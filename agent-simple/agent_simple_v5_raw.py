# agent_simple_v5_raw.py
# =========================================================
# v5 RAW 版本定位：
# - 核心思想：State = 事实快照（immutable 风格）
# - Agent 仍然直接“生成下一个 State”
# - Runtime 负责推进时间，但不做状态解释
# - Episode 只是状态轨迹的简单记录
#
# 👉 这是从「状态即行为结果」走向
#    「状态 / 决策 / Runtime 解耦」之前的关键过渡版本
# =========================================================

import json
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional


# -------------------------
# Agent State（事实快照）
# -------------------------
@dataclass()
class AgentState:
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    last_observation: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def snapshot(self) -> "AgentState":
        """
        返回当前状态的深拷贝
        核心语义：State 是不可变的，每一次变化都会生成新 State
        """
        return deepcopy(self)

    # ---- 以下方法都是“状态演化 API” ----
    # 每个方法都会返回一个全新的 AgentState

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

    def next_step(self) -> "AgentState":
        """
        时间推进：
        step 的递增依然发生在 State 内部
        （这在 v6 会被 Runtime 接管）
        """
        new_state = self.snapshot()
        new_state.step += 1
        return new_state

    # ---- 序列化相关 ----
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


# -------------------------
# Episode Step（状态轨迹节点）
# -------------------------
@dataclass
class EpisodeStep:
    """
    v5 中：
    - EpisodeStep 只是“状态的记录”
    - action / observation / reward 仍然直接来自 State
    """
    state: AgentState
    action: Optional[str] = None
    observation: Optional[str] = None
    reward: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# -------------------------
# Episode（轨迹）
# -------------------------
@dataclass
class Episode:
    steps: List[EpisodeStep] = field(default_factory=list)

    def snapshot(self) -> "Episode":
        """Episode 也遵循不可变设计"""
        return deepcopy(self)

    def add_step(self, state: AgentState) -> "Episode":
        """
        记录某一时刻的 State
        """
        new_episode = self.snapshot()
        step_record = EpisodeStep(
            state=state,
            action=state.last_action,
            observation=state.last_observation,
            reward=state.reward
        )
        new_episode.steps.append(step_record)
        return new_episode

    def show(self):
        print("Episode trajectory:")
        for s in self.steps:
            print(
                f"Step {s.state.step} | "
                f"Reward: {s.reward} | "
                f"Action: {s.action} | "
                f"Observation: {s.observation}"
            )
        print("=" * 60)


# -------------------------
# Agent（仍然直接“产出 State”）
# -------------------------
@dataclass()
class SimpleAgent:
    """
    v5 的 Agent 仍然：
    - 直接修改 / 生成下一个 State
    - 尚未引入 Decision 概念
    """
    def act(self, state: AgentState) -> "AgentState":
        if "箱子" in "".join(state.memory):
            return (
                state
                .add_memory("检查箱子是否有金币")
                .set_action("打开箱子")
                .add_reward(0.0)
                .next_step()
            )
        else:
            return state.next_step()


# -------------------------
# Runtime（执行驱动）
# -------------------------
@dataclass()
class AgentRuntime:
    """
    v5 Runtime 的职责：
    - 驱动循环
    - 调用 agent.act
    - 记录 Episode
    """
    def run(self, agent: SimpleAgent, init_state: AgentState, steps=3) -> "Episode":
        state = init_state
        episode = Episode()

        for _ in range(steps):
            episode = episode.add_step(state)
            state = agent.act(state)

        return episode


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

    episode = Episode()
    episode = episode.add_step(state0)
    episode = episode.add_step(state1)
    episode = episode.add_step(state2)
    episode = episode.add_step(state3)
    episode.show()

    agent = SimpleAgent()
    runtime = AgentRuntime()
    runtime.run(agent, state1, 3).show()
