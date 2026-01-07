# agent_simple_v6_final.py
# =========================================================
# v6 FINAL 版本定位：
# - 完成 Agent / Decision / State / Runtime 的彻底解耦
# - Agent 只输出 Decision（意图）
# - State 只存“事实”
# - Runtime 是唯一推动世界前进的角色
# - Episode 记录「State + Decision」
#
# 👉 这是一个可以直接接 LLM 的架构
# =========================================================

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# -------------------------
# Agent State（世界事实）
# -------------------------
@dataclass
class AgentState:
    """
    AgentState：
    - 不包含“怎么想的”
    - 只包含“世界现在是什么样”
    """
    agent_id: str
    step: int = 0
    reward: float = 0.0
    memory: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    last_observation: Optional[str] = None

    def snapshot(self) -> "AgentState":
        return deepcopy(self)

    def apply(self, decision: "Decision") -> "AgentState":
        """
        核心跃迁点：
        - Agent 不再直接改 State
        - Runtime 调用 apply，把 Decision 映射为新 State
        """
        new_state = self.snapshot()

        if decision.memory:
            new_state.memory.append(decision.memory)

        if decision.action:
            new_state.last_action = decision.action

        if decision.observation:
            new_state.last_observation = decision.observation

        new_state.reward += decision.reward
        new_state.step += 1

        return new_state

    def show(self):
        print(f"Step {self.step} | Reward: {self.reward}")
        print(f"Memory: {self.memory}")
        print(f"Action: {self.last_action}")
        print(f"Observation: {self.last_observation}")
        print("-" * 50)


# -------------------------
# Decision（Agent 的输出）
# -------------------------
@dataclass
class Decision:
    """
    Decision：
    - 不是事实
    - 是“Agent 希望世界发生什么”
    """
    action: Optional[str] = None
    observation: Optional[str] = None
    memory: Optional[str] = None
    reward: float = 0.0


# -------------------------
# Episode（轨迹）
# -------------------------
@dataclass
class EpisodeStep:
    """
    EpisodeStep：
    - 记录 State + Decision
    - 允许完整复盘 Agent 行为
    """
    state: AgentState
    decision: Decision
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Episode:
    steps: List[EpisodeStep] = field(default_factory=list)

    def add(self, state: AgentState, decision: Decision) -> None:
        self.steps.append(EpisodeStep(state=state, decision=decision))

    def show(self):
        print("Episode trajectory:")
        for s in self.steps:
            print(
                f"Step {s.state.step} | "
                f"Action: {s.decision.action} | "
                f"Memory: {s.decision.memory} | "
                f"Reward: {s.decision.reward}"
            )
        print("=" * 60)


# -------------------------
# Agent（现在是规则，未来是 LLM）
# -------------------------
class SimpleAgent:
    """
    Agent 的唯一职责：
    - 读取 State
    - 输出 Decision
    """

    def decide(self, state: AgentState) -> Decision:
        if "箱子" in "".join(state.memory):
            return Decision(
                memory="检查箱子是否有金币",
                action="打开箱子",
                observation="箱子里面有金币",
                reward=1.0
            )
        return Decision()


# -------------------------
# Scenario（世界初始化）
# -------------------------
class Scenario:
    """
    用于：
    - 注入初始世界状态
    - 人类叙事入口
    """

    def bootstrap(self) -> AgentState:
        return AgentState(
            agent_id="agent-001",
            memory=["观察到房间内有一个箱子"]
        )


# -------------------------
# Runtime（系统心脏）
# -------------------------
class AgentRuntime:
    """
    Runtime：
    - 控制时间
    - 调用 Agent
    - 应用 Decision
    """

    def run(self, agent: SimpleAgent, init_state: AgentState, steps: int) -> Episode:
        state = init_state
        episode = Episode()

        for _ in range(steps):
            decision = agent.decide(state)
            episode.add(state, decision)
            state = state.apply(decision)

        return episode


# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    scenario = Scenario()
    agent = SimpleAgent()
    runtime = AgentRuntime()

    init_state = scenario.bootstrap()
    init_state.show()

    episode = runtime.run(agent, init_state, steps=3)
    episode.show()
