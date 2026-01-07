"""
===========================================================
Agent Simple v7 — Minimal End-to-End with Replay Foundation
===========================================================

这是一个 v7 阶段的最小可运行 Agent 架构示例，目标不是“聪明”，
而是 **结构正确、职责清晰、可 replay、可 offline eval**。

--------------------------------
v7 的核心设计原则
--------------------------------
1. State ≠ Decision
   - State 是世界状态，只能被 Runtime 修改
   - Decision 是一次性决策结果，可被 replay / 对比 / 评估

2. Agent 不产出 Action
   - Agent 只产出 DecisionInput（观察 + 约束）
   - “做什么”永远由 Provider + Arbitration 决定

3. Provider ≠ Final Decision
   - Provider 只能提出 DecisionProposal
   - Proposal 必须经过 Validation 才能生效

4. Runtime 是唯一的执行者
   - 只有 Runtime 能 apply Decision 到 State
   - 这是 replay / offline eval 的硬边界

5. Episode 是一等公民
   - Episode 不是 log，而是可复算、可分析的数据结构

--------------------------------
当前文件包含的模块
--------------------------------
- AgentState                : 智能体状态
- DecisionInput             : 给 Agent / LLM 的决策输入
- DecisionProposal          : Provider 的候选提案
- Decision                  : 最终可执行决策
- DecisionValidationLayer   : 安全与约束边界
- AgentRuntime              : 状态演化执行层
- Episode / EpisodeStep     : 轨迹记录（Replay 基础）

--------------------------------
刻意未包含的内容（v8 才引入）
--------------------------------
- 多 Provider Arbitration
- Offline Evaluator / A-B Test
- 长期 Memory / Reward
- 多 Step Loop

这是一个 **“结构先于智能”** 的 v7 基线实现。

===========================================================
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict


# =====================================================
# Agent State —— 智能体长期状态（v7：纯状态，不含决策）
# =====================================================
@dataclass
class AgentState:
    status: str
    user_intent: str
    meeting: Dict
    permissions: List[str] = field(default_factory=list)

    def snapshot(self) -> "AgentState":
        return deepcopy(self)

    def apply(self, decision: "Decision") -> "AgentState":
        """
        Runtime 调用：把 Decision 的结果写回 State
        """
        new_state = self.snapshot()
        if decision.type == "AskForConfirmation":
            new_state.status = "waiting_user_response"
        elif decision.type == "Abort":
            new_state.status = "aborted"
        return new_state


# =====================================================
# Episode —— 轨迹记录（Replay / Offline Eval 核心）
# =====================================================
@dataclass
class EpisodeStep:
    last_decision: Optional[str]
    state_snapshot: AgentState
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def show(self):
        print(f"Decision: {self.last_decision}")
        print(f"State: {self.state_snapshot.status}")
        print("-" * 50)


@dataclass
class Episode:
    steps: List[EpisodeStep] = field(default_factory=list)


# =====================================================
# Decision Input —— 给 LLM / Agent 的“观察窗口”
# =====================================================
@dataclass
class DecisionInput:
    task: str
    observable_state_snapshot: dict
    allowed_intents: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


# =====================================================
# Agent —— 只负责生成 DecisionInput（v7 关键点）
# =====================================================
class SimpleAgent:
    def decide(self, state: AgentState) -> DecisionInput:
        if state.status == "awaiting_user_confirmation":
            return DecisionInput(
                task="Handle user's request",
                observable_state_snapshot={
                    "status": state.status,
                    "meeting_time": state.meeting["time"],
                    "context": "User wants to reschedule a meeting",
                },
                allowed_intents=[
                    "AskForConfirmation",
                    "SendRescheduleEmail",
                    "Abort",
                ],
                constraints=[
                    "Do not send email without confirmation",
                    "Only propose one intent",
                ],
            )
        return DecisionInput(task="No-op", observable_state_snapshot={})


# =====================================================
# Scenario —— 人类叙事入口（初始化世界）
# =====================================================
class Scenario:
    def bootstrap(self) -> AgentState:
        return AgentState(
            status="awaiting_user_confirmation",
            user_intent="reschedule_meeting",
            meeting={
                "id": "m_123",
                "time": "2026-01-06 10:00",
            },
            permissions=["send_email"],
        )


# =====================================================
# Decision Proposal —— Provider 输出（尚未被采纳）
# =====================================================
@dataclass
class DecisionProposal:
    intent: str
    args: dict
    confidence: float


# =====================================================
# Decision —— Arbitration + Validation 之后的最终决策
# =====================================================
@dataclass
class Decision:
    type: str
    payload: dict


# =====================================================
# Validation Layer —— v7 安全边界
# =====================================================
class DecisionValidationLayer:
    def validate(
        self,
        state: AgentState,
        proposal: DecisionProposal,
        decision_input: DecisionInput,
    ) -> Decision:

        # intent 合法性
        assert proposal.intent in decision_input.allowed_intents

        # 类型校验
        assert isinstance(proposal.args.get("message"), str)

        # 状态前置条件
        if proposal.intent == "AskForConfirmation":
            assert state.status == "awaiting_user_confirmation"

        # 置信度下限
        assert proposal.confidence >= 0.6

        return Decision(
            type=proposal.intent,
            payload=proposal.args,
        )


# =====================================================
# Provider（现在是规则 / mock，未来是 LLM）
# =====================================================
class LLMDecisionProvider:
    name = "llm"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="AskForConfirmation",
            args={
                "message": "Do you want me to reschedule the meeting to a new time?"
            },
            confidence=0.71,
        )


class DecisionProviderRegistry:
    def choose(self, strategy: str):
        # v7：这里未来可以返回多个 provider
        return LLMDecisionProvider()


# =====================================================
# Runtime —— 系统执行层（唯一能改 State）
# =====================================================
class AgentRuntime:
    def execute(
        self,
        init_state: AgentState,
        decision: Decision,
    ) -> EpisodeStep:

        new_state = init_state.apply(decision)

        print("SYSTEM:", decision.payload.get("message"))

        return EpisodeStep(
            last_decision=decision.type,
            state_snapshot=new_state,
        )


# =====================================================
# Demo（单步 Episode，用于 Replay 基线）
# =====================================================
if __name__ == "__main__":
    scenario = Scenario()
    agent = SimpleAgent()
    runtime = AgentRuntime()
    registry = DecisionProviderRegistry()
    validator = DecisionValidationLayer()

    # 初始化
    state = scenario.bootstrap()
    episode = Episode()

    # Agent → DecisionInput
    decision_input = agent.decide(state)

    # Provider → Proposal
    provider = registry.choose(strategy="prefer_llm")
    proposal = provider.propose(decision_input)

    # Validation → Final Decision
    decision = validator.validate(state, proposal, decision_input)

    # Runtime 执行
    step = runtime.execute(state, decision)
    episode.steps.append(step)

    # 展示 Episode
    step.show()
