"""
====================================================================
Agent Simple v7 — Replayable Episode & Offline A/B Evaluation
====================================================================

这是一个 v7 阶段的 **完整实验型 Agent 示例**，重点不在于“在线智能”，
而在于：

------------------------------------------------------------
让每一次决策都可以：
- 被完整记录（Episode）
- 被序列化存盘（Replay）
- 被不同策略重新裁决（Offline A/B Eval）
------------------------------------------------------------

这份代码回答的不是：
❌ “这次 Agent 做得对不对？”

而是：
✅ “如果当时换一种 Arbitration 策略，系统会不会做出不同选择？”
✅ “规则 vs LLM，在同一批 Proposal 上谁更稳定？”
✅ “当前系统的人格边界是否清晰？”

------------------------------------------------------------
v7 设计核心（在此文件中全部落地）
------------------------------------------------------------
1. State 是唯一事实来源（source of truth）
2. Provider 只产出 Proposal，不产出 Decision
3. Scoring 是显式的、可重算的
4. Arbitration 是可替换、可对比的
5. Episode 是一等数据资产，而不是日志
6. Offline Eval 不依赖 Runtime，不依赖真实执行

------------------------------------------------------------
本文件包含模块
------------------------------------------------------------
- AgentState                : 世界状态
- DecisionInput             : 决策观察窗口
- Providers                 : 不同决策来源
- ProposalScorer            : 多维评分
- Arbitrators               : 多种裁决人格
- Runtime                   : 在线执行（最小）
- Episode / EpisodeStore    : 决策轨迹与持久化
- ABOfflineEvaluator        : 离线 A/B 对比器

------------------------------------------------------------
刻意不包含（v8 内容）
------------------------------------------------------------
- 多 step episode loop
- 自动 reward / learning
- 动态 provider 策略
- 模型训练或自我更新

这是一个 **“可解释性优先于智能性”** 的 v7 标准实现。

====================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict
from copy import deepcopy
from datetime import datetime
import json


# =====================================================
# Agent State —— 系统唯一事实来源（Source of Truth）
# =====================================================
@dataclass
class AgentState:
    """
    AgentState 表示当前世界状态。

    v7 约束：
    - 不包含决策逻辑
    - 不依赖 Provider
    - 只能通过 Runtime.apply 演化
    """
    status: str
    user_intent: str
    meeting: dict

    def snapshot(self):
        return deepcopy(self)

    def apply(self, decision: "Decision"):
        new_state = self.snapshot()
        if decision.type == "AskForConfirmation":
            new_state.status = "waiting_user_response"
        return new_state


# =====================================================
# Decision 核心类型
# =====================================================
@dataclass
class DecisionInput:
    """
    Agent / LLM 能看到的决策输入（部分世界）
    """
    task: str
    observable_state_snapshot: dict
    allowed_intents: List[str]
    constraints: List[str]


@dataclass
class DecisionProposal:
    """
    Provider 给出的候选提案（尚未生效）
    """
    intent: str
    args: dict
    confidence: float


@dataclass
class Decision:
    """
    Arbitration 后的最终决策
    """
    type: str
    payload: dict


# =====================================================
# Providers —— 决策来源（无最终权力）
# =====================================================
class LLMDecisionProvider:
    """
    模拟 LLM 决策来源：
    - 较低置信度
    - 偏探索
    """
    name = "llm"

    def propose(self, decision_input: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="AskForConfirmation",
            args={"message": "Do you want me to reschedule the meeting?"},
            confidence=0.65,
        )


class RuleBasedDecisionProvider:
    """
    规则决策来源：
    - 高置信度
    - 稳定、保守
    """
    name = "rule"

    def propose(self, decision_input: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="AskForConfirmation",
            args={"message": "Please confirm if you want to reschedule the meeting."},
            confidence=0.95,
        )


# =====================================================
# Scoring & Arbitration
# =====================================================
@dataclass
class DecisionScore:
    """
    Proposal 的可比较评分向量
    """
    confidence: float
    intent_valid: bool
    constraint_violation: int
    provider_priority: int


class ProposalScorer:
    """
    将 Proposal 映射为 DecisionScore
    """

    def score(
        self,
        proposal: DecisionProposal,
        decision_input: DecisionInput,
        provider_name: str,
    ) -> DecisionScore:

        intent_valid = proposal.intent in decision_input.allowed_intents

        constraint_violation = 0
        for c in decision_input.constraints:
            if "Do not send email" in c and proposal.intent == "SendRescheduleEmail":
                constraint_violation += 1

        provider_priority = {
            "rule": 2,
            "llm": 1,
        }.get(provider_name, 0)

        return DecisionScore(
            confidence=proposal.confidence,
            intent_valid=intent_valid,
            constraint_violation=constraint_violation,
            provider_priority=provider_priority,
        )


class DecisionArbitrator:
    """
    基线 Arbitrator：
    - 先过滤非法 proposal
    - 再按 priority
    - 再按 confidence
    """

    def choose(self, scored):
        valid = [
            (p, s)
            for p, s in scored
            if s.intent_valid and s.constraint_violation == 0
        ]

        if not valid:
            raise RuntimeError("No valid proposals")

        valid.sort(
            key=lambda x: (
                x[1].constraint_violation,
                -x[1].provider_priority,
                -x[1].confidence,
            )
        )
        return valid[0][0]


class RuleOnlyArbitrator(DecisionArbitrator):
    """
    极端策略：
    - 完全由规则主导
    """

    def choose(self, scored):
        scored.sort(key=lambda x: -x[1].provider_priority)
        return scored[0][0]


class LLMOnlyArbitrator(DecisionArbitrator):
    """
    极端策略：
    - 完全由 confidence 主导
    """

    def choose(self, scored):
        scored.sort(key=lambda x: -x[1].confidence)
        return scored[0][0]


# =====================================================
# Runtime —— 在线执行层（最小实现）
# =====================================================
class AgentRuntime:
    """
    Runtime 是唯一允许修改 State 的组件
    """

    def execute(self, state, decision):
        print(f"[Runtime] {decision.payload['message']}")
        return state.apply(decision)


# =====================================================
# Episode / Replay —— v7 的“时间机器”
# =====================================================
@dataclass
class EpisodeStep:
    """
    单步决策的完整记录
    """
    timestamp: str
    state_snapshot: AgentState
    decision_input: DecisionInput
    proposals: Dict[str, DecisionProposal]
    scores: Dict[str, DecisionScore]
    chosen_provider: str
    decision: Decision


@dataclass
class Episode:
    """
    一条可 replay 的决策轨迹
    """
    steps: List[EpisodeStep] = field(default_factory=list)

    def add(self, step: EpisodeStep):
        self.steps.append(step)


class EpisodeStore:
    """
    Episode 持久化工具
    """

    def save(self, episode: Episode, path: str):
        with open(path, "w") as f:
            json.dump(
                [self._serialize_step(s) for s in episode.steps],
                f,
                indent=2,
            )

    def _serialize_step(self, step: EpisodeStep):
        return {
            "timestamp": step.timestamp,
            "state": step.state_snapshot.__dict__,
            "decision_input": step.decision_input.__dict__,
            "proposals": {k: v.__dict__ for k, v in step.proposals.items()},
            "scores": {k: v.__dict__ for k, v in step.scores.items()},
            "chosen_provider": step.chosen_provider,
            "decision": step.decision.__dict__,
        }


# =====================================================
# Offline A/B Evaluator —— 不重跑世界，只重算决策
# =====================================================
class ABOfflineEvaluator:
    """
    在同一批 Episode 数据上对比不同 Arbitrator
    """

    def evaluate(self, episode: Episode, arbitrator_A, arbitrator_B):
        print("\n=== Offline A/B Replay ===")

        for i, step in enumerate(episode.steps):
            scored = list(
                zip(step.proposals.values(), step.scores.values())
            )

            a = arbitrator_A.choose(scored)
            b = arbitrator_B.choose(scored)

            print(
                f"[Step {i}] "
                f"original={step.decision.type} | "
                f"A={a.intent} | "
                f"B={b.intent}"
            )


# =====================================================
# Scenario & DecisionInput Builder
# =====================================================
def build_decision_input(state: AgentState) -> DecisionInput:
    """
    从 State 构建 DecisionInput
    """
    return DecisionInput(
        task="Handle user request",
        observable_state_snapshot={
            "status": state.status,
            "meeting_time": state.meeting["time"],
        },
        allowed_intents=[
            "AskForConfirmation",
            "SendRescheduleEmail",
            "Abort",
        ],
        constraints=[
            "Do not send email without confirmation",
        ],
    )


# =====================================================
# Demo —— Online Run + Offline Replay
# =====================================================
if __name__ == "__main__":
    # 初始化世界
    state = AgentState(
        status="awaiting_user_confirmation",
        user_intent="reschedule_meeting",
        meeting={"id": "m_123", "time": "2026-01-06 10:00"},
    )

    providers = [
        LLMDecisionProvider(),
        RuleBasedDecisionProvider(),
    ]

    scorer = ProposalScorer()
    arbitrator = DecisionArbitrator()
    runtime = AgentRuntime()

    episode = Episode()

    # === Online 决策 ===
    decision_input = build_decision_input(state)

    proposals = {}
    scores = {}

    for p in providers:
        proposal = p.propose(decision_input)
        proposals[p.name] = proposal
        scores[p.name] = scorer.score(proposal, decision_input, p.name)

    chosen = arbitrator.choose(
        list(zip(proposals.values(), scores.values()))
    )

    chosen_provider = next(
        name for name, p in proposals.items() if p == chosen
    )

    decision = Decision(type=chosen.intent, payload=chosen.args)
    new_state = runtime.execute(state, decision)

    episode.add(
        EpisodeStep(
            timestamp=datetime.now().isoformat(),
            state_snapshot=state.snapshot(),
            decision_input=decision_input,
            proposals=proposals,
            scores=scores,
            chosen_provider=chosen_provider,
            decision=decision,
        )
    )

    # === 保存 Episode ===
    EpisodeStore().save(episode, "episode.json")

    # === Offline A/B Replay ===
    evaluator = ABOfflineEvaluator()
    evaluator.evaluate(
        episode,
        arbitrator_A=RuleOnlyArbitrator(),
        arbitrator_B=LLMOnlyArbitrator(),
    )
