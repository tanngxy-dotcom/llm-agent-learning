"""
====================================================================
Agent Simple v7 — Multi-Provider Arbitration & Episode Distribution
====================================================================

这是一个 v7 阶段用于 **行为分布验证（Behavioral Distribution Test）**
的最小可运行实验脚本。

本文件的目标不是构建“智能 Agent”，而是验证：

------------------------------------------------------------
在多 Provider + 多 Arbitration 策略下
系统的行为是否【稳定、可预期、可解释】
------------------------------------------------------------

你可以用它回答以下问题：
- 不同 Arbitration 策略会产生怎样的 intent 分布？
- LLM 的不确定性会在什么条件下影响最终行为？
- 规则是否真的“兜住了系统底线”？
- 随机策略在 Offline Eval 中是否明显劣于基线？

------------------------------------------------------------
v7 核心思想回顾
------------------------------------------------------------
1. Provider 只负责「提案」，不负责决策
2. Proposal 永远不是 Action
3. Arbitration 是系统人格，而不是模型人格
4. 单次运行没有意义，分布才有意义
5. Episode 是 Offline Eval 的基本单位

------------------------------------------------------------
本文件包含内容
------------------------------------------------------------
- AgentState               : 智能体状态（最小化）
- DecisionInput            : Agent 可观察状态
- Providers                : 多种决策来源（LLM / Rule / Abort）
- ProposalScorer           : 多维评分系统
- Arbitrators              : 不同裁决策略
- Episode Runner           : 单次决策仿真
- Main                     : 批量 Episode 分布统计

------------------------------------------------------------
刻意不包含（v8 内容）
------------------------------------------------------------
- 多 step episode
- Memory / Reward
- Replay 存盘与重放
- 学习型 Arbitration

====================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from copy import deepcopy
import random
from collections import Counter


# =====================================================
# Agent State —— 世界状态（v7：无智能）
# =====================================================
@dataclass
class AgentState:
    """
    AgentState 代表系统当前所处的“世界状态”。

    v7 中的重要约束：
    - State 不包含决策逻辑
    - State 只能被 Runtime 修改
    - State 必须可 snapshot / replay
    """
    status: str
    user_intent: str
    meeting_time: str

    def snapshot(self):
        return deepcopy(self)

    def apply(self, decision: "Decision"):
        """
        Runtime 调用：根据 Decision 演化状态
        """
        new_state = self.snapshot()
        if decision.type == "AskForConfirmation":
            new_state.status = "waiting_user_response"
        elif decision.type == "Abort":
            new_state.status = "aborted"
        return new_state


# =====================================================
# Decision 相关数据结构
# =====================================================
@dataclass
class DecisionInput:
    """
    决策输入：
    Agent / LLM 能“看到”的世界切片
    """
    observable_state: Dict
    allowed_intents: List[str]
    constraints: List[str]


@dataclass
class DecisionProposal:
    """
    Provider 的候选提案（尚未被采纳）
    """
    intent: str
    args: dict
    confidence: float


@dataclass
class DecisionScore:
    """
    对 Proposal 的多维度评分结果
    """
    confidence: float
    intent_valid: bool
    constraint_violation: int
    provider_priority: int


@dataclass
class Decision:
    """
    Arbitration 之后的最终可执行决策
    """
    type: str
    payload: dict


# =====================================================
# Providers —— 决策来源（不拥有最终权力）
# =====================================================
class LLMProvider:
    """
    模拟 LLM Provider：
    - intent 随机
    - confidence 不稳定
    """
    name = "llm"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        intent = random.choice(
            ["AskForConfirmation", "SendRescheduleEmail"]
        )
        return DecisionProposal(
            intent=intent,
            args={"message": f"LLM suggests {intent}"},
            confidence=random.uniform(0.4, 0.95),
        )


class RuleProvider:
    """
    规则 Provider：
    - 稳定
    - 高优先级
    - 保守
    """
    name = "rule"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="AskForConfirmation",
            args={"message": "Rule: please confirm first"},
            confidence=0.7,
        )


class AbortProvider:
    """
    兜底 Provider：
    - 用于测试 Arbitration 是否允许危险路径
    """
    name = "abort"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="Abort",
            args={"reason": "User intent unclear"},
            confidence=0.6,
        )


# =====================================================
# Scorer —— 多维评分系统（v7 核心）
# =====================================================
class ProposalScorer:
    """
    将 Proposal 映射为可比较的 DecisionScore
    """

    def score(
        self,
        proposal: DecisionProposal,
        decision_input: DecisionInput,
        provider_name: str,
    ) -> DecisionScore:

        intent_valid = proposal.intent in decision_input.allowed_intents

        constraint_violation = 0
        if (
            proposal.intent == "SendRescheduleEmail"
            and "Do not send email without confirmation"
            in decision_input.constraints
        ):
            constraint_violation += 1

        provider_priority = {
            "rule": 3,
            "llm": 1,
            "abort": 2,
        }.get(provider_name, 0)

        return DecisionScore(
            confidence=proposal.confidence,
            intent_valid=intent_valid,
            constraint_violation=constraint_violation,
            provider_priority=provider_priority,
        )


# =====================================================
# Arbitrators —— 系统人格的具体体现
# =====================================================
class DecisionArbitrator:
    """
    基线 Arbitrator：
    - 先过滤非法 proposal
    - 再按 provider_priority
    - 再按 confidence
    """

    def choose(
        self, scored: List[Tuple[DecisionProposal, DecisionScore]]
    ) -> DecisionProposal:

        valid = [
            (p, s)
            for p, s in scored
            if s.intent_valid and s.constraint_violation == 0
        ]

        valid.sort(
            key=lambda x: (
                -x[1].provider_priority,
                -x[1].confidence,
            )
        )
        return valid[0][0]


class ConfidenceOnlyArbitrator(DecisionArbitrator):
    """
    危险策略：
    - 完全相信 confidence
    - 无视规则与优先级
    """

    def choose(self, scored):
        scored.sort(key=lambda x: -x[1].confidence)
        return scored[0][0]


class RandomArbitrator(DecisionArbitrator):
    """
    对照实验用：
    - 用于 Offline Eval 证明“随机很糟”
    """

    def choose(self, scored):
        return random.choice(scored)[0]


# =====================================================
# Runtime —— 执行层（最小化）
# =====================================================
class AgentRuntime:
    """
    Runtime 是唯一能修改 State 的组件
    """

    def execute(self, state: AgentState, decision: Decision):
        return state.apply(decision)


# =====================================================
# Episode Runner —— 单次决策仿真
# =====================================================
def run_episode(arbitrator) -> str:
    """
    执行一个最小 Episode：
    - 单 State
    - 单 Decision
    """

    state = AgentState(
        status="awaiting_user_confirmation",
        user_intent="reschedule_meeting",
        meeting_time="2026-01-06 10:00",
    )

    decision_input = DecisionInput(
        observable_state={
            "status": state.status,
            "meeting_time": state.meeting_time,
        },
        allowed_intents=[
            "AskForConfirmation",
            "SendRescheduleEmail",
            "Abort",
        ],
        constraints=["Do not send email without confirmation"],
    )

    providers = [
        LLMProvider(),
        RuleProvider(),
        AbortProvider(),
    ]

    scorer = ProposalScorer()
    proposals_scores = []

    for p in providers:
        proposal = p.propose(decision_input)
        score = scorer.score(proposal, decision_input, p.name)
        proposals_scores.append((proposal, score))

    chosen = arbitrator.choose(proposals_scores)

    decision = Decision(type=chosen.intent, payload=chosen.args)
    AgentRuntime().execute(state, decision)

    return decision.type


# =====================================================
# Main —— 批量 Episode 分布统计（Offline Eval）
# =====================================================
if __name__ == "__main__":
    EPISODES = 100

    arbitrators = {
        "baseline": DecisionArbitrator(),
        "confidence_only": ConfidenceOnlyArbitrator(),
        "random": RandomArbitrator(),
    }

    for name, arb in arbitrators.items():
        results = []

        for _ in range(EPISODES):
            result = run_episode(arb)
            results.append(result)

        counter = Counter(results)

        print("\n==============================")
        print(f"Arbitrator: {name}")
        print("==============================")
        for k, v in counter.items():
            print(f"{k:25s}: {v}")
