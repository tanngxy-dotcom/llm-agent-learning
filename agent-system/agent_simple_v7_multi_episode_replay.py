"""
====================================================================
Agent Simple v7 — Multi-Episode Replay & Arbitration Distribution
====================================================================

这是 v7 阶段的 **第二种 replay 形态**，与“单 Episode + A/B 对比”不同，
它关注的是：

------------------------------------------------------------
在大量真实 Episode 上：
- 不同 Arbitration 策略会产生怎样的 Intent 分布？
- LLM 的随机性会不会被规则压制？
- Abort 是否会被系统性边缘化？
------------------------------------------------------------

换句话说：
❌ 它不关心“单步对不对”
✅ 它关心“长期系统行为是什么样子”

------------------------------------------------------------
这一版与前一版 replay 的核心差异
------------------------------------------------------------

前一版：
- 强调 **单条 Episode 的可解释性**
- 对比 original / A / B
- 更偏 Debug / Case Study

这一版：
- 强调 **统计分布**
- 多 Episode（Monte Carlo）
- 更偏 Strategy Evaluation / Governance

------------------------------------------------------------
v7 在此文件中的关键思想
------------------------------------------------------------
1. Episode 是 frozen 的历史事实（不可修改）
2. Offline Replay ≠ Online Run（不重跑 provider）
3. Arbitration 是“人格”，不是逻辑分支
4. Intent 分布本身就是评测指标
5. 随机性是被显式建模的（RandomArbitrator）

------------------------------------------------------------
刻意不做的事情（v8 内容）
------------------------------------------------------------
- reward 学习
- intent correctness 判断
- feedback loop
- 自动 arbitration tuning

====================================================================
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from copy import deepcopy
from collections import Counter
import random
import json


# =====================================================
# Agent State —— 世界状态（冻结快照）
# =====================================================
@dataclass
class AgentState:
    """
    Agent 的最小状态模型

    v7 原则：
    - 只描述事实
    - 不包含任何策略判断
    """
    status: str
    user_intent: str
    meeting_time: str

    def snapshot(self):
        return deepcopy(self)

    def apply(self, decision: "Decision"):
        """
        Runtime 语义上的状态演化（这里只是占位）
        """
        new_state = self.snapshot()
        if decision.type == "AskForConfirmation":
            new_state.status = "waiting_user_response"
        elif decision.type == "Abort":
            new_state.status = "aborted"
        return new_state


# =====================================================
# Decision Core Types
# =====================================================
@dataclass
class DecisionInput:
    """
    决策时 Agent/Provider 能看到的世界切片
    """
    observable_state: Dict
    allowed_intents: List[str]
    constraints: List[str]


@dataclass
class DecisionProposal:
    """
    Provider 给出的候选决策
    """
    intent: str
    args: dict
    confidence: float


@dataclass
class DecisionScore:
    """
    Proposal 的可比较评分结果
    """
    confidence: float
    intent_valid: bool
    constraint_violation: int
    provider_priority: int


@dataclass
class Decision:
    """
    最终被执行的决策（本文件中不真正执行）
    """
    type: str
    payload: dict


# =====================================================
# Providers —— 多源决策候选
# =====================================================
class LLMProvider:
    """
    模拟 LLM：
    - intent 随机
    - confidence 连续分布
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
    规则系统：
    - 稳定
    - 可预测
    - 高优先级
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
    防御性 Provider：
    - 只在不确定时兜底
    """
    name = "abort"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="Abort",
            args={"reason": "User intent unclear"},
            confidence=0.6,
        )


# =====================================================
# Scorer —— Proposal → Score
# =====================================================
class ProposalScorer:
    """
    显式、可重算的评分系统
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
            "abort": 2,
            "llm": 1,
        }.get(provider_name, 0)

        return DecisionScore(
            confidence=proposal.confidence,
            intent_valid=intent_valid,
            constraint_violation=constraint_violation,
            provider_priority=provider_priority,
        )


# =====================================================
# Arbitrators —— 不同“人格”
# =====================================================
class DecisionArbitrator:
    """
    Baseline Arbitration：
    - priority first
    - confidence second
    """
    name = "baseline"

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
    极端人格：
    - 完全相信 confidence
    """
    name = "confidence_only"

    def choose(self, scored):
        scored.sort(key=lambda x: -x[1].confidence)
        return scored[0][0]


class RandomArbitrator(DecisionArbitrator):
    """
    对照组：
    - 无理性
    - 仅用于 sanity check
    """
    name = "random"

    def choose(self, scored):
        return random.choice(scored)[0]


# =====================================================
# Episode —— 冻结的历史事实
# =====================================================
@dataclass
class EpisodeStep:
    """
    单步决策记录（不可变）
    """
    state_snapshot: AgentState
    decision_input: DecisionInput
    proposals: Dict[str, DecisionProposal]
    scores: Dict[str, DecisionScore]
    chosen_intent: str


@dataclass
class Episode:
    """
    一个 Episode = 一次决策实验样本
    """
    steps: List[EpisodeStep]


# =====================================================
# Online Episode Runner
# =====================================================
def run_online_episode(arbitrator) -> Episode:
    """
    在线运行：
    - provider 真跑
    - arbitration 真选
    - 历史被冻结
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

    proposals = {}
    scores = {}
    scored_list = []

    for p in providers:
        proposal = p.propose(decision_input)
        score = scorer.score(proposal, decision_input, p.name)
        proposals[p.name] = proposal
        scores[p.name] = score
        scored_list.append((proposal, score))

    chosen = arbitrator.choose(scored_list)

    return Episode(
        steps=[
            EpisodeStep(
                state_snapshot=state.snapshot(),
                decision_input=decision_input,
                proposals=proposals,
                scores=scores,
                chosen_intent=chosen.intent,
            )
        ]
    )


# =====================================================
# Offline Replay —— 核心实验能力
# =====================================================
def replay_episode(episode: Episode, arbitrator) -> List[str]:
    """
    不重跑 Provider
    只重算 Arbitration
    """
    intents = []

    for step in episode.steps:
        scored = list(
            zip(step.proposals.values(), step.scores.values())
        )
        chosen = arbitrator.choose(scored)
        intents.append(chosen.intent)

    return intents


# =====================================================
# Main —— 分布对比实验
# =====================================================
if __name__ == "__main__":
    EPISODES = 50

    online_arbitrator = DecisionArbitrator()

    replay_arbitrators = [
        DecisionArbitrator(),
        ConfidenceOnlyArbitrator(),
        RandomArbitrator(),
    ]

    episodes: List[Episode] = []

    # === Online run（冻结历史） ===
    for _ in range(EPISODES):
        ep = run_online_episode(online_arbitrator)
        episodes.append(ep)

    print("=== Online Decisions ===")
    online_results = Counter(
        step.chosen_intent
        for ep in episodes
        for step in ep.steps
    )
    print(online_results)

    # === Offline Replay（策略对比） ===
    for arb in replay_arbitrators:
        replay_results = []

        for ep in episodes:
            replay_results.extend(replay_episode(ep, arb))

        print("\n=== Replay with Arbitrator:", arb.name, "===")
        print(Counter(replay_results))
