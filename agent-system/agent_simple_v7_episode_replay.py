"""
====================================================================
Agent Simple v7 — Episode JSON & True Offline Replay
====================================================================

这一文件只做一件事，但把它做到“不可替代”：

------------------------------------------------------------
1. Online：
   - Provider 生成 proposal
   - Scorer 计算 score
   - Arbitrator 选中 intent
   - 将“决策现场”冻结为 Episode
   - dump 成 JSON

2. Offline：
   - 不调用 Provider
   - 不重新 score
   - 仅加载 Episode JSON
   - 用不同 Arbitrator 重放决策

------------------------------------------------------------
这标志着 v7 正式完成：
✔ 决策可回放
✔ 人格可替换
✔ 行为可离线评估

注意：
- 这是 v7，不涉及 learning / reward（v8 才有）
====================================================================
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from copy import deepcopy
from collections import Counter
import random
import json


# =====================================================
# Agent State（决策上下文，而非 Runtime 状态）
# =====================================================
@dataclass
class AgentState:
    status: str
    user_intent: str
    meeting_time: str

    def snapshot(self):
        return deepcopy(self)


# =====================================================
# Decision Core Types
# =====================================================
@dataclass
class DecisionInput:
    observable_state: Dict
    allowed_intents: List[str]
    constraints: List[str]


@dataclass
class DecisionProposal:
    intent: str
    args: dict
    confidence: float


@dataclass
class DecisionScore:
    confidence: float
    intent_valid: bool
    constraint_violation: int
    provider_priority: int


# =====================================================
# Providers（仅 Online 使用）
# =====================================================
class LLMProvider:
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
    name = "rule"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="AskForConfirmation",
            args={"message": "Rule: please confirm first"},
            confidence=0.7,
        )


class AbortProvider:
    name = "abort"

    def propose(self, _: DecisionInput) -> DecisionProposal:
        return DecisionProposal(
            intent="Abort",
            args={"reason": "User intent unclear"},
            confidence=0.6,
        )


# =====================================================
# Scorer（决策评价体系）
# =====================================================
class ProposalScorer:
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
# Arbitrators（系统人格）
# =====================================================
class DecisionArbitrator:
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
    name = "confidence_only"

    def choose(self, scored):
        scored.sort(key=lambda x: -x[1].confidence)
        return scored[0][0]


class RandomArbitrator(DecisionArbitrator):
    name = "random"

    def choose(self, scored):
        return random.choice(scored)[0]


# =====================================================
# Episode（冻结的决策事实）
# =====================================================
@dataclass
class EpisodeStep:
    state_snapshot: AgentState
    decision_input: DecisionInput
    proposals: Dict[str, DecisionProposal]
    scores: Dict[str, DecisionScore]
    chosen_intent: str


@dataclass
class Episode:
    steps: List[EpisodeStep]


# =====================================================
# Serialization（Episode <-> JSON）
# =====================================================
def serialize_episode(ep: Episode) -> dict:
    return {
        "steps": [
            {
                "state_snapshot": asdict(step.state_snapshot),
                "decision_input": asdict(step.decision_input),
                "proposals": {
                    k: asdict(v) for k, v in step.proposals.items()
                },
                "scores": {
                    k: asdict(v) for k, v in step.scores.items()
                },
                "chosen_intent": step.chosen_intent,
            }
            for step in ep.steps
        ]
    }


def deserialize_episode(data: dict) -> Episode:
    steps = []
    for s in data["steps"]:
        steps.append(
            EpisodeStep(
                state_snapshot=AgentState(**s["state_snapshot"]),
                decision_input=DecisionInput(**s["decision_input"]),
                proposals={
                    k: DecisionProposal(**v)
                    for k, v in s["proposals"].items()
                },
                scores={
                    k: DecisionScore(**v)
                    for k, v in s["scores"].items()
                },
                chosen_intent=s["chosen_intent"],
            )
        )
    return Episode(steps=steps)


# =====================================================
# Online Episode Runner
# =====================================================
def run_online_episode(arbitrator) -> Episode:
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
# Offline Replay
# =====================================================
def replay_episode(ep: Episode, arbitrator) -> List[str]:
    intents = []
    for step in ep.steps:
        scored = list(
            zip(step.proposals.values(), step.scores.values())
        )
        chosen = arbitrator.choose(scored)
        intents.append(chosen.intent)
    return intents


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    EPISODES = 100

    # === Online run ===
    online_arb = DecisionArbitrator()
    episodes = [run_online_episode(online_arb) for _ in range(EPISODES)]

    with open("episodes.json", "w", encoding="utf-8") as f:
        json.dump(
            [serialize_episode(ep) for ep in episodes],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=== Online Distribution ===")
    print(
        Counter(
            step.chosen_intent
            for ep in episodes
            for step in ep.steps
        )
    )

    # === Offline replay ===
    with open("episodes.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)

    loaded_eps = [deserialize_episode(e) for e in loaded]

    for arb in [
        DecisionArbitrator(),
        ConfidenceOnlyArbitrator(),
        RandomArbitrator(),
    ]:
        results = []
        for ep in loaded_eps:
            results.extend(replay_episode(ep, arb))

        print(f"\n=== Replay with {arb.name} ===")
        print(Counter(results))
