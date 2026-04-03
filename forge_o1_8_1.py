from __future__ import annotations

import argparse
import hashlib
import json
import heapq
import math
import random
import statistics
from collections import deque
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


                                                              
                     
                                                              


@dataclass
class Arena(Generic[T]):
    name: str
    cases: List[Tuple[str, T]]


@dataclass
class EvalMetrics:
    correct: int
    total: int
    avg_cost: float
    worst_cost: float
    complexity: float
    scalar: float
    failures: Dict[str, int]


@dataclass(frozen=True)
class DomainOperator(Generic[T]):
    name: str
    fn: Callable[[T], T]
    tags: Tuple[str, ...] = ()


@dataclass
class ThoughtHypothesis:
    kind: str
    target: str
    confidence: float
    reason: str
    expected_effect: str
    status: str = "pending"


@dataclass
class PlannerThought:
    depth: int
    mode: str
    root_family: str
    pathology: str
    beam_best_scalar: float
    beam_diversity: int
    prioritized_ops: List[str]
    rejected_ops: List[str]
    beam_width: int
    suggest_reforge: bool
    convergence: str
    strategy: str
    uncertainty: float
    exploit_weight: float
    diversify_weight: float
    hidden_probe: bool
    local_refine: bool
    family_hypotheses: List[Tuple[str, float]]
    stagnation_steps: int
    hypotheses: List[ThoughtHypothesis]
    note: str

    def short(self) -> str:
        pri = ", ".join(self.prioritized_ops[:3]) if self.prioritized_ops else "none"
        rej = ", ".join(self.rejected_ops[:2]) if self.rejected_ops else "none"
        fam = ", ".join(f"{name}:{prob:.2f}" for name, prob in self.family_hypotheses[:2]) if self.family_hypotheses else "unknown"
        return (
            f"depth={self.depth} mode={self.mode} family={self.root_family} pathology={self.pathology} "
            f"beam_best={self.beam_best_scalar:.3f} diversity={self.beam_diversity} convergence={self.convergence} "
            f"strategy={self.strategy} uncertainty={self.uncertainty:.2f} stagnation={self.stagnation_steps} "
            f"families=[{fam}] prioritize=[{pri}] reject=[{rej}] beam_width={self.beam_width} "
            f"probe={self.hidden_probe} refine={self.local_refine} reforge={self.suggest_reforge}"
        )

@dataclass
class StepRecord(Generic[T]):
    candidate: T
    parent_hash: Optional[str]
    op_name: Optional[str]
    planner: EvalMetrics
    proof: Optional[EvalMetrics]
    hidden: Optional[EvalMetrics]
    depth: int
    thought: Optional[PlannerThought] = None


@dataclass
class Trajectory(Generic[T]):
    domain_name: str
    mode: str
    root_name: str
    root: T
    champion: T
    champion_hash: str
    path_hashes: List[str]
    steps: Dict[str, StepRecord[T]]
    champion_before_reforge: T
    reforge_op: Optional[str]


@dataclass
class PatternCard:
    domain_name: str
    motif: str
    mode: str
    root_family: str
    pathology: str
    support: int = 0
    planner_gain: List[float] = field(default_factory=list)
    proof_gain: List[float] = field(default_factory=list)
    hidden_gain: List[float] = field(default_factory=list)
    hidden_passes: int = 0
    planner_only_traps: int = 0
    requires: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    thought_support: int = 0
    thought_confirmed: int = 0
    thought_wrong: int = 0
    attack_support: int = 0
    attack_failures: int = 0
    attack_breaks: int = 0

    @property
    def confidence(self) -> float:
        if self.support == 0:
            return 0.0
        base = min(1.0, self.support / 4.0)
        survival = self.hidden_passes / max(1, self.support)
        proof_mean = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_mean = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        trap_penalty = self.planner_only_traps / max(1, self.support)
        effect = max(0.0, min(1.5, (proof_mean + hidden_mean) / 50.0))
        calibration = self.thought_confirmed / max(1, self.thought_support) if self.thought_support else 0.5
        wrong_rate = self.thought_wrong / max(1, self.thought_support) if self.thought_support else 0.0
        attack_fail_rate = self.attack_failures / max(1, self.attack_support) if self.attack_support else 0.0
        attack_break_rate = self.attack_breaks / max(1, self.attack_support) if self.attack_support else 0.0
        return max(
            0.0,
            base
            * survival
            * (1.0 - 0.6 * trap_penalty)
            * (0.5 + effect)
            * (0.7 + 0.6 * calibration)
            * (1.0 - 0.5 * wrong_rate)
            * (1.0 - 0.55 * attack_fail_rate)
            * (1.0 - 0.7 * attack_break_rate),
        )

    def summary(self) -> str:
        planner_gain = statistics.mean(self.planner_gain) if self.planner_gain else 0.0
        proof_gain = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_gain = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        return (
            f"motif={self.motif} domain={self.domain_name} mode={self.mode} family={self.root_family} pathology={self.pathology} "
            f"support={self.support} conf={self.confidence:.3f} planner_gain={planner_gain:.3f} "
            f"proof_gain={proof_gain:.3f} hidden_gain={hidden_gain:.3f} traps={self.planner_only_traps} "
            f"thought_confirmed={self.thought_confirmed}/{max(1, self.thought_support)} "
            f"attack_failures={self.attack_failures}/{max(1, self.attack_support)} attack_breaks={self.attack_breaks}"
        )


@dataclass
class MotifNode:
    domain_name: str
    motif: str
    mode: str
    root_family: str
    pathology: str
    support: int = 0
    proof_gain: List[float] = field(default_factory=list)
    hidden_gain: List[float] = field(default_factory=list)
    planner_traps: int = 0
    attack_failures: int = 0
    attack_breaks: int = 0

    @property
    def confidence(self) -> float:
        if self.support == 0:
            return 0.0
        proof_mean = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_mean = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        trap_penalty = self.planner_traps / max(1, self.support)
        attack_fail_rate = self.attack_failures / max(1, self.support)
        attack_break_rate = self.attack_breaks / max(1, self.support)
        effect = max(0.0, min(1.5, (proof_mean + hidden_mean) / 45.0))
        return max(0.0, min(1.0, min(1.0, self.support / 4.0) * (0.45 + effect) * (1.0 - 0.55 * trap_penalty) * (1.0 - 0.55 * attack_fail_rate) * (1.0 - 0.7 * attack_break_rate)))

    def summary(self) -> str:
        proof_mean = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_mean = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        return (
            f"motif={self.motif} domain={self.domain_name} mode={self.mode} family={self.root_family} pathology={self.pathology} "
            f"support={self.support} conf={self.confidence:.3f} proof_gain={proof_mean:.3f} hidden_gain={hidden_mean:.3f} "
            f"traps={self.planner_traps} attack_failures={self.attack_failures}/{max(1, self.support)} attack_breaks={self.attack_breaks}"
        )


@dataclass
class MotifEdge:
    domain_name: str
    mode: str
    root_family: str
    pathology: str
    src: str
    dst: str
    support: int = 0
    proof_gain: List[float] = field(default_factory=list)
    hidden_gain: List[float] = field(default_factory=list)
    planner_traps: int = 0
    attack_failures: int = 0
    attack_breaks: int = 0

    @property
    def confidence(self) -> float:
        if self.support == 0:
            return 0.0
        proof_mean = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_mean = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        trap_penalty = self.planner_traps / max(1, self.support)
        attack_fail_rate = self.attack_failures / max(1, self.support)
        attack_break_rate = self.attack_breaks / max(1, self.support)
        effect = max(0.0, min(1.5, (proof_mean + hidden_mean) / 40.0))
        return max(0.0, min(1.0, min(1.0, self.support / 3.0) * (0.45 + effect) * (1.0 - 0.55 * trap_penalty) * (1.0 - 0.6 * attack_fail_rate) * (1.0 - 0.75 * attack_break_rate)))

    @property
    def motif(self) -> str:
        return f"{self.src}->{self.dst}"

    def summary(self) -> str:
        proof_mean = statistics.mean(self.proof_gain) if self.proof_gain else 0.0
        hidden_mean = statistics.mean(self.hidden_gain) if self.hidden_gain else 0.0
        return (
            f"edge={self.src}->{self.dst} domain={self.domain_name} mode={self.mode} family={self.root_family} pathology={self.pathology} "
            f"support={self.support} conf={self.confidence:.3f} proof_gain={proof_mean:.3f} hidden_gain={hidden_mean:.3f} "
            f"traps={self.planner_traps} attack_failures={self.attack_failures}/{max(1, self.support)} attack_breaks={self.attack_breaks}"
        )


@dataclass
class PatternMemory(Generic[T]):
    cards: Dict[Tuple[str, str, str, str, str], PatternCard] = field(default_factory=dict)
    nodes: Dict[Tuple[str, str, str, str, str], MotifNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str, str, str, str], MotifEdge] = field(default_factory=dict)
    promoted_ops: Dict[Tuple[str, str, str, str], List[DomainOperator[T]]] = field(default_factory=lambda: defaultdict(list))
    promoted_sequences: Dict[Tuple[str, str, str, str], List[Tuple[str, ...]]] = field(default_factory=lambda: defaultdict(list))
    anti_patterns: List[PatternCard] = field(default_factory=list)
    planner_hypotheses: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def record_hypothesis(self, label: str, status: str) -> None:
        self.planner_hypotheses[label][status] += 1

    def relevant_cards(self, domain_name: str, mode: str, root_family: str, pathology: str) -> List[PatternCard]:
        cards = [
            c for key, c in self.cards.items()
            if key[0] == domain_name and key[1] == mode and key[2] == root_family and (key[3] == pathology or key[3] == "any")
        ]
        cards.sort(
            key=lambda c: (
                c.confidence,
                statistics.mean(c.hidden_gain) if c.hidden_gain else 0.0,
                statistics.mean(c.proof_gain) if c.proof_gain else 0.0,
            ),
            reverse=True,
        )
        return cards

    def relevant_nodes(self, domain_name: str, mode: str, root_family: str, pathology: str) -> List[MotifNode]:
        nodes = [
            n for key, n in self.nodes.items()
            if key[0] == domain_name and key[1] == mode and key[2] == root_family and (key[3] == pathology or key[3] == "any")
        ]
        nodes.sort(
            key=lambda n: (
                n.confidence,
                statistics.mean(n.hidden_gain) if n.hidden_gain else 0.0,
                statistics.mean(n.proof_gain) if n.proof_gain else 0.0,
            ),
            reverse=True,
        )
        return nodes

    def relevant_edges(self, domain_name: str, mode: str, root_family: str, pathology: str, src: Optional[str] = None) -> List[MotifEdge]:
        edges = [
            e for key, e in self.edges.items()
            if key[0] == domain_name and key[1] == mode and key[2] == root_family and (key[3] == pathology or key[3] == "any") and (src is None or key[4] == src)
        ]
        edges.sort(
            key=lambda e: (
                e.confidence,
                statistics.mean(e.hidden_gain) if e.hidden_gain else 0.0,
                statistics.mean(e.proof_gain) if e.proof_gain else 0.0,
            ),
            reverse=True,
        )
        return edges

    def relevant_sequences(self, domain_name: str, mode: str, root_family: str, pathology: str) -> List[Tuple[str, ...]]:
        out: List[Tuple[str, ...]] = []
        for key in (
            (domain_name, mode, root_family, pathology),
            (domain_name, mode, root_family, "any"),
        ):
            out.extend(self.promoted_sequences.get(key, []))
        unique: List[Tuple[str, ...]] = []
        seen = set()
        for seq in out:
            if seq in seen:
                continue
            seen.add(seq)
            unique.append(seq)
        return unique

    def recommend_continuations(self, domain_name: str, mode: str, root_family: str, pathology: str, last_ops: Sequence[str]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        if last_ops:
            last_op = last_ops[-1]
            for edge in self.relevant_edges(domain_name, mode, root_family, pathology, last_op)[:8]:
                scores[edge.dst] += 2.8 * edge.confidence
            for seq in self.relevant_sequences(domain_name, mode, root_family, pathology):
                if len(seq) < 2:
                    continue
                if len(last_ops) >= 2 and tuple(last_ops[-2:]) == tuple(seq[:2]) and len(seq) >= 3:
                    scores[seq[2]] += 2.6
                elif last_op == seq[0] and len(seq) >= 2:
                    scores[seq[1]] += 1.9
        else:
            for node in self.relevant_nodes(domain_name, mode, root_family, pathology)[:6]:
                scores[node.motif] += 1.2 * node.confidence
        return dict(scores)

    def relevant_anti_patterns(self, domain_name: str, mode: str, root_family: str, pathology: str) -> List[PatternCard]:
        cards = [
            c for c in self.anti_patterns
            if c.domain_name == domain_name and c.mode == mode and c.root_family == root_family and (c.pathology == pathology or c.pathology == "any")
        ]
        cards.sort(key=lambda c: (c.planner_only_traps + c.attack_breaks, c.confidence), reverse=True)
        return cards

    def relevant_promoted(self, domain_name: str, mode: str, root_family: str, pathology: str) -> List[DomainOperator[T]]:
        out: List[DomainOperator[T]] = []
        for key in (
            (domain_name, mode, root_family, pathology),
            (domain_name, mode, root_family, "any"),
        ):
            out.extend(self.promoted_ops.get(key, []))
        seen = set()
        unique = []
        for op in out:
            if op.name in seen:
                continue
            seen.add(op.name)
            unique.append(op)
        return unique



@dataclass
class SearchNode(Generic[T]):
    sort_key: Tuple[float, float, int]
    candidate: T = field(compare=False)
    path: List[str] = field(compare=False)
    parent_hashes: List[str] = field(compare=False)
    planner: EvalMetrics = field(compare=False)


@dataclass
class ModeResult(Generic[T]):
    mode: str
    root_name: str
    root: T
    champion: T
    planner: EvalMetrics
    proof: EvalMetrics
    hidden: EvalMetrics
    path: List[str]
    trajectory: Trajectory[T]
    reforge_gain: float


@dataclass
class CausalEdit:
    edit: str
    proof_delta: float
    hidden_delta: float
    verdict: str


@dataclass
class InteractionEffect:
    pair: str
    proof_delta: float
    hidden_delta: float
    verdict: str


@dataclass
class ForensicReport:
    domain_name: str
    root_name: str
    mode: str
    root_family: str
    champion_family: str
    path: List[str]
    causal_edits: List[CausalEdit]
    interactions: List[InteractionEffect]
    suite_attribution: List[Tuple[str, float, float]]
    why: str
    reforge_op: Optional[str]
    reforge_gain: float
    rediscovery: str
    planner_thoughts: List[str]


class DomainPack(ABC, Generic[T]):
    name: str

    @abstractmethod
    def root_sets(self) -> Dict[str, Dict[str, T]]:
        raise NotImplementedError

    @abstractmethod
    def candidate_hash(self, candidate: T) -> str:
        raise NotImplementedError

    @abstractmethod
    def summarize(self, candidate: T) -> str:
        raise NotImplementedError

    @abstractmethod
    def complexity(self, candidate: T) -> int:
        raise NotImplementedError

    @abstractmethod
    def classify_family(self, candidate: T) -> str:
        raise NotImplementedError

    @abstractmethod
    def detect_pathology(self, candidate: T, mode: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def structural_edit_distance(self, root: Optional[T], candidate: T) -> int:
        raise NotImplementedError

    @abstractmethod
    def operator_bank(self, mode: str, root: T, promoted: Sequence[DomainOperator[T]]) -> List[DomainOperator[T]]:
        raise NotImplementedError

    @abstractmethod
    def operator_lookup(self) -> Dict[str, DomainOperator[T]]:
        raise NotImplementedError

    @abstractmethod
    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[Any], Arena[Any], Arena[Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, candidate: T, arena: Arena[Any], mode: str, root: Optional[T] = None) -> EvalMetrics:
        raise NotImplementedError

    @abstractmethod
    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def rediscovery_verdict(self, root: T, champion: T) -> str:
        raise NotImplementedError

    @abstractmethod
    def suite_attribution(self, root: T, champion: T, arena: Arena[Any], mode: str) -> List[Tuple[str, float, float]]:
        raise NotImplementedError


                                                              
                     
                                                              

CYCLE_RULES = ("1", "half_n", "n", "two_n", "log2n")
GAP_SCHEDULES = ("none", "fixed2", "n_half_halving")
PARITY_MODES = ("both", "even_only", "odd_only")
CLEANUPS = ("none", "adjacent", "insertion")
FAMILIES = ("bubble", "odd_even", "insertion", "shell", "selection", "cocktail")


def cycle_value(rule: str, n: int) -> int:
    if rule == "1":
        return 1
    if rule == "half_n":
        return max(1, n // 2)
    if rule == "n":
        return max(1, n)
    if rule == "two_n":
        return max(1, 2 * n)
    if rule == "log2n":
        return max(1, math.ceil(math.log2(max(2, n))))
    raise ValueError(rule)


@dataclass(frozen=True)
class SortingCandidate:
    family: str
    cycles_rule: str = "n"
    gap_schedule: str = "none"
    early_stop: bool = False
    break_inner: bool = False
    bidirectional: bool = False
    parity_mode: str = "both"
    cleanup: str = "none"

    def complexity(self) -> int:
        score = 1
        score += 1 if self.family != "bubble" else 0
        score += 1 if self.cycles_rule != "n" else 0
        score += 1 if self.gap_schedule != "none" else 0
        score += 1 if self.early_stop else 0
        score += 1 if self.break_inner else 0
        score += 1 if self.bidirectional else 0
        score += 1 if self.parity_mode != "both" else 0
        score += 1 if self.cleanup != "none" else 0
        return score

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.family,
            self.cycles_rule,
            self.gap_schedule,
            self.early_stop,
            self.break_inner,
            self.bidirectional,
            self.parity_mode,
            self.cleanup,
        )


@dataclass
class CostProfile:
    compares: int = 0
    swaps: int = 0
    branches: int = 0

    def scalar(self) -> float:
        return self.compares + 2.0 * self.swaps + 0.25 * self.branches


@dataclass
class ExecResult:
    output: Optional[List[int]]
    cost: CostProfile
    timeout: bool
    error: Optional[str]


class Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.steps = 0

    def tick(self, amount: int = 1) -> None:
        self.steps += amount
        if self.steps > self.limit:
            raise TimeoutError("budget")


class SortingDomainPack(DomainPack[SortingCandidate]):
    name = "sorting"

    def __init__(self) -> None:
        self._operators = self._build_operators()

    def root_sets(self) -> Dict[str, Dict[str, SortingCandidate]]:
        return {
            "repair": {
                "one_pass_bubble": SortingCandidate("bubble", cycles_rule="1", early_stop=False),
                "even_only": SortingCandidate("odd_even", cycles_rule="n", parity_mode="even_only", early_stop=False),
                "frozen_gap": SortingCandidate("shell", cycles_rule="n", gap_schedule="fixed2", break_inner=False, cleanup="none"),
            },
            "harden": {
                "naive_bubble": SortingCandidate("bubble", cycles_rule="two_n", early_stop=False),
                "overcycled_bubble": SortingCandidate("bubble", cycles_rule="two_n", early_stop=True),
                "odd_even": SortingCandidate("odd_even", cycles_rule="n", parity_mode="both", early_stop=False),
                "selection": SortingCandidate("selection"),
            },
            "optimize": {
                "selection": SortingCandidate("selection"),
                "naive_bubble": SortingCandidate("bubble", cycles_rule="two_n", early_stop=False),
            },
        }

    def candidate_hash(self, candidate: SortingCandidate) -> str:
        return hashlib.sha1(repr(candidate.signature()).encode()).hexdigest()[:12]

    def summarize(self, candidate: SortingCandidate) -> str:
        return (
            f"family={candidate.family}, cycles={candidate.cycles_rule}, gap={candidate.gap_schedule}, "
            f"early_stop={candidate.early_stop}, break_inner={candidate.break_inner}, "
            f"bidirectional={candidate.bidirectional}, parity={candidate.parity_mode}, cleanup={candidate.cleanup}"
        )

    def complexity(self, candidate: SortingCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: SortingCandidate) -> str:
        if candidate.family == "shell" and candidate.gap_schedule == "n_half_halving":
            return "shell_gap"
        if candidate.family == "insertion":
            return "insertion_like"
        if candidate.family == "odd_even" and candidate.parity_mode == "both":
            return "odd_even"
        if candidate.family == "bubble" and candidate.bidirectional:
            return "cocktail_like"
        if candidate.family == "bubble":
            return "bubble_like"
        if candidate.family == "selection":
            return "selection_like"
        return "mixed"

    def detect_pathology(self, candidate: SortingCandidate, mode: str) -> str:
        if candidate.family == "bubble" and candidate.cycles_rule == "two_n":
            return "overcycled_adjacent"
        if candidate.family == "bubble" and candidate.cycles_rule == "1":
            return "underpowered_adjacent"
        if candidate.family == "odd_even" and candidate.parity_mode != "both":
            return "parity_broken"
        if candidate.family == "shell" and candidate.gap_schedule == "fixed2" and not candidate.break_inner:
            return "frozen_gap"
        if candidate.family == "selection":
            return "tail_heavy_selection"
        if self.classify_family(candidate) == "shell_gap":
            return "shell_like"
        return "generic"

    def structural_edit_distance(self, root: Optional[SortingCandidate], candidate: SortingCandidate) -> int:
        if root is None:
            return candidate.complexity()
        return sum(x != y for x, y in zip(root.signature(), candidate.signature()))

    def _build_operators(self) -> Dict[str, DomainOperator[SortingCandidate]]:
        ops = {
            "enable_early_stop": DomainOperator("enable_early_stop", lambda c: replace(c, early_stop=True)),
            "set_cycles_n": DomainOperator("set_cycles_n", lambda c: replace(c, cycles_rule="n")),
            "set_cycles_log2n": DomainOperator("set_cycles_log2n", lambda c: replace(c, cycles_rule="log2n")),
            "set_parity_both": DomainOperator("set_parity_both", lambda c: replace(c, family="odd_even", parity_mode="both", cycles_rule="n", early_stop=True)),
            "rewrite_insertion": DomainOperator("rewrite_insertion", lambda c: SortingCandidate("insertion", break_inner=True)),
            "rewrite_shell": DomainOperator("rewrite_shell", lambda c: SortingCandidate("shell", cycles_rule="log2n", gap_schedule="n_half_halving", break_inner=True, cleanup="insertion")),
            "rewrite_bubble": DomainOperator("rewrite_bubble", lambda c: SortingCandidate("bubble", cycles_rule="n", early_stop=True)),
            "rewrite_cocktail": DomainOperator("rewrite_cocktail", lambda c: SortingCandidate("cocktail", cycles_rule="half_n", early_stop=True, bidirectional=True)),
            "rewrite_selection": DomainOperator("rewrite_selection", lambda c: SortingCandidate("selection", cleanup="insertion")),
            "set_gap_halving": DomainOperator("set_gap_halving", lambda c: replace(c, family="shell", gap_schedule="n_half_halving")),
            "set_gap_fixed2": DomainOperator("set_gap_fixed2", lambda c: replace(c, family="shell", gap_schedule="fixed2")),
            "enable_break_inner": DomainOperator("enable_break_inner", lambda c: replace(c, break_inner=True)),
            "cleanup_insertion": DomainOperator("cleanup_insertion", lambda c: replace(c, cleanup="insertion")),
            "cleanup_adjacent": DomainOperator("cleanup_adjacent", lambda c: replace(c, cleanup="adjacent")),
            "set_bidirectional": DomainOperator("set_bidirectional", self._set_bidirectional),
            "shellify_from_bubble": DomainOperator("shellify_from_bubble", lambda c: SortingCandidate("shell", cycles_rule="log2n", gap_schedule="n_half_halving", break_inner=True, cleanup="adjacent")),
            "insertionize_repair": DomainOperator("insertionize_repair", lambda c: SortingCandidate("insertion", break_inner=True)),
            "harden_shell_tail": DomainOperator("harden_shell_tail", lambda c: SortingCandidate("shell", cycles_rule="log2n", gap_schedule="n_half_halving", break_inner=True, cleanup="insertion")),
            "deovercycle": DomainOperator("deovercycle", self._deovercycle),
        }
        combos = {
            "rewrite_selection+shell_refine": ("rewrite_selection", "rewrite_shell"),
            "set_gap_halving+enable_break_inner": ("set_gap_halving", "enable_break_inner"),
        }
        for name, parts in combos.items():
            ops[name] = DomainOperator(name, self._compose(parts, ops))
        return ops

    def _compose(self, names: Sequence[str], lookup: Dict[str, DomainOperator[SortingCandidate]]) -> Callable[[SortingCandidate], SortingCandidate]:
        def fn(c: SortingCandidate) -> SortingCandidate:
            for name in names:
                c = lookup[name].fn(c)
            return c
        return fn

    def _set_bidirectional(self, c: SortingCandidate) -> SortingCandidate:
        if c.family == "bubble":
            return replace(c, family="cocktail", bidirectional=True, early_stop=True, cycles_rule="half_n")
        return replace(c, bidirectional=True)

    def _deovercycle(self, c: SortingCandidate) -> SortingCandidate:
        if c.family in {"bubble", "cocktail", "odd_even"}:
            return replace(c, cycles_rule="n", early_stop=True)
        return c

    def operator_lookup(self) -> Dict[str, DomainOperator[SortingCandidate]]:
        return self._operators

    def operator_bank(self, mode: str, root: SortingCandidate, promoted: Sequence[DomainOperator[SortingCandidate]]) -> List[DomainOperator[SortingCandidate]]:
        names_by_mode = {
            "repair": {
                "set_parity_both", "rewrite_insertion", "rewrite_shell", "set_gap_halving", "enable_break_inner",
                "cleanup_insertion", "shellify_from_bubble", "insertionize_repair", "deovercycle",
                "enable_early_stop", "set_cycles_n", "set_gap_halving+enable_break_inner",
            },
            "harden": {
                "rewrite_shell", "harden_shell_tail", "cleanup_insertion", "enable_break_inner", "deovercycle",
                "set_cycles_log2n", "set_bidirectional", "rewrite_cocktail", "rewrite_selection",
                "enable_early_stop", "rewrite_selection+shell_refine",
            },
            "optimize": set(self._operators.keys()) - {"insertionize_repair", "harden_shell_tail"},
        }
        allowed = names_by_mode[mode]
        out = []
        seen = set()
        for op in list(promoted) + [self._operators[n] for n in sorted(allowed)]:
            if op.name in seen:
                continue
            seen.add(op.name)
            out.append(op)
        return out

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[List[int]], Arena[List[int]], Arena[List[int]]]:
        if mode == "repair":
            planner = self._build_arena(seed + 1, 24, [0, 1, 2, 3, 4, 5, 8, 12, 16], f"{mode}_planner")
            proof = self._build_arena(seed + 2, 72, [0, 1, 2, 3, 4, 5, 7, 11, 16, 20, 24], f"{mode}_proof")
            hidden = self._build_arena(seed + 3, 80, [0, 1, 2, 3, 4, 5, 9, 13, 19, 27], f"{mode}_hidden")
        elif mode == "harden":
            hard_kinds = ("reversed", "organ", "saw", "few_unique", "extremes", "nearly")
            planner = self._build_arena(seed + 11, 30, [8, 12, 16, 20, 24], f"{mode}_planner", hard_kinds)
            proof = self._build_arena(seed + 12, 96, [8, 12, 16, 24, 32, 40], f"{mode}_proof", hard_kinds)
            hidden = self._build_arena(seed + 13, 112, [10, 14, 18, 26, 34, 42], f"{mode}_hidden", hard_kinds)
        else:
            planner = self._build_arena(seed + 21, 24, [2, 4, 8, 12, 16], f"{mode}_planner")
            proof = self._build_arena(seed + 22, 72, [2, 4, 8, 12, 18, 24], f"{mode}_proof")
            hidden = self._build_arena(seed + 23, 80, [3, 5, 9, 13, 19, 27], f"{mode}_hidden")
        return planner, proof, hidden

    def evaluate(self, candidate: SortingCandidate, arena: Arena[List[int]], mode: str, root: Optional[SortingCandidate] = None) -> EvalMetrics:
        costs: List[float] = []
        failures: Dict[str, int] = defaultdict(int)
        correct = 0
        for _, arr in arena.cases:
            res = self._execute_candidate(candidate, arr)
            if res.output is None:
                failures["timeout" if res.timeout else (res.error or "invalid")] += 1
                continue
            if not self._is_correct_sort(arr, res.output):
                failures["wrong"] += 1
                continue
            correct += 1
            n = max(1, len(arr))
            costs.append(res.cost.scalar() / (n * n))
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = candidate.complexity()
        edit_distance = self.structural_edit_distance(root, candidate) if root else 0.0
        failure_penalty = (len(arena.cases) - correct) * 1000.0
        scalar = failure_penalty
        if mode == "repair":
            scalar += avg_cost * 100.0 + worst_cost * 20.0 + edit_distance * 0.8 + complexity * 0.2
        elif mode == "harden":
            scalar += worst_cost * 120.0 + avg_cost * 40.0 + complexity * 0.3 + edit_distance * 0.15
        else:
            scalar += avg_cost * 110.0 + worst_cost * 45.0 + complexity * 0.25
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, dict(failures))

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        hints = {
            "overcycled_adjacent": {"rewrite_selection": 2.6, "rewrite_shell": 2.4, "harden_shell_tail": 2.8, "deovercycle": 2.2, "enable_early_stop": 1.4},
            "underpowered_adjacent": {"rewrite_shell": 2.6, "set_gap_halving": 2.4, "enable_break_inner": 1.9, "shellify_from_bubble": 2.8, "insertionize_repair": 1.3, "set_cycles_n": 1.5},
            "parity_broken": {"set_parity_both": 2.8, "rewrite_shell": 2.2, "set_gap_halving": 2.0, "enable_break_inner": 1.7},
            "frozen_gap": {"set_gap_halving": 2.7, "enable_break_inner": 2.0, "cleanup_insertion": 1.9, "rewrite_shell": 2.2},
            "tail_heavy_selection": {"cleanup_insertion": 2.0, "rewrite_shell": 2.1, "harden_shell_tail": 2.6, "enable_break_inner": 1.2},
            "shell_like": {"enable_break_inner": 1.4, "cleanup_insertion": 1.1, "set_cycles_log2n": 1.4},
        }
        return hints.get(pathology, {})

    def rediscovery_verdict(self, root: SortingCandidate, champion: SortingCandidate) -> str:
        fam = self.classify_family(champion)
        if fam == "shell_gap":
            return "rediscovered known shell-gap family"
        if fam == "insertion_like":
            return "rediscovered insertion-like family"
        if fam == self.classify_family(root):
            return "refined same family"
        return "family shift without clear novelty proof"

    def suite_attribution(self, root: SortingCandidate, champion: SortingCandidate, arena: Arena[List[int]], mode: str) -> List[Tuple[str, float, float]]:
        buckets: Dict[str, List[float]] = defaultdict(list)
        corr: Dict[str, List[float]] = defaultdict(list)
        for kind, arr in arena.cases:
            root_res = self._execute_candidate(root, arr)
            champ_res = self._execute_candidate(champion, arr)
            n = max(1, len(arr))
            root_cost = float("inf") if root_res.output is None else root_res.cost.scalar() / (n * n)
            champ_cost = float("inf") if champ_res.output is None else champ_res.cost.scalar() / (n * n)
            if math.isfinite(root_cost) and math.isfinite(champ_cost):
                buckets[kind].append(root_cost - champ_cost)
            root_ok = 1.0 if self._is_correct_sort(arr, root_res.output) else 0.0
            champ_ok = 1.0 if self._is_correct_sort(arr, champ_res.output) else 0.0
            corr[kind].append(champ_ok - root_ok)
        out = []
        for kind in buckets:
            out.append((kind, statistics.mean(buckets[kind]), statistics.mean(corr[kind]) if corr[kind] else 0.0))
        out.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return out

    def _is_correct_sort(self, inp: List[int], out: Optional[List[int]]) -> bool:
        if out is None or len(inp) != len(out):
            return False
        if any(out[i] > out[i + 1] for i in range(len(out) - 1)):
            return False
        return out == sorted(inp)

    def _adjacent_pass(self, arr: List[int], cost: CostProfile, budget: Budget, direction: str = "lr", start: int = 0, step: int = 1) -> int:
        swaps = 0
        n = len(arr)
        if direction == "lr":
            indices = range(start, max(0, n - 1), step)
        else:
            last = n - 2
            if step == 2:
                last = last if last % 2 == start % 2 else last - 1
                indices = range(last, -1, -2)
            else:
                indices = range(n - 2, -1, -1)
        for i in indices:
            if i < 0 or i + 1 >= n:
                continue
            budget.tick()
            cost.compares += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                cost.swaps += 1
                swaps += 1
        return swaps

    def _insertion_pass(self, arr: List[int], cost: CostProfile, budget: Budget, break_inner: bool) -> int:
        swaps = 0
        for i in range(1, len(arr)):
            j = i
            while j > 0:
                budget.tick()
                cost.compares += 1
                if arr[j - 1] > arr[j]:
                    arr[j - 1], arr[j] = arr[j], arr[j - 1]
                    cost.swaps += 1
                    swaps += 1
                    j -= 1
                else:
                    cost.branches += 1
                    if break_inner:
                        break
                    j -= 1
        return swaps

    def _selection_pass(self, arr: List[int], cost: CostProfile, budget: Budget) -> int:
        swaps = 0
        n = len(arr)
        for i in range(n):
            m = i
            for j in range(i + 1, n):
                budget.tick()
                cost.compares += 1
                if arr[j] < arr[m]:
                    m = j
            if m != i:
                arr[i], arr[m] = arr[m], arr[i]
                cost.swaps += 1
                swaps += 1
        return swaps

    def _shell_pass(self, arr: List[int], gap: int, cost: CostProfile, budget: Budget, break_inner: bool) -> int:
        swaps = 0
        n = len(arr)
        for i in range(gap, n):
            j = i
            while j >= gap:
                budget.tick()
                cost.compares += 1
                if arr[j - gap] > arr[j]:
                    arr[j - gap], arr[j] = arr[j], arr[j - gap]
                    cost.swaps += 1
                    swaps += 1
                    j -= gap
                else:
                    cost.branches += 1
                    if break_inner:
                        break
                    j -= gap
        return swaps

    def _execute_candidate(self, c: SortingCandidate, data: List[int]) -> ExecResult:
        arr = list(data)
        cost = CostProfile()
        budget = Budget(max(128, 30 * max(1, len(arr)) * max(1, len(arr)) + 1000))
        try:
            n = len(arr)
            if c.family == "bubble":
                cycles = cycle_value(c.cycles_rule, n)
                for _ in range(cycles):
                    swaps = self._adjacent_pass(arr, cost, budget, "lr")
                    if c.bidirectional:
                        swaps += self._adjacent_pass(arr, cost, budget, "rl")
                    if c.early_stop:
                        cost.branches += 1
                        if swaps == 0:
                            break
            elif c.family == "cocktail":
                cycles = cycle_value(c.cycles_rule, n)
                for _ in range(cycles):
                    swaps = self._adjacent_pass(arr, cost, budget, "lr")
                    swaps += self._adjacent_pass(arr, cost, budget, "rl")
                    if c.early_stop:
                        cost.branches += 1
                        if swaps == 0:
                            break
            elif c.family == "odd_even":
                cycles = cycle_value(c.cycles_rule, n)
                modes = {"both": ((0, "lr"), (1, "lr")), "even_only": ((0, "lr"),), "odd_only": ((1, "lr"),)}[c.parity_mode]
                for _ in range(cycles):
                    swaps = 0
                    for start, direction in modes:
                        swaps += self._adjacent_pass(arr, cost, budget, direction, start=start, step=2)
                    if c.early_stop:
                        cost.branches += 1
                        if swaps == 0:
                            break
            elif c.family == "insertion":
                self._insertion_pass(arr, cost, budget, break_inner=c.break_inner)
            elif c.family == "selection":
                self._selection_pass(arr, cost, budget)
                if c.cleanup == "adjacent":
                    self._adjacent_pass(arr, cost, budget, "lr")
                elif c.cleanup == "insertion":
                    self._insertion_pass(arr, cost, budget, break_inner=True)
            elif c.family == "shell":
                if c.gap_schedule == "fixed2":
                    self._shell_pass(arr, 2, cost, budget, break_inner=c.break_inner)
                else:
                    gap = max(1, n // 2) if c.gap_schedule == "n_half_halving" else 1
                    while gap >= 1:
                        self._shell_pass(arr, gap, cost, budget, break_inner=c.break_inner)
                        cost.branches += 1
                        if gap == 1:
                            break
                        gap = max(1, gap // 2)
                if c.cleanup == "adjacent":
                    self._adjacent_pass(arr, cost, budget, "lr")
                elif c.cleanup == "insertion":
                    self._insertion_pass(arr, cost, budget, break_inner=True)
            else:
                return ExecResult(None, cost, False, "unknown_family")
            return ExecResult(arr, cost, False, None)
        except TimeoutError:
            return ExecResult(None, cost, True, None)

    def _make_case(self, rng: random.Random, n: int, kind: str) -> List[int]:
        if kind == "rand_wide":
            return [rng.randint(-10000, 10000) for _ in range(n)]
        if kind == "rand_small":
            return [rng.randint(-5, 5) for _ in range(n)]
        if kind == "reversed":
            arr = [rng.randint(-100, 100) for _ in range(n)]
            arr.sort(reverse=True)
            return arr
        if kind == "sorted":
            arr = [rng.randint(-100, 100) for _ in range(n)]
            arr.sort()
            return arr
        if kind == "nearly":
            arr = list(range(n))
            for _ in range(max(1, n // 8)):
                if n:
                    a, b = rng.randrange(n), rng.randrange(n)
                    arr[a], arr[b] = arr[b], arr[a]
            return arr
        if kind == "organ":
            return list(range((n + 1) // 2)) + list(range(n // 2))[::-1]
        if kind == "saw":
            mod = max(2, min(7, n // 2 + 1))
            return [i % mod for i in range(n)][::-1]
        if kind == "few_unique":
            vals = [rng.randint(-3, 3) for _ in range(3)]
            return [vals[rng.randrange(len(vals))] for _ in range(n)]
        if kind == "all_equal":
            v = rng.randint(-9, 9)
            return [v] * n
        if kind == "extremes":
            arr = [(-10**6 if i % 2 == 0 else 10**6) + rng.randint(-3, 3) for i in range(n)]
            rng.shuffle(arr)
            return arr
        raise ValueError(kind)

    def _build_arena(self, seed: int, count: int, sizes: Sequence[int], name: str, kinds: Optional[Sequence[str]] = None) -> Arena[List[int]]:
        rng = random.Random(seed)
        kinds = list(kinds or ("rand_wide", "rand_small", "reversed", "sorted", "nearly", "organ", "saw", "few_unique", "all_equal", "extremes"))
        cases = []
        for _ in range(count):
            kind = rng.choice(kinds)
            n = rng.choice(list(sizes))
            cases.append((kind, self._make_case(rng, n, kind)))
        return Arena(name, cases)


@dataclass(frozen=True)
class GridCase:
    grid: Tuple[str, ...]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    optimal_len: Optional[int]
    has_path: bool

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def open_cells(self) -> int:
        return sum(ch == "." for row in self.grid for ch in row)


@dataclass(frozen=True)
class PathfindingCandidate:
    family: str
    heuristic: str = "manhattan"
    weight: int = 1
    tie_break: str = "fifo"
    reopen: bool = False
    frontier_cap: int = 0
    budget_rule: str = "normal"
    fallback_bfs: bool = False
    neighbor_order: str = "standard"

    def complexity(self) -> int:
        score = 1
        score += 1 if self.family != "bfs" else 0
        score += 1 if self.heuristic != "manhattan" else 0
        score += 1 if self.weight != 1 else 0
        score += 1 if self.tie_break != "fifo" else 0
        score += 1 if self.reopen else 0
        score += 1 if self.frontier_cap else 0
        score += 1 if self.budget_rule != "normal" else 0
        score += 1 if self.fallback_bfs else 0
        score += 1 if self.neighbor_order != "standard" else 0
        return score

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.family,
            self.heuristic,
            self.weight,
            self.tie_break,
            self.reopen,
            self.frontier_cap,
            self.budget_rule,
            self.fallback_bfs,
            self.neighbor_order,
        )


@dataclass
class PathExecResult:
    found: bool
    path_len: Optional[int]
    expansions: int
    timeout: bool
    used_fallback: bool = False


class PathfindingDomainPack(DomainPack[PathfindingCandidate]):
    name = "pathfinding"

    def __init__(self) -> None:
        self._operators = self._build_operators()

    def root_sets(self) -> Dict[str, Dict[str, PathfindingCandidate]]:
        return {
            "repair": {
                "greedy_tight": PathfindingCandidate("greedy", heuristic="zero", weight=3, frontier_cap=20, budget_rule="tight"),
                "bfs_capped": PathfindingCandidate("bfs", frontier_cap=14, budget_rule="normal"),
                "weighted_blind": PathfindingCandidate("weighted_astar", heuristic="zero", weight=4, frontier_cap=28, budget_rule="tight"),
            },
            "harden": {
                "bfs_plain": PathfindingCandidate("bfs", budget_rule="normal"),
                "greedy_fragile": PathfindingCandidate("greedy", heuristic="manhattan", frontier_cap=32, budget_rule="normal"),
                "weighted_unstable": PathfindingCandidate("weighted_astar", heuristic="manhattan", weight=4, budget_rule="normal"),
            },
            "optimize": {
                "bfs_plain": PathfindingCandidate("bfs", budget_rule="normal"),
                "dijkstra_plain": PathfindingCandidate("dijkstra", heuristic="zero", budget_rule="normal"),
            },
        }

    def candidate_hash(self, candidate: PathfindingCandidate) -> str:
        return hashlib.sha1(repr(candidate.signature()).encode()).hexdigest()[:12]

    def summarize(self, candidate: PathfindingCandidate) -> str:
        return (
            f"family={candidate.family}, heuristic={candidate.heuristic}, weight={candidate.weight}, "
            f"tie_break={candidate.tie_break}, reopen={candidate.reopen}, frontier_cap={candidate.frontier_cap}, "
            f"budget={candidate.budget_rule}, fallback_bfs={candidate.fallback_bfs}, neighbor_order={candidate.neighbor_order}"
        )

    def complexity(self, candidate: PathfindingCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: PathfindingCandidate) -> str:
        if candidate.family == "weighted_astar" and candidate.weight >= 3:
            return "weighted_astar"
        if candidate.family == "astar":
            return "astar_like"
        if candidate.family == "greedy":
            return "greedy_like"
        if candidate.family == "dijkstra":
            return "dijkstra_like"
        if candidate.family == "bfs":
            return "bfs_like"
        return "mixed"

    def detect_pathology(self, candidate: PathfindingCandidate, mode: str) -> str:
        if candidate.frontier_cap:
            return "capped_frontier"
        if candidate.budget_rule == "tight":
            return "tight_budget"
        if candidate.family == "greedy" and not candidate.fallback_bfs:
            return "fragile_greedy"
        if candidate.family == "weighted_astar" and candidate.weight >= 3:
            return "overweighted"
        if candidate.heuristic == "zero" and candidate.family in {"greedy", "astar", "weighted_astar"}:
            return "blind_heuristic"
        if candidate.family == "bfs":
            return "tail_heavy_bfs"
        if candidate.family == "astar" and candidate.fallback_bfs:
            return "robust_astar"
        return "generic"

    def structural_edit_distance(self, root: Optional[PathfindingCandidate], candidate: PathfindingCandidate) -> int:
        if root is None:
            return candidate.complexity()
        return sum(x != y for x, y in zip(root.signature(), candidate.signature()))

    def _build_operators(self) -> Dict[str, DomainOperator[PathfindingCandidate]]:
        ops = {
            "rewrite_bfs": DomainOperator("rewrite_bfs", lambda c: replace(c, family="bfs", heuristic="zero", weight=1)),
            "rewrite_dijkstra": DomainOperator("rewrite_dijkstra", lambda c: replace(c, family="dijkstra", heuristic="zero", weight=1)),
            "rewrite_astar": DomainOperator("rewrite_astar", lambda c: replace(c, family="astar", heuristic="manhattan", weight=1)),
            "rewrite_weighted": DomainOperator("rewrite_weighted", lambda c: replace(c, family="weighted_astar", heuristic="manhattan", weight=max(2, c.weight or 2))),
            "rewrite_greedy": DomainOperator("rewrite_greedy", lambda c: replace(c, family="greedy", heuristic="manhattan", weight=1)),
            "heuristic_zero": DomainOperator("heuristic_zero", lambda c: replace(c, heuristic="zero")),
            "heuristic_manhattan": DomainOperator("heuristic_manhattan", lambda c: replace(c, heuristic="manhattan")),
            "heuristic_euclidean": DomainOperator("heuristic_euclidean", lambda c: replace(c, heuristic="euclidean")),
            "weight_1": DomainOperator("weight_1", lambda c: replace(c, weight=1, family="astar" if c.family == "weighted_astar" else c.family)),
            "weight_2": DomainOperator("weight_2", lambda c: replace(c, weight=2, family="weighted_astar")),
            "weight_3": DomainOperator("weight_3", lambda c: replace(c, weight=3, family="weighted_astar")),
            "weight_4": DomainOperator("weight_4", lambda c: replace(c, weight=4, family="weighted_astar")),
            "tie_fifo": DomainOperator("tie_fifo", lambda c: replace(c, tie_break="fifo")),
            "tie_low_h": DomainOperator("tie_low_h", lambda c: replace(c, tie_break="low_h")),
            "tie_high_g": DomainOperator("tie_high_g", lambda c: replace(c, tie_break="high_g")),
            "enable_reopen": DomainOperator("enable_reopen", lambda c: replace(c, reopen=True)),
            "disable_reopen": DomainOperator("disable_reopen", lambda c: replace(c, reopen=False)),
            "remove_frontier_cap": DomainOperator("remove_frontier_cap", lambda c: replace(c, frontier_cap=0)),
            "cap_24": DomainOperator("cap_24", lambda c: replace(c, frontier_cap=24)),
            "normal_budget": DomainOperator("normal_budget", lambda c: replace(c, budget_rule="normal")),
            "loose_budget": DomainOperator("loose_budget", lambda c: replace(c, budget_rule="loose")),
            "enable_fallback_bfs": DomainOperator("enable_fallback_bfs", lambda c: replace(c, fallback_bfs=True)),
            "disable_fallback_bfs": DomainOperator("disable_fallback_bfs", lambda c: replace(c, fallback_bfs=False)),
            "neighbor_goal_bias": DomainOperator("neighbor_goal_bias", lambda c: replace(c, neighbor_order="goal_bias")),
            "neighbor_reverse": DomainOperator("neighbor_reverse", lambda c: replace(c, neighbor_order="reverse")),
            "neighbor_standard": DomainOperator("neighbor_standard", lambda c: replace(c, neighbor_order="standard")),
        }
        combos = {
            "robust_astar": ("rewrite_astar", "heuristic_manhattan", "tie_high_g", "remove_frontier_cap", "loose_budget", "enable_fallback_bfs"),
            "repair_safe": ("rewrite_astar", "heuristic_manhattan", "remove_frontier_cap", "normal_budget", "enable_fallback_bfs"),
            "weighted_trim": ("rewrite_weighted", "heuristic_manhattan", "weight_2", "tie_low_h", "remove_frontier_cap"),
        }
        for name, parts in combos.items():
            ops[name] = DomainOperator(name, self._compose(parts, ops))
        return ops

    def _compose(self, names: Sequence[str], lookup: Dict[str, DomainOperator[PathfindingCandidate]]) -> Callable[[PathfindingCandidate], PathfindingCandidate]:
        def fn(c: PathfindingCandidate) -> PathfindingCandidate:
            for name in names:
                c = lookup[name].fn(c)
            return c
        return fn

    def operator_lookup(self) -> Dict[str, DomainOperator[PathfindingCandidate]]:
        return self._operators

    def operator_bank(self, mode: str, root: PathfindingCandidate, promoted: Sequence[DomainOperator[PathfindingCandidate]]) -> List[DomainOperator[PathfindingCandidate]]:
        names_by_mode = {
            "repair": {
                "rewrite_astar", "rewrite_bfs", "repair_safe", "robust_astar", "heuristic_manhattan",
                "remove_frontier_cap", "normal_budget", "loose_budget", "enable_fallback_bfs", "tie_high_g",
                "enable_reopen", "neighbor_goal_bias", "weighted_trim",
            },
            "harden": {
                "robust_astar", "weighted_trim", "rewrite_dijkstra", "rewrite_astar", "heuristic_manhattan",
                "tie_high_g", "remove_frontier_cap", "enable_fallback_bfs", "normal_budget", "loose_budget",
                "enable_reopen", "neighbor_goal_bias",
            },
            "optimize": set(self._operators.keys()) - {"cap_24", "disable_fallback_bfs"},
        }
        allowed = names_by_mode[mode]
        out: List[DomainOperator[PathfindingCandidate]] = []
        seen = set()
        for op in list(promoted) + [self._operators[n] for n in sorted(allowed)]:
            if op.name in seen:
                continue
            seen.add(op.name)
            out.append(op)
        return out

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[GridCase], Arena[GridCase], Arena[GridCase]]:
        if mode == "repair":
            planner = self._build_arena(seed + 1, 18, [7, 9, 11], f"{mode}_planner", ("open", "medium", "corridor", "blocked"))
            proof = self._build_arena(seed + 2, 48, [7, 9, 11, 13], f"{mode}_proof", ("open", "medium", "corridor", "blocked", "dense"))
            hidden = self._build_arena(seed + 3, 56, [7, 9, 11, 13], f"{mode}_hidden", ("open", "medium", "corridor", "blocked", "dense", "mazeish"))
        elif mode == "harden":
            planner = self._build_arena(seed + 11, 20, [9, 11, 13], f"{mode}_planner", ("medium", "corridor", "dense", "mazeish", "blocked"))
            proof = self._build_arena(seed + 12, 54, [9, 11, 13, 15], f"{mode}_proof", ("medium", "corridor", "dense", "mazeish", "blocked"))
            hidden = self._build_arena(seed + 13, 60, [9, 11, 13, 15], f"{mode}_hidden", ("medium", "corridor", "dense", "mazeish", "blocked"))
        else:
            planner = self._build_arena(seed + 21, 18, [7, 9, 11], f"{mode}_planner", ("open", "medium", "corridor"))
            proof = self._build_arena(seed + 22, 50, [7, 9, 11, 13], f"{mode}_proof", ("open", "medium", "corridor", "dense"))
            hidden = self._build_arena(seed + 23, 56, [7, 9, 11, 13], f"{mode}_hidden", ("open", "medium", "corridor", "dense", "mazeish"))
        return planner, proof, hidden

    def evaluate(self, candidate: PathfindingCandidate, arena: Arena[GridCase], mode: str, root: Optional[PathfindingCandidate] = None) -> EvalMetrics:
        costs: List[float] = []
        failures: Dict[str, int] = defaultdict(int)
        correct = 0
        for _, case in arena.cases:
            res = self._execute_candidate(candidate, case)
            norm = max(1, case.open_cells)
            base_cost = res.expansions / norm
            if case.has_path:
                if not res.found or res.path_len is None:
                    failures["miss"] += 1
                    continue
                path_penalty = max(0.0, (res.path_len - (case.optimal_len or res.path_len)) / max(1, case.optimal_len or res.path_len))
                costs.append(base_cost + 0.8 * path_penalty)
                correct += 1
            else:
                if res.found:
                    failures["false_path"] += 1
                    continue
                costs.append(base_cost)
                correct += 1
            if res.timeout:
                failures["timeout"] += 1
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = candidate.complexity()
        edit_distance = self.structural_edit_distance(root, candidate) if root else 0.0
        failure_penalty = (len(arena.cases) - correct) * 1000.0
        scalar = failure_penalty
        if mode == "repair":
            scalar += avg_cost * 115.0 + worst_cost * 35.0 + edit_distance * 0.8 + complexity * 0.25
        elif mode == "harden":
            scalar += worst_cost * 130.0 + avg_cost * 38.0 + edit_distance * 0.2 + complexity * 0.35
        else:
            scalar += avg_cost * 90.0 + worst_cost * 55.0 + complexity * 0.3
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, dict(failures))

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        hints = {
            "capped_frontier": {"remove_frontier_cap": 2.8, "repair_safe": 2.2, "robust_astar": 2.0},
            "tight_budget": {"normal_budget": 2.3, "loose_budget": 2.6, "enable_fallback_bfs": 1.7},
            "fragile_greedy": {"rewrite_astar": 2.8, "repair_safe": 2.5, "robust_astar": 2.2, "enable_fallback_bfs": 1.8},
            "overweighted": {"weighted_trim": 2.7, "weight_2": 2.0, "rewrite_astar": 1.8},
            "blind_heuristic": {"heuristic_manhattan": 2.6, "repair_safe": 2.1, "robust_astar": 2.0},
            "tail_heavy_bfs": {"rewrite_astar": 2.0, "robust_astar": 2.4, "rewrite_dijkstra": 1.5},
            "robust_astar": {"tie_high_g": 1.4, "neighbor_goal_bias": 1.0},
        }
        return hints.get(pathology, {})

    def rediscovery_verdict(self, root: PathfindingCandidate, champion: PathfindingCandidate) -> str:
        fam = self.classify_family(champion)
        if fam == "astar_like":
            return "rediscovered strong A*-like family"
        if fam == "weighted_astar":
            return "rediscovered weighted A*-like family"
        if fam == self.classify_family(root):
            return "refined same family"
        return "family shift without clear novelty proof"

    def suite_attribution(self, root: PathfindingCandidate, champion: PathfindingCandidate, arena: Arena[GridCase], mode: str) -> List[Tuple[str, float, float]]:
        buckets: Dict[str, List[float]] = defaultdict(list)
        corr: Dict[str, List[float]] = defaultdict(list)
        for kind, case in arena.cases:
            root_res = self._execute_candidate(root, case)
            champ_res = self._execute_candidate(champion, case)
            norm = max(1, case.open_cells)
            root_cost = root_res.expansions / norm
            champ_cost = champ_res.expansions / norm
            if case.has_path and root_res.path_len is not None:
                root_cost += 0.8 * max(0.0, (root_res.path_len - (case.optimal_len or root_res.path_len)) / max(1, case.optimal_len or root_res.path_len))
            if case.has_path and champ_res.path_len is not None:
                champ_cost += 0.8 * max(0.0, (champ_res.path_len - (case.optimal_len or champ_res.path_len)) / max(1, case.optimal_len or champ_res.path_len))
            buckets[kind].append(root_cost - champ_cost)
            root_ok = 1.0 if ((not case.has_path and not root_res.found) or (case.has_path and root_res.found and root_res.path_len is not None)) else 0.0
            champ_ok = 1.0 if ((not case.has_path and not champ_res.found) or (case.has_path and champ_res.found and champ_res.path_len is not None)) else 0.0
            corr[kind].append(champ_ok - root_ok)
        out = []
        for kind in buckets:
            out.append((kind, statistics.mean(buckets[kind]), statistics.mean(corr[kind]) if corr[kind] else 0.0))
        out.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return out

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], kind: str) -> float:
        dr = abs(pos[0] - goal[0])
        dc = abs(pos[1] - goal[1])
        if kind == "zero":
            return 0.0
        if kind == "euclidean":
            return math.sqrt(dr * dr + dc * dc)
        return float(dr + dc)

    def _neighbors(self, case: GridCase, pos: Tuple[int, int], order: str) -> List[Tuple[int, int]]:
        r, c = pos
        neigh = [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]
        valid = []
        for nr, nc in neigh:
            if 0 <= nr < case.rows and 0 <= nc < case.cols and case.grid[nr][nc] == ".":
                valid.append((nr, nc))
        if order == "reverse":
            valid.reverse()
        elif order == "goal_bias":
            valid.sort(key=lambda p: self._heuristic(p, case.goal, "manhattan"))
        return valid

    def _budget_limit(self, case: GridCase, rule: str) -> int:
        base = max(16, case.open_cells)
        if rule == "tight":
            return int(base * 1.25)
        if rule == "loose":
            return int(base * 6.0)
        return int(base * 3.0)

    def _priority(self, cand: PathfindingCandidate, g: int, h: float) -> float:
        if cand.family == "bfs":
            return float(g)
        if cand.family == "dijkstra":
            return float(g)
        if cand.family == "greedy":
            return h
        if cand.family == "weighted_astar":
            return float(g) + cand.weight * h
        return float(g) + h

    def _tie_value(self, cand: PathfindingCandidate, g: int, h: float, seq: int) -> Tuple[float, int]:
        if cand.tie_break == "low_h":
            return (h, seq)
        if cand.tie_break == "high_g":
            return (-g, seq)
        return (0.0, seq)

    def _search_once(self, cand: PathfindingCandidate, case: GridCase) -> PathExecResult:
        start, goal = case.start, case.goal
        if start == goal:
            return PathExecResult(True, 0, 0, False)
        limit = self._budget_limit(case, cand.budget_rule)
        expansions = 0
        best_g: Dict[Tuple[int, int], int] = {start: 0}
        if cand.family == "bfs":
            frontier = deque([(start, 0)])
            while frontier:
                pos, g = frontier.popleft()
                expansions += 1
                if expansions > limit:
                    return PathExecResult(False, None, expansions, True)
                if pos == goal:
                    return PathExecResult(True, g, expansions, False)
                for nb in self._neighbors(case, pos, cand.neighbor_order):
                    if nb in best_g and not cand.reopen:
                        continue
                    ng = g + 1
                    if nb not in best_g or ng < best_g[nb]:
                        if cand.frontier_cap and len(frontier) >= cand.frontier_cap:
                            continue
                        best_g[nb] = ng
                        frontier.append((nb, ng))
            return PathExecResult(False, None, expansions, False)
        frontier: List[Tuple[float, float, int, Tuple[int, int], int]] = []
        seq = 0
        h0 = self._heuristic(start, goal, cand.heuristic)
        heapq.heappush(frontier, (self._priority(cand, 0, h0), *self._tie_value(cand, 0, h0, seq), start, 0))
        while frontier:
            _, _, _, pos, g = heapq.heappop(frontier)
            expansions += 1
            if expansions > limit:
                return PathExecResult(False, None, expansions, True)
            if pos == goal:
                return PathExecResult(True, g, expansions, False)
            current_best = best_g.get(pos)
            if current_best is not None and g > current_best:
                continue
            for nb in self._neighbors(case, pos, cand.neighbor_order):
                ng = g + 1
                prev = best_g.get(nb)
                if prev is not None and ng >= prev:
                    continue
                if cand.frontier_cap and len(frontier) >= cand.frontier_cap:
                    continue
                best_g[nb] = ng
                h = self._heuristic(nb, goal, cand.heuristic)
                seq += 1
                heapq.heappush(frontier, (self._priority(cand, ng, h), *self._tie_value(cand, ng, h, seq), nb, ng))
        return PathExecResult(False, None, expansions, False)

    def _execute_candidate(self, cand: PathfindingCandidate, case: GridCase) -> PathExecResult:
        primary = self._search_once(cand, case)
        if primary.found or not cand.fallback_bfs:
            return primary
        fallback = PathfindingCandidate("bfs", heuristic="zero", weight=1, tie_break="fifo", reopen=False, frontier_cap=0, budget_rule="loose", fallback_bfs=False, neighbor_order=cand.neighbor_order)
        fb = self._search_once(fallback, case)
        return PathExecResult(fb.found, fb.path_len, primary.expansions + fb.expansions, primary.timeout or fb.timeout, True)

    def _reference_shortest(self, grid: Tuple[str, ...], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
        q = deque([(start, 0)])
        seen = {start}
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        while q:
            (r, c), d = q.popleft()
            if (r, c) == goal:
                return d
            for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "." and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), d + 1))
        return None

    def _make_case(self, rng: random.Random, n: int, kind: str) -> GridCase:
        start = (0, 0)
        goal = (n - 1, n - 1)
        grid = [["." for _ in range(n)] for _ in range(n)]
        if kind == "blocked":
            wall = n // 2
            for c in range(n):
                grid[wall][c] = "#"
        else:
            density = {"open": 0.08, "medium": 0.18, "dense": 0.28, "mazeish": 0.22, "corridor": 0.14}.get(kind, 0.18)
            for r in range(n):
                for c in range(n):
                    if (r, c) in {start, goal}:
                        continue
                    if rng.random() < density:
                        grid[r][c] = "#"
            if kind == "corridor":
                for r in range(n):
                    x = min(n - 1, (2 * r) % n)
                    grid[r][x] = "."
                    if r + 1 < n:
                        grid[r + 1][x] = "."
            elif kind == "mazeish":
                for r in range(2, n - 1, 2):
                    for c in range(n):
                        grid[r][c] = "#"
                    grid[r][rng.randrange(n)] = "."
        grid[start[0]][start[1]] = "."
        grid[goal[0]][goal[1]] = "."
        frozen = tuple("".join(row) for row in grid)
        optimal = self._reference_shortest(frozen, start, goal)
        return GridCase(frozen, start, goal, optimal, optimal is not None)

    def _build_arena(self, seed: int, count: int, sizes: Sequence[int], name: str, kinds: Sequence[str]) -> Arena[GridCase]:
        rng = random.Random(seed)
        cases: List[Tuple[str, GridCase]] = []
        for _ in range(count):
            kind = rng.choice(list(kinds))
            n = rng.choice(list(sizes))
            cases.append((kind, self._make_case(rng, n, kind)))
        return Arena(name, cases)



                                                              
                      
                                                              


@dataclass
class EngineConfig:
    max_depth: int = 4
    base_beam_width: int = 8
    meta_rounds: int = 2
    base_seed: int = 19
    stagnation_patience: int = 2
    hidden_probe_topk: int = 6
    hidden_probe_cases: int = 18



class ForgeEngine(Generic[T]):
    def __init__(self, domain: DomainPack[T], config: Optional[EngineConfig] = None):
        self.domain = domain
        self.config = config or EngineConfig()
        self.eval_cache: Dict[Tuple[str, str, str], EvalMetrics] = {}

    def _eval_on_arena(self, candidate: T, arena: Arena[Any], mode: str, root: Optional[T]) -> EvalMetrics:
        key = (self.domain.candidate_hash(candidate), arena.name, mode)
        if key not in self.eval_cache:
            self.eval_cache[key] = self.domain.evaluate(candidate, arena, mode, root)
        return self.eval_cache[key]

    def _planner_key(self, metrics: EvalMetrics, candidate: T, path_len: int) -> Tuple[float, float, int]:
        return (metrics.scalar, metrics.worst_cost, path_len + self.domain.complexity(candidate))

    def _op_style(self, op: DomainOperator[T]) -> str:
        tags = set(op.tags)
        name = op.name.lower()
        if {"rewrite", "family_shift", "major"} & tags:
            return "rewrite"
        if {"local", "tune", "cleanup", "toggle"} & tags:
            return "local"
        if "rewrite" in name or "shell" in name or "selection" in name or "astar" in name or "dijkstra" in name or "bfs" in name:
            return "rewrite"
        if name.startswith(("set_", "enable_", "cleanup_", "tie_", "neighbor_", "deovercycle", "weight_", "cap_")):
            return "local"
        return "generic"

    def _family_hypotheses(self, beam: Sequence[SearchNode[T]]) -> List[Tuple[str, float]]:
        counts: Dict[str, int] = defaultdict(int)
        for node in beam:
            counts[self.domain.classify_family(node.candidate)] += 1
        total = max(1, len(beam))
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [(name, count / total) for name, count in ranked]

    def _stagnation_steps(self, best_history: Sequence[float]) -> int:
        if len(best_history) < 2:
            return 0
        steps = 0
        for idx in range(len(best_history) - 1, 0, -1):
            prev = best_history[idx - 1]
            cur = best_history[idx]
            improvement = prev - cur
            threshold = max(0.05, abs(prev) * 0.005)
            if improvement > threshold:
                break
            steps += 1
        return steps

    def _probe_arena(self, hidden: Arena[Any]) -> Arena[Any]:
        count = min(len(hidden.cases), self.config.hidden_probe_cases)
        if count <= 0:
            return hidden
        step = max(1, len(hidden.cases) // max(1, count))
        cases = hidden.cases[::step][:count]
        return Arena(f"{hidden.name}:probe", list(cases))

    def _probe_candidates(self, candidates: Sequence[SearchNode[T]], probe_arena: Arena[Any], mode: str, root: T, limit: int) -> List[SearchNode[T]]:
        reranked: List[Tuple[Tuple[float, float, int], SearchNode[T]]] = []
        for idx, node in enumerate(candidates):
            if idx < limit:
                probe = self._eval_on_arena(node.candidate, probe_arena, mode, root)
                combined = 0.62 * node.planner.scalar + 0.38 * probe.scalar
                reranked.append(((combined, probe.worst_cost, len(node.path) + self.domain.complexity(node.candidate)), node))
            else:
                reranked.append(((node.planner.scalar, node.planner.worst_cost, len(node.path) + self.domain.complexity(node.candidate)), node))
        reranked.sort(key=lambda item: item[0])
        return [node for _, node in reranked]

    def _expand_plan(self, thought: PlannerThought, beam: Sequence[SearchNode[T]], ordered_ops: Sequence[DomainOperator[T]]) -> Tuple[List[SearchNode[T]], List[DomainOperator[T]]]:
        if thought.strategy == "reforge":
            node_count = min(len(beam), 2)
            op_count = min(len(ordered_ops), 4)
        elif thought.strategy == "exploit":
            node_count = max(2, math.ceil(len(beam) * 0.6))
            op_count = min(len(ordered_ops), max(5, thought.beam_width))
        elif thought.strategy == "hedge":
            node_count = len(beam)
            op_count = min(len(ordered_ops), max(8, thought.beam_width + 2))
        elif thought.strategy == "explore":
            node_count = len(beam)
            op_count = len(ordered_ops)
        else:
            node_count = len(beam)
            op_count = min(len(ordered_ops), max(6, thought.beam_width + 1))
        return list(beam[:node_count]), list(ordered_ops[:op_count])

    def _select_candidates(self, candidates: Sequence[SearchNode[T]], thought: PlannerThought) -> List[SearchNode[T]]:
        selected: List[SearchNode[T]] = []
        seen = set()
        family_counts: Dict[str, int] = defaultdict(int)
        family_targets: Dict[str, int] = {}
        if thought.strategy in {"hedge", "explore"}:
            for fam, prob in thought.family_hypotheses[: min(3, len(thought.family_hypotheses))]:
                target = 1 if thought.strategy == "hedge" else max(1, round(prob * thought.beam_width))
                family_targets[fam] = max(1, target)
        family_cap = max(1, thought.beam_width // (2 if thought.strategy in {"hedge", "explore"} else 1))
        for fam, target in family_targets.items():
            for node in candidates:
                sig = self.domain.candidate_hash(node.candidate)
                if sig in seen or self.domain.classify_family(node.candidate) != fam:
                    continue
                selected.append(node)
                seen.add(sig)
                family_counts[fam] += 1
                if family_counts[fam] >= target or len(selected) >= thought.beam_width:
                    break
        for node in candidates:
            if len(selected) >= thought.beam_width:
                break
            sig = self.domain.candidate_hash(node.candidate)
            if sig in seen:
                continue
            fam = self.domain.classify_family(node.candidate)
            if family_counts[fam] >= family_cap and thought.strategy not in {"exploit", "reforge"}:
                continue
            selected.append(node)
            seen.add(sig)
            family_counts[fam] += 1
        if len(selected) < min(thought.beam_width, len(candidates)):
            for node in candidates:
                if len(selected) >= thought.beam_width:
                    break
                sig = self.domain.candidate_hash(node.candidate)
                if sig in seen:
                    continue
                selected.append(node)
                seen.add(sig)
        return selected[: thought.beam_width]

    def _make_planner_thought(
        self,
        depth: int,
        beam: Sequence[SearchNode[T]],
        mode: str,
        root: T,
        memory: PatternMemory[T],
        base_beam_width: int,
        best_history: Sequence[float],
    ) -> PlannerThought:
        root_family = self.domain.classify_family(root)
        pathology = self.domain.detect_pathology(root, mode)
        beam_best = min(node.planner.scalar for node in beam)
        family_hypotheses = self._family_hypotheses(beam)
        beam_diversity = len(family_hypotheses)
        top_prob = family_hypotheses[0][1] if family_hypotheses else 1.0
        second_prob = family_hypotheses[1][1] if len(family_hypotheses) > 1 else 0.0
        convergence_ratio = top_prob
        convergence = "high" if convergence_ratio >= 0.8 else "medium" if convergence_ratio >= 0.55 else "low"
        uncertainty = max(0.0, min(1.0, (1.0 - top_prob) + 0.6 * second_prob + (0.1 if beam_diversity >= 3 else 0.0)))
        stagnation_steps = self._stagnation_steps(best_history)
        best_node = min(beam, key=lambda n: n.planner.scalar)
        last_ops = best_node.path[-2:] if best_node.path else []

        ops = self.domain.operator_bank(mode, root, memory.relevant_promoted(self.domain.name, mode, root_family, pathology))
        scores: Dict[str, float] = {op.name: 0.0 for op in ops}
        reasons: Dict[str, List[str]] = defaultdict(list)

        for op_name, boost in self.domain.policy_hints(mode, pathology).items():
            if op_name in scores:
                scores[op_name] += boost
                reasons[op_name].append(f"pathology:{pathology}")

        for node in memory.relevant_nodes(self.domain.name, mode, root_family, pathology)[:10]:
            if node.motif in scores:
                scores[node.motif] += 1.6 * node.confidence
                reasons[node.motif].append(f"motif_node:{node.confidence:.2f}")

        continuation_scores = memory.recommend_continuations(self.domain.name, mode, root_family, pathology, last_ops)
        for op_name, boost in continuation_scores.items():
            if op_name in scores:
                scores[op_name] += boost
                reasons[op_name].append(f"motif_continue:{'->'.join(last_ops) if last_ops else 'root'}")

        for card in memory.relevant_cards(self.domain.name, mode, root_family, pathology)[:10]:
            proof_mean = statistics.mean(card.proof_gain) if card.proof_gain else 0.0
            hidden_mean = statistics.mean(card.hidden_gain) if card.hidden_gain else 0.0
            attack_penalty = 0.25 * card.attack_failures + 0.35 * card.attack_breaks
            planner_conf = memory.planner_hypotheses.get(f"prioritize:{card.motif}", {})
            calibration_bonus = 0.15 * planner_conf.get("confirmed", 0) - 0.25 * planner_conf.get("wrong", 0)
            if card.motif in scores:
                scores[card.motif] += 3.0 * card.confidence + 0.02 * proof_mean + 0.03 * hidden_mean + calibration_bonus - attack_penalty
                reasons[card.motif].append(f"memory_conf={card.confidence:.2f}")
            elif "+" in card.motif and card.confidence > 0.6:
                parts = [p for p in card.motif.split("+") if p]
                if len(parts) >= 2 and last_ops and parts[: len(last_ops)] == list(last_ops) and len(parts) > len(last_ops):
                    nxt = parts[len(last_ops)]
                    if nxt in scores:
                        scores[nxt] += 1.2 * card.confidence
                        reasons[nxt].append(f"motif_chain:{card.motif}")
                else:
                    for part in parts:
                        if part in scores:
                            scores[part] += 0.75 * card.confidence
                            reasons[part].append(f"composite_hint:{card.motif}")

        rejected: List[str] = []
        for card in memory.relevant_anti_patterns(self.domain.name, mode, root_family, pathology)[:8]:
            parts = [p for p in card.motif.split("+") if p]
            affected = parts if parts else [card.motif]
            for part in affected:
                if part in scores:
                    scores[part] -= 2.2 * max(0.35, card.confidence) + 0.25 * card.planner_only_traps + 0.25 * card.attack_breaks
                    reasons[part].append("anti_pattern")
                    rejected.append(part)

        if beam_diversity <= 1 and depth >= 2:
            for op in ops:
                if self._op_style(op) == "rewrite":
                    scores[op.name] += 0.7
                    reasons[op.name].append("diversify_family_shift")

        promoted = memory.relevant_promoted(self.domain.name, mode, root_family, pathology)
        if convergence == "high" and promoted:
            for op in promoted[:3]:
                if op.name in scores:
                    scores[op.name] += 1.4
                    reasons[op.name].append("promoted_operator")

        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        prioritized = [name for name, score in ranked if score > 0][:6]
        rejected_ops = list(dict.fromkeys([name for name, score in ranked if score < -0.25][:3] + rejected[:3]))
        strong_score = ranked[0][1] if ranked else 0.0

        if convergence == "high" and strong_score >= 2.35 and promoted and stagnation_steps >= 1:
            strategy = "reforge"
        elif uncertainty >= 0.55:
            strategy = "hedge"
        elif stagnation_steps >= self.config.stagnation_patience:
            strategy = "explore"
        elif convergence == "high" and strong_score >= 1.7:
            strategy = "exploit"
        else:
            strategy = "balance"

        if strategy == "reforge":
            exploit_weight, diversify_weight = 0.9, 0.1
            beam_width = max(4, base_beam_width - 2)
        elif strategy == "exploit":
            exploit_weight, diversify_weight = 0.78, 0.22
            beam_width = max(5, base_beam_width - 1)
        elif strategy == "hedge":
            exploit_weight, diversify_weight = 0.45, 0.55
            beam_width = base_beam_width + 4
        elif strategy == "explore":
            exploit_weight, diversify_weight = 0.28, 0.72
            beam_width = base_beam_width + 5
        else:
            exploit_weight, diversify_weight = 0.58, 0.42
            beam_width = base_beam_width + (2 if rejected_ops else 0)

        hidden_probe = strategy in {"hedge", "exploit", "reforge"} or stagnation_steps >= 1 or bool(rejected_ops)
        local_refine = strategy in {"exploit", "reforge"} and uncertainty < 0.45
        suggest_reforge = strategy == "reforge"

        hypotheses: List[ThoughtHypothesis] = []
        for name in prioritized[:3]:
            conf = max(0.05, min(0.99, 0.42 + 0.13 * scores[name]))
            hypotheses.append(ThoughtHypothesis("prioritize", name, conf, ";".join(reasons.get(name, ["search_default"]))[:140], "lower proof+hidden scalar"))
        for name in rejected_ops[:2]:
            hypotheses.append(ThoughtHypothesis("avoid", name, 0.58, "anti_pattern or attack failure evidence", "avoid planner-only trap"))
        if continuation_scores:
            strongest = max(continuation_scores.items(), key=lambda kv: kv[1])[0]
            hypotheses.append(ThoughtHypothesis("motif_follow", strongest, 0.63, f"continuation after {'->'.join(last_ops) if last_ops else 'root'}", "follow attack-surviving motif continuation"))
        if strategy in {"explore", "hedge"}:
            hypotheses.append(ThoughtHypothesis("diversify", "beam", 0.64 if strategy == "explore" else 0.57, "family ambiguity or stagnation", "keep diverse fallbacks alive"))
        if strategy == "hedge" and len(family_hypotheses) > 1:
            fam, prob = family_hypotheses[1]
            hypotheses.append(ThoughtHypothesis("hedge_family", fam, max(0.4, min(0.8, prob + 0.15)), "secondary family still plausible", "preserve alternate family branch"))
        if hidden_probe:
            target = prioritized[0] if prioritized else "top_candidates"
            hypotheses.append(ThoughtHypothesis("probe", target, 0.62, "planner confidence requires hidden-side check", "catch planner-only gains early"))
        if local_refine:
            target = prioritized[0] if prioritized else "top_candidates"
            hypotheses.append(ThoughtHypothesis("local_refine", target, 0.61, "high convergence and low uncertainty", "favor precise low-noise search"))
        if strategy == "exploit":
            target = prioritized[0] if prioritized else "top_candidates"
            hypotheses.append(ThoughtHypothesis("exploit", target, 0.67, "dominant family with strong priors", "compress search onto best branch"))
        if suggest_reforge:
            target = prioritized[0] if prioritized else "promoted_operator"
            hypotheses.append(ThoughtHypothesis("reforge", target, 0.72, "stagnation plus high-confidence promoted operator", "skip wasted exploration and jump to champion reforge"))

        fam_note = ", ".join(f"{name}:{prob:.2f}" for name, prob in family_hypotheses[:2]) if family_hypotheses else "unknown"
        motif_note = "root" if not last_ops else "->".join(last_ops)
        note = (
            f"strategy={strategy}; family_hypotheses={fam_note}; motif_context={motif_note}; memory points at "
            f"{', '.join(prioritized[:2]) if prioritized else 'no strong priors'} and rejects "
            f"{', '.join(rejected_ops[:2]) if rejected_ops else 'no strong rejections'}"
        )
        return PlannerThought(
            depth,
            mode,
            root_family,
            pathology,
            beam_best,
            beam_diversity,
            prioritized,
            rejected_ops,
            beam_width,
            suggest_reforge,
            convergence,
            strategy,
            uncertainty,
            exploit_weight,
            diversify_weight,
            hidden_probe,
            local_refine,
            family_hypotheses,
            stagnation_steps,
            hypotheses,
            note,
        )

    def _reorder_ops(self, ops: Sequence[DomainOperator[T]], thought: PlannerThought) -> List[DomainOperator[T]]:
        priority_rank = {name: idx for idx, name in enumerate(thought.prioritized_ops)}
        reject_set = set(thought.rejected_ops)

        def style_bias(op: DomainOperator[T]) -> int:
            style = self._op_style(op)
            if thought.strategy in {"explore", "hedge"}:
                return 0 if style == "rewrite" else 1 if style == "generic" else 2
            if thought.strategy in {"exploit", "reforge"} or thought.local_refine:
                return 0 if style == "local" else 1 if style == "generic" else 2
            return 1 if style == "generic" else 0 if style == "local" else 2

        def key(op: DomainOperator[T]) -> Tuple[int, int, int, str]:
            if op.name in priority_rank:
                return (0, priority_rank[op.name], style_bias(op), op.name)
            if op.name in reject_set:
                return (3, 0, style_bias(op), op.name)
            return (1, style_bias(op), 0, op.name)

        return sorted(ops, key=key)


    def _expand_node(self, node: SearchNode[T], ops: Sequence[DomainOperator[T]], planner: Arena[Any], mode: str, root: T) -> List[SearchNode[T]]:
        children: List[SearchNode[T]] = []
        seen = set()
        for op in ops:
            cand = op.fn(node.candidate)
            sig = self.domain.candidate_hash(cand)
            if sig in seen:
                continue
            seen.add(sig)
            planner_metrics = self._eval_on_arena(cand, planner, mode, root)
            path = node.path + [op.name]
            parent_hashes = node.parent_hashes + [self.domain.candidate_hash(node.candidate)]
            children.append(SearchNode(self._planner_key(planner_metrics, cand, len(path)), cand, path, parent_hashes, planner_metrics))
        return children

    def _search_rooted(self, root: T, mode: str, memory: PatternMemory[T], seed: int) -> Tuple[T, List[str], Dict[str, StepRecord[T]]]:
        planner, proof, hidden = self.domain.mode_arenas(mode, seed)
        probe_arena = self._probe_arena(hidden)
        root_hash = self.domain.candidate_hash(root)
        root_planner = self._eval_on_arena(root, planner, mode, root)
        root_proof = self._eval_on_arena(root, proof, mode, root)
        root_hidden = self._eval_on_arena(root, hidden, mode, root)
        root_node = SearchNode(self._planner_key(root_planner, root, 0), root, [], [], root_planner)
        beam = [root_node]
        steps: Dict[str, StepRecord[T]] = {
            root_hash: StepRecord(root, None, None, root_planner, root_proof, root_hidden, depth=0, thought=None)
        }
        seen_best = {root_hash: root_planner.scalar}
        best_history: List[float] = [root_planner.scalar]
        ops_bank = self.domain.operator_bank(mode, root, memory.relevant_promoted(self.domain.name, mode, self.domain.classify_family(root), self.domain.detect_pathology(root, mode)))

        for depth in range(1, self.config.max_depth + 1):
            thought = self._make_planner_thought(depth, beam, mode, root, memory, self.config.base_beam_width, best_history)
            ordered_ops = self._reorder_ops(ops_bank, thought)
            nodes_to_expand, ops_to_use = self._expand_plan(thought, sorted(beam, key=lambda n: n.sort_key), ordered_ops)
            if thought.suggest_reforge and depth > 2 and thought.stagnation_steps >= 1:
                break

            pool: List[SearchNode[T]] = []
            for node in nodes_to_expand:
                pool.extend(self._expand_node(node, ops_to_use, planner, mode, root))
            if not pool:
                break

            unique: Dict[str, SearchNode[T]] = {}
            for node in sorted(pool, key=lambda n: n.sort_key):
                sig = self.domain.candidate_hash(node.candidate)
                prev_best = seen_best.get(sig)
                if prev_best is not None and node.planner.scalar >= prev_best - 1e-9:
                    continue
                if sig in unique:
                    continue
                seen_best[sig] = node.planner.scalar
                unique[sig] = node

            candidates = list(unique.values())
            if not candidates:
                break
            if thought.hidden_probe:
                candidates = self._probe_candidates(candidates, probe_arena, mode, root, min(self.config.hidden_probe_topk, len(candidates)))
            beam = self._select_candidates(candidates, thought)
            if not beam:
                break
            best_history.append(min(n.planner.scalar for n in beam))
            for node in beam:
                sig = self.domain.candidate_hash(node.candidate)
                parent_hash = root_hash if not node.parent_hashes else node.parent_hashes[-1]
                steps[sig] = StepRecord(
                    candidate=node.candidate,
                    parent_hash=parent_hash,
                    op_name=node.path[-1] if node.path else None,
                    planner=node.planner,
                    proof=self._eval_on_arena(node.candidate, proof, mode, root),
                    hidden=self._eval_on_arena(node.candidate, hidden, mode, root),
                    depth=depth,
                    thought=thought,
                )

        def recover_path(sig: str) -> List[str]:
            path: List[str] = []
            cur = sig
            while cur in steps and steps[cur].op_name:
                path.append(steps[cur].op_name or "")
                cur = steps[cur].parent_hash or root_hash
                if cur == root_hash:
                    break
            return list(reversed(path))

        scored: List[Tuple[Tuple[float, float, float, int, int], str, List[str]]] = []
        for sig, step in steps.items():
            path = recover_path(sig)
            scored.append(((step.proof.scalar if step.proof else float("inf"), step.hidden.scalar if step.hidden else float("inf"), step.planner.scalar, len(path), self.domain.complexity(step.candidate)), sig, path))
        scored.sort(key=lambda item: item[0])
        champion_sig = scored[0][1]
        return steps[champion_sig].candidate, scored[0][2], steps


    def _mine_patterns(self, trajectories: Sequence[Trajectory[T]]) -> PatternMemory[T]:
        memory: PatternMemory[T] = PatternMemory()
        op_lookup = self.domain.operator_lookup()
        for traj in trajectories:
            root_family = self.domain.classify_family(traj.root)
            pathology = self.domain.detect_pathology(traj.root, traj.mode)
            ordered_hashes = traj.path_hashes
            op_path: List[str] = []

            for i in range(1, len(ordered_hashes)):
                prev = traj.steps[ordered_hashes[i - 1]]
                cur = traj.steps[ordered_hashes[i]]
                if not cur.op_name or not prev.proof or not cur.proof or not prev.hidden or not cur.hidden:
                    continue
                op_path.append(cur.op_name)

                planner_gain = prev.planner.scalar - cur.planner.scalar
                proof_gain = prev.proof.scalar - cur.proof.scalar
                hidden_gain = prev.hidden.scalar - cur.hidden.scalar
                survived = cur.hidden.correct == cur.hidden.total

                card_key = (self.domain.name, traj.mode, root_family, pathology, cur.op_name)
                card = memory.cards.setdefault(card_key, PatternCard(self.domain.name, cur.op_name, traj.mode, root_family, pathology))
                card.support += 1
                card.planner_gain.append(planner_gain)
                card.proof_gain.append(proof_gain)
                card.hidden_gain.append(hidden_gain)
                if survived:
                    card.hidden_passes += 1
                if planner_gain > 0 and proof_gain <= 0:
                    card.planner_only_traps += 1
                if i >= 2:
                    prev_op = traj.steps[ordered_hashes[i - 1]].op_name
                    if prev_op:
                        card.requires[prev_op] += 1

                node_key = (self.domain.name, traj.mode, root_family, pathology, cur.op_name)
                node = memory.nodes.setdefault(node_key, MotifNode(self.domain.name, cur.op_name, traj.mode, root_family, pathology))
                node.support += 1
                node.proof_gain.append(proof_gain)
                node.hidden_gain.append(hidden_gain)
                if planner_gain > 0 and proof_gain <= 0:
                    node.planner_traps += 1

                if i >= 2:
                    prev_op = traj.steps[ordered_hashes[i - 1]].op_name
                    if prev_op:
                        edge_key = (self.domain.name, traj.mode, root_family, pathology, prev_op, cur.op_name)
                        edge = memory.edges.setdefault(edge_key, MotifEdge(self.domain.name, traj.mode, root_family, pathology, prev_op, cur.op_name))
                        edge.support += 1
                        edge.proof_gain.append(proof_gain)
                        edge.hidden_gain.append(hidden_gain)
                        if planner_gain > 0 and proof_gain <= 0:
                            edge.planner_traps += 1

                if cur.thought:
                    good = proof_gain > 0 and hidden_gain >= 0 and cur.proof.correct == cur.proof.total and cur.hidden.correct == cur.hidden.total
                    for hyp in cur.thought.hypotheses:
                        label = f"{hyp.kind}:{hyp.target}"
                        if hyp.kind in {"prioritize", "motif_follow"}:
                            tkey = (self.domain.name, traj.mode, root_family, pathology, hyp.target)
                            tcard = memory.cards.setdefault(tkey, PatternCard(self.domain.name, hyp.target, traj.mode, root_family, pathology))
                            tcard.thought_support += 1
                            if cur.op_name == hyp.target and good:
                                tcard.thought_confirmed += 1
                                memory.record_hypothesis(label, "confirmed")
                            elif cur.op_name == hyp.target and not good:
                                tcard.thought_wrong += 1
                                tcard.planner_only_traps += 1
                                memory.record_hypothesis(label, "wrong")
                            else:
                                memory.record_hypothesis(label, "unresolved")
                        elif hyp.kind == "avoid":
                            tkey = (self.domain.name, traj.mode, root_family, pathology, hyp.target)
                            tcard = memory.cards.setdefault(tkey, PatternCard(self.domain.name, hyp.target, traj.mode, root_family, pathology))
                            tcard.thought_support += 1
                            if cur.op_name == hyp.target and good:
                                tcard.thought_wrong += 1
                                memory.record_hypothesis(label, "wrong")
                            else:
                                tcard.thought_confirmed += 1
                                memory.record_hypothesis(label, "confirmed")
                        elif hyp.kind == "reforge":
                            if traj.reforge_op and traj.reforge_op == hyp.target and traj.champion != traj.champion_before_reforge:
                                memory.record_hypothesis(label, "confirmed")
                            elif traj.reforge_op is None:
                                memory.record_hypothesis(label, "unresolved")
                            else:
                                memory.record_hypothesis(label, "wrong")
                        elif hyp.kind == "diversify":
                            if cur.thought.beam_diversity > 1 and good:
                                memory.record_hypothesis(label, "confirmed")
                            elif cur.thought.beam_diversity <= 1 and not good:
                                memory.record_hypothesis(label, "wrong")
                            else:
                                memory.record_hypothesis(label, "unresolved")
                        else:
                            if good:
                                memory.record_hypothesis(label, "confirmed")
                            elif hyp.confidence >= 0.6:
                                memory.record_hypothesis(label, "wrong")
                            else:
                                memory.record_hypothesis(label, "unresolved")

            ops_only = [traj.steps[h].op_name for h in ordered_hashes[1:] if traj.steps[h].op_name]
            for j in range(1, len(ops_only)):
                a, b = ops_only[j - 1], ops_only[j]
                before = traj.steps[ordered_hashes[j - 0]]
                after = traj.steps[ordered_hashes[j + 1]]
                motif = f"{a}+{b}"
                key = (self.domain.name, traj.mode, root_family, pathology, motif)
                card = memory.cards.setdefault(key, PatternCard(self.domain.name, motif, traj.mode, root_family, pathology))
                card.support += 1
                card.planner_gain.append(before.planner.scalar - after.planner.scalar)
                card.proof_gain.append(before.proof.scalar - after.proof.scalar)
                card.hidden_gain.append(before.hidden.scalar - after.hidden.scalar)
                if after.hidden.correct == after.hidden.total:
                    card.hidden_passes += 1
                card.requires[a] += 1
                card.requires[b] += 1

            for j in range(2, len(ops_only)):
                a, b, c = ops_only[j - 2], ops_only[j - 1], ops_only[j]
                before = traj.steps[ordered_hashes[j - 1]]
                after = traj.steps[ordered_hashes[j + 1]]
                motif = f"{a}+{b}+{c}"
                key = (self.domain.name, traj.mode, root_family, pathology, motif)
                card = memory.cards.setdefault(key, PatternCard(self.domain.name, motif, traj.mode, root_family, pathology))
                card.support += 1
                card.planner_gain.append(before.planner.scalar - after.planner.scalar)
                card.proof_gain.append(before.proof.scalar - after.proof.scalar)
                card.hidden_gain.append(before.hidden.scalar - after.hidden.scalar)
                if after.hidden.correct == after.hidden.total:
                    card.hidden_passes += 1
                card.requires[a] += 1
                card.requires[b] += 1
                card.requires[c] += 1

        for card in memory.cards.values():
            hidden_mean = statistics.mean(card.hidden_gain) if card.hidden_gain else 0.0
            planner_mean = statistics.mean(card.planner_gain) if card.planner_gain else 0.0
            context = (card.domain_name, card.mode, card.root_family, card.pathology)
            if card.confidence >= 0.52 and hidden_mean > 0:
                if card.motif in op_lookup:
                    if op_lookup[card.motif] not in memory.promoted_ops[context]:
                        memory.promoted_ops[context].append(op_lookup[card.motif])
                elif "+" in card.motif:
                    seq = tuple(part for part in card.motif.split("+") if part)
                    if 2 <= len(seq) <= 3 and seq not in memory.promoted_sequences[context]:
                        memory.promoted_sequences[context].append(seq)
            if card.planner_only_traps >= 2 or (card.support >= 2 and card.confidence < 0.18 and planner_mean > 0):
                memory.anti_patterns.append(card)

        for node in memory.nodes.values():
            context = (node.domain_name, node.mode, node.root_family, node.pathology)
            hidden_mean = statistics.mean(node.hidden_gain) if node.hidden_gain else 0.0
            if node.confidence >= 0.58 and hidden_mean > 0 and node.motif in op_lookup:
                if op_lookup[node.motif] not in memory.promoted_ops[context]:
                    memory.promoted_ops[context].append(op_lookup[node.motif])

        return memory

    def _champion_reforge(self, root: T, champion: T, mode: str, memory: PatternMemory[T], seed: int) -> Tuple[T, Optional[str], float]:
        _, proof, hidden = self.domain.mode_arenas(mode, seed)
        base_proof = self._eval_on_arena(champion, proof, mode, root)
        base_hidden = self._eval_on_arena(champion, hidden, mode, root)
        best = champion
        best_name = None
        best_gain = 0.0
        family = self.domain.classify_family(root)
        pathology = self.domain.detect_pathology(root, mode)
        relevant = memory.relevant_promoted(self.domain.name, mode, family, pathology)
        if not relevant:
            return champion, None, 0.0
        for op in relevant[:6]:
            cand = op.fn(champion)
            proof_m = self._eval_on_arena(cand, proof, mode, root)
            hidden_m = self._eval_on_arena(cand, hidden, mode, root)
            gain = (base_proof.scalar + base_hidden.scalar) - (proof_m.scalar + hidden_m.scalar)
            if hidden_m.correct == hidden_m.total and proof_m.correct == proof_m.total and gain > best_gain + 1e-9:
                best = cand
                best_name = op.name
                best_gain = gain
        return best, best_name, best_gain

    def _causal_test(self, root: T, champion: T, path: Sequence[str], mode: str, seed: int) -> List[CausalEdit]:
        op_lookup = self.domain.operator_lookup()
        _, proof, hidden = self.domain.mode_arenas(mode, seed)
        base_proof = self._eval_on_arena(champion, proof, mode, root)
        base_hidden = self._eval_on_arena(champion, hidden, mode, root)
        out: List[CausalEdit] = []
        for idx, edit in enumerate(path):
            candidate = root
            for j, step in enumerate(path):
                if j == idx:
                    continue
                candidate = op_lookup[step].fn(candidate)
            proof_m = self._eval_on_arena(candidate, proof, mode, root)
            hidden_m = self._eval_on_arena(candidate, hidden, mode, root)
            proof_delta = proof_m.scalar - base_proof.scalar
            hidden_delta = hidden_m.scalar - base_hidden.scalar
            verdict = "essential" if proof_delta > 0.5 or hidden_delta > 0.5 else "neutral_or_redundant"
            out.append(CausalEdit(edit, proof_delta, hidden_delta, verdict))
        out.sort(key=lambda x: (x.proof_delta + x.hidden_delta), reverse=True)
        return out

    def _interaction_effects(self, root: T, champion: T, path: Sequence[str], mode: str, seed: int) -> List[InteractionEffect]:
        if len(path) < 2:
            return []
        op_lookup = self.domain.operator_lookup()
        _, proof, hidden = self.domain.mode_arenas(mode, seed)
        base_proof = self._eval_on_arena(champion, proof, mode, root)
        base_hidden = self._eval_on_arena(champion, hidden, mode, root)
        out: List[InteractionEffect] = []
        for i in range(len(path) - 1):
            pair = (path[i], path[i + 1])
            candidate = root
            skip = set(pair)
            skipped_first = skipped_second = False
            for step in path:
                if step == pair[0] and not skipped_first:
                    skipped_first = True
                    continue
                if step == pair[1] and not skipped_second:
                    skipped_second = True
                    continue
                candidate = op_lookup[step].fn(candidate)
            proof_m = self._eval_on_arena(candidate, proof, mode, root)
            hidden_m = self._eval_on_arena(candidate, hidden, mode, root)
            proof_delta = proof_m.scalar - base_proof.scalar
            hidden_delta = hidden_m.scalar - base_hidden.scalar
            verdict = "synergistic" if proof_delta + hidden_delta > 1.0 else "weak_or_redundant"
            out.append(InteractionEffect("+".join(pair), proof_delta, hidden_delta, verdict))
        out.sort(key=lambda x: (x.proof_delta + x.hidden_delta), reverse=True)
        return out

    def _thought_trace(self, traj: Trajectory[T]) -> List[str]:
        trace = []
        for sig in traj.path_hashes:
            thought = traj.steps[sig].thought
            if thought:
                trace.append(thought.short())
        return trace

    def _why_this_won(self, root: T, champion: T, proof: EvalMetrics, hidden: EvalMetrics, causal: Sequence[CausalEdit]) -> str:
        essential = [c.edit for c in causal if c.verdict == "essential"]
        root_family = self.domain.classify_family(root)
        champ_family = self.domain.classify_family(champion)
        lead = essential[:3] if essential else ["no single dominant edit"]
        return (
            f"moved from {root_family} to {champ_family}, proof avg={proof.avg_cost:.4f}, hidden worst={hidden.worst_cost:.4f}, "
            f"main causal edits: {', '.join(lead)}"
        )

    def _run_single(self, mode: str, root_name: str, root: T, memory: PatternMemory[T], seed: int) -> ModeResult[T]:
        champion, path, steps = self._search_rooted(root, mode, memory, seed)
        planner, proof, hidden = self.domain.mode_arenas(mode, seed)
        champion_before_reforge = champion
        reforge_candidate, reforge_op, reforge_gain = self._champion_reforge(root, champion, mode, memory, seed)
        champion = reforge_candidate
        champion_hash = self.domain.candidate_hash(champion)
        champion_planner = self._eval_on_arena(champion, planner, mode, root)
        champion_proof = self._eval_on_arena(champion, proof, mode, root)
        champion_hidden = self._eval_on_arena(champion, hidden, mode, root)
        if champion_hash not in steps:
            steps[champion_hash] = StepRecord(champion, self.domain.candidate_hash(champion_before_reforge), reforge_op, champion_planner, champion_proof, champion_hidden, depth=max(s.depth for s in steps.values()) + 1)
        path_hashes = [self.domain.candidate_hash(root)]
        cur = root
        for op_name in path:
            cur = self.domain.operator_lookup()[op_name].fn(cur)
            path_hashes.append(self.domain.candidate_hash(cur))
        if champion_hash != path_hashes[-1]:
            path_hashes.append(champion_hash)
        traj = Trajectory(self.domain.name, mode, root_name, root, champion, champion_hash, path_hashes, steps, champion_before_reforge, reforge_op)
        return ModeResult(mode, root_name, root, champion, champion_planner, champion_proof, champion_hidden, list(path), traj, reforge_gain)

    def run(self) -> Tuple[List[ModeResult[T]], PatternMemory[T], List[ForensicReport], Dict[str, float]]:
        roots_by_mode = self.domain.root_sets()
        memory: PatternMemory[T] = PatternMemory()
        last_results: List[ModeResult[T]] = []
        round_scores: Dict[str, float] = {}
        for round_idx in range(self.config.meta_rounds):
            trajectories: List[Trajectory[T]] = []
            results: List[ModeResult[T]] = []
            total_scalar = 0.0
            for mode, roots in roots_by_mode.items():
                for idx, (root_name, root) in enumerate(roots.items()):
                    seed = self.config.base_seed + round_idx * 100 + idx * 17 + (0 if mode == "repair" else 1000 if mode == "harden" else 2000)
                    res = self._run_single(mode, root_name, root, memory, seed)
                    results.append(res)
                    trajectories.append(res.trajectory)
                    total_scalar += res.proof.scalar + res.hidden.scalar
            round_scores[f"round_{round_idx}"] = total_scalar
            memory = self._mine_patterns(trajectories)
            last_results = results

        forensic_reports: List[ForensicReport] = []
        for mode, roots in roots_by_mode.items():
            for idx, (root_name, root) in enumerate(roots.items()):
                seed = self.config.base_seed + (self.config.meta_rounds - 1) * 100 + idx * 17 + (0 if mode == "repair" else 1000 if mode == "harden" else 2000)
                res = next(r for r in last_results if r.mode == mode and r.root_name == root_name)
                _, _, hidden_arena = self.domain.mode_arenas(mode, seed)
                causal = self._causal_test(root, res.champion, res.path, mode, seed)
                forensic_reports.append(
                    ForensicReport(
                        domain_name=self.domain.name,
                        root_name=root_name,
                        mode=mode,
                        root_family=self.domain.classify_family(root),
                        champion_family=self.domain.classify_family(res.champion),
                        path=res.path,
                        causal_edits=causal,
                        interactions=self._interaction_effects(root, res.champion, res.path, mode, seed),
                        suite_attribution=self.domain.suite_attribution(root, res.champion, hidden_arena, mode),
                        why=self._why_this_won(root, res.champion, res.proof, res.hidden, causal),
                        reforge_op=res.trajectory.reforge_op,
                        reforge_gain=res.reforge_gain,
                        rediscovery=self.domain.rediscovery_verdict(root, res.champion),
                        planner_thoughts=self._thought_trace(res.trajectory),
                    )
                )
        return last_results, memory, forensic_reports, round_scores

    def build_report(self, results: List[ModeResult[T]], memory: PatternMemory[T], forensic_reports: List[ForensicReport], round_scores: Dict[str, float]) -> str:
        lines: List[str] = []
        lines.append("# Smart Forge 1.7\n")
        lines.append("## What changed")
        lines.append("- Added adversarial escalation on top of motif graph memory.")
        lines.append("- Memory now tracks motif nodes, transition edges, and promoted operator sequences.")
        lines.append("- Planner thoughts can follow graph continuations instead of only relying on flat operator confidence.")
        lines.append("- Counter-Forge now attacks promoted sequences, planner beliefs, and triggers recovery passes for fragile champions.")
        lines.append("- This is the transformation-grammar upgrade: Forge now learns relationships, not just fragments.\n")

        lines.append("## Meta-round scores")
        for k, v in round_scores.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

        lines.append("## Final champions")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            lines.append(
                f"- [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total}, hidden {r.hidden.correct}/{r.hidden.total}, avg {r.proof.avg_cost:.4f}, worst_hidden {r.hidden.worst_cost:.4f}, reforge_gain {r.reforge_gain:.4f}"
            )
            lines.append(f"  - root: {self.domain.summarize(r.root)}")
            lines.append(f"  - champion: {self.domain.summarize(r.champion)}")
            lines.append(f"  - path: {' -> '.join(r.path) if r.path else '(root kept)'}")
        lines.append("")

        lines.append("## Promoted operators")
        promoted_any = False
        for key, ops in sorted(memory.promoted_ops.items()):
            if not ops:
                continue
            promoted_any = True
            domain_name, mode, family, pathology = key
            lines.append(f"- context=({domain_name}, {mode}, {family}, {pathology})")
            for op in ops[:5]:
                lines.append(f"  - {op.name}")
        if not promoted_any:
            lines.append("- none")
        lines.append("")

        lines.append("## Promoted motif sequences")
        if memory.promoted_sequences:
            for key, seqs in sorted(memory.promoted_sequences.items()):
                if not seqs:
                    continue
                domain_name, mode, family, pathology = key
                lines.append(f"- context=({domain_name}, {mode}, {family}, {pathology})")
                for seq in seqs[:6]:
                    lines.append(f"  - {' -> '.join(seq)}")
        else:
            lines.append("- none")
        lines.append("")

        lines.append("## Strongest pattern cards")
        for card in sorted(memory.cards.values(), key=lambda c: c.confidence, reverse=True)[:12]:
            lines.append(f"- {card.summary()}")
        lines.append("")

        lines.append("## Strongest motif nodes")
        for node in sorted(memory.nodes.values(), key=lambda n: n.confidence, reverse=True)[:12]:
            lines.append(f"- {node.summary()}")
        lines.append("")

        lines.append("## Strongest motif edges")
        for edge in sorted(memory.edges.values(), key=lambda e: e.confidence, reverse=True)[:12]:
            lines.append(f"- {edge.summary()}")
        lines.append("")

        lines.append("## Thinking mode / planner calibration")
        if memory.planner_hypotheses:
            for label, counts in sorted(memory.planner_hypotheses.items(), key=lambda kv: (kv[1].get('confirmed', 0), -kv[1].get('wrong', 0)), reverse=True)[:20]:
                lines.append(
                    f"- {label}: confirmed={counts.get('confirmed', 0)} wrong={counts.get('wrong', 0)} unresolved={counts.get('unresolved', 0)}"
                )
        else:
            lines.append("- no planner hypotheses recorded")
        lines.append("")

        lines.append("## Forensics")
        for fr in sorted(forensic_reports, key=lambda x: (x.mode, x.root_name)):
            lines.append(f"### {fr.mode} / {fr.root_name}")
            lines.append(f"- root_family: {fr.root_family}")
            lines.append(f"- champion_family: {fr.champion_family}")
            lines.append(f"- rediscovery: {fr.rediscovery}")
            lines.append(f"- path: {' -> '.join(fr.path) if fr.path else '(root kept)'}")
            if fr.reforge_op:
                lines.append(f"- champion_reforge: {fr.reforge_op} (gain {fr.reforge_gain:.4f})")
            lines.append(f"- why_this_won: {fr.why}")
            lines.append("- causal_edits:")
            for ce in fr.causal_edits[:8]:
                lines.append(f"  - {ce.edit}: verdict={ce.verdict}, proof_delta={ce.proof_delta:.4f}, hidden_delta={ce.hidden_delta:.4f}")
            if fr.interactions:
                lines.append("- interactions:")
                for ie in fr.interactions[:6]:
                    lines.append(f"  - {ie.pair}: verdict={ie.verdict}, proof_delta={ie.proof_delta:.4f}, hidden_delta={ie.hidden_delta:.4f}")
            if fr.planner_thoughts:
                lines.append("- planner_thought_trace:")
                for line in fr.planner_thoughts[:6]:
                    lines.append(f"  - {line}")
            if fr.suite_attribution:
                lines.append("- suite_attribution:")
                for kind, cost_delta, corr_delta in fr.suite_attribution[:6]:
                    lines.append(f"  - {kind}: cost_delta={cost_delta:.4f}, correctness_delta={corr_delta:.4f}")
            lines.append("")
        return "\n".join(lines)




@dataclass
class PromotedOpAttack:
    op_name: str
    hidden_scalar: float
    attack_scalar: float
    hidden_correct: str
    attack_correct: str
    verdict: str


@dataclass
class SequenceAttackCheck:
    sequence: Tuple[str, ...]
    hidden_scalar: float
    attack_scalar: float
    hidden_correct: str
    attack_correct: str
    verdict: str


@dataclass
class RecoveryReport:
    attempted: bool
    recovered: bool
    path: List[str]
    hidden_scalar: float
    attack_scalar: float
    gain: float
    verdict: str


@dataclass
class CounterCaseReport:
    domain_name: str
    mode: str
    root_name: str
    weak_kinds: List[str]
    root_hidden_scalar: float
    champion_hidden_scalar: float
    champion_attack_scalar: float
    root_attack_scalar: float
    degradation_ratio: float
    verdict: str
    promoted_checks: List[PromotedOpAttack]
    sequence_checks: List[SequenceAttackCheck]
    planner_flags: List[str]
    recovery: Optional[RecoveryReport]


class CounterForge:
    def __init__(self, bundles: Dict[str, Any]):
        self.bundles = bundles

    def _seed_for(self, domain: DomainPack[Any], config: EngineConfig, mode: str, root_name: str) -> int:
        roots = list(domain.root_sets()[mode].keys())
        idx = roots.index(root_name)
        offset = 0 if mode == "repair" else 1000 if mode == "harden" else 2000
        return config.base_seed + (config.meta_rounds - 1) * 100 + idx * 17 + offset

    def _sorting_case_cost(self, domain: SortingDomainPack, candidate: SortingCandidate, arr: List[int]) -> Tuple[bool, float]:
        res = domain._execute_candidate(candidate, arr)
        ok = domain._is_correct_sort(arr, res.output)
        n = max(1, len(arr))
        if not ok:
            return False, float("inf")
        return True, res.cost.scalar() / (n * n)

    def _path_case_cost(self, domain: PathfindingDomainPack, candidate: PathfindingCandidate, case: GridCase) -> Tuple[bool, float]:
        res = domain._execute_candidate(candidate, case)
        ok = ((not case.has_path and not res.found) or (case.has_path and res.found and res.path_len is not None))
        norm = max(1, case.open_cells)
        cost = res.expansions / norm
        if case.has_path and res.path_len is not None:
            opt = case.optimal_len or res.path_len
            cost += 0.8 * max(0.0, (res.path_len - opt) / max(1, opt))
        if not ok:
            return False, float("inf")
        return True, cost

    def _weak_kinds(self, domain: DomainPack[Any], result: ModeResult[Any], seed: int) -> List[str]:
        _, _, hidden = domain.mode_arenas(result.mode, seed)
        buckets: Dict[str, List[float]] = defaultdict(list)
        fails: Dict[str, int] = defaultdict(int)
        for kind, case in hidden.cases:
            if isinstance(domain, SortingDomainPack):
                ok, cost = self._sorting_case_cost(domain, result.champion, case)
            else:
                ok, cost = self._path_case_cost(domain, result.champion, case)
            if not ok or not math.isfinite(cost):
                fails[kind] += 1
                buckets[kind].append(10.0)
            else:
                buckets[kind].append(cost)
        scored = []
        for kind, vals in buckets.items():
            scored.append((statistics.mean(vals) + 2.0 * fails[kind], kind))
        scored.sort(reverse=True)
        weak = [k for _, k in scored[:3]]
        return weak or [hidden.cases[0][0]]

    def _build_attack_arena(self, domain: DomainPack[Any], result: ModeResult[Any], weak_kinds: Sequence[str], seed: int) -> Arena[Any]:
        path = list(result.path)
        if isinstance(domain, SortingDomainPack):
            extras = ["reversed", "organ", "saw", "few_unique", "extremes", "nearly"]
            if any("shell" in op or "gap" in op for op in path):
                extras.extend(["reversed", "few_unique", "extremes"])
            if any("selection" in op for op in path):
                extras.extend(["organ", "nearly"])
            kinds = list(dict.fromkeys(list(weak_kinds) + extras))
            sizes = [32, 48, 64, 96, 128]
            return domain._build_arena(seed + 9001, 96, sizes, f"counter_{result.mode}_{result.root_name}", kinds)
        extras = ["dense", "mazeish", "blocked", "corridor", "medium"]
        if any("greedy" in op or "weight_" in op or "weighted" in op for op in path):
            extras.extend(["blocked", "mazeish", "corridor"])
        if any("bfs" in op or "dijkstra" in op for op in path):
            extras.extend(["dense", "mazeish"])
        kinds = list(dict.fromkeys(list(weak_kinds) + extras))
        sizes = [15, 17, 19, 21]
        return domain._build_arena(seed + 9001, 72, sizes, f"counter_{result.mode}_{result.root_name}", kinds)

    def _promoted_checks(self, domain: DomainPack[Any], memory: PatternMemory[Any], result: ModeResult[Any], hidden: Arena[Any], attack: Arena[Any], champion_hidden_scalar: float) -> List[PromotedOpAttack]:
        root_family = domain.classify_family(result.root)
        pathology = domain.detect_pathology(result.root, result.mode)
        ops = memory.relevant_promoted(domain.name, result.mode, root_family, pathology)[:5]
        checks: List[PromotedOpAttack] = []
        for op in ops:
            cand = op.fn(result.champion)
            hidden_m = domain.evaluate(cand, hidden, result.mode, result.root)
            attack_m = domain.evaluate(cand, attack, result.mode, result.root)
            hidden_ok = f"{hidden_m.correct}/{hidden_m.total}"
            attack_ok = f"{attack_m.correct}/{attack_m.total}"
            if attack_m.correct < attack_m.total:
                verdict = "broken_under_attack"
            elif attack_m.scalar <= champion_hidden_scalar * 1.08:
                verdict = "stable"
            elif attack_m.scalar <= champion_hidden_scalar * 1.25:
                verdict = "stress_sensitive"
            else:
                verdict = "brittle"
            checks.append(PromotedOpAttack(op.name, hidden_m.scalar, attack_m.scalar, hidden_ok, attack_ok, verdict))
        return checks

    def _sequence_checks(self, domain: DomainPack[Any], memory: PatternMemory[Any], result: ModeResult[Any], hidden: Arena[Any], attack: Arena[Any], champion_hidden_scalar: float) -> List[SequenceAttackCheck]:
        root_family = domain.classify_family(result.root)
        pathology = domain.detect_pathology(result.root, result.mode)
        sequences: List[Tuple[str, ...]] = []
        seen = set()
        for seq in memory.relevant_sequences(domain.name, result.mode, root_family, pathology)[:8]:
            if 2 <= len(seq) <= 3 and seq not in seen:
                sequences.append(seq)
                seen.add(seq)
        path = list(result.path)
        for k in (2, 3):
            for i in range(0, max(0, len(path) - k + 1)):
                seq = tuple(path[i:i+k])
                if len(seq) == k and seq not in seen:
                    sequences.append(seq)
                    seen.add(seq)
        lookup = domain.operator_lookup()
        checks: List[SequenceAttackCheck] = []
        for seq in sequences[:8]:
            cand = result.root
            valid = True
            for op_name in seq:
                op = lookup.get(op_name)
                if op is None:
                    valid = False
                    break
                cand = op.fn(cand)
            if not valid:
                continue
            hidden_m = domain.evaluate(cand, hidden, result.mode, result.root)
            attack_m = domain.evaluate(cand, attack, result.mode, result.root)
            hidden_ok = f"{hidden_m.correct}/{hidden_m.total}"
            attack_ok = f"{attack_m.correct}/{attack_m.total}"
            if attack_m.correct < attack_m.total:
                verdict = "broken_under_attack"
            elif attack_m.scalar <= champion_hidden_scalar * 1.1:
                verdict = "stable"
            elif attack_m.scalar <= champion_hidden_scalar * 1.28:
                verdict = "stress_sensitive"
            else:
                verdict = "brittle"
            checks.append(SequenceAttackCheck(seq, hidden_m.scalar, attack_m.scalar, hidden_ok, attack_ok, verdict))
        return checks

    def _planner_flags(self, result: ModeResult[Any], degradation_ratio: float, weak_kinds: Sequence[str]) -> List[str]:
        flags: List[str] = []
        seen = set()
        thoughts = []
        for sig in result.trajectory.path_hashes:
            step = result.trajectory.steps.get(sig)
            if step and step.thought:
                thoughts.append(step.thought)
        for thought in thoughts:
            if degradation_ratio > 1.35 and thought.strategy == "exploit":
                flags.append("planner_exploit_regime_appears_overconfident_under_attack")
            if degradation_ratio > 1.25 and thought.strategy == "reforge":
                flags.append("planner_reforge_trigger_looks_optimistic_under_attack")
            if thought.convergence == "high" and degradation_ratio > 1.30:
                flags.append("high_convergence_may_have_hidden_brittleness")
            if thought.uncertainty < 0.35 and degradation_ratio > 1.25 and len(thought.family_hypotheses) > 1:
                flags.append("family_classifier_may_be_overconfident")
            if not thought.hidden_probe and degradation_ratio > 1.2:
                flags.append("hidden_probe_missed_attack_sensitive_branch")
            for hyp in thought.hypotheses:
                label = f"{hyp.kind}:{hyp.target}"
                if label in seen:
                    continue
                seen.add(label)
                if degradation_ratio > 1.35 and hyp.kind in {"prioritize", "motif_follow"}:
                    flags.append(f"planner_prioritized_{hyp.target}_before_large_attack_drop")
        if weak_kinds:
            flags.append(f"counter_targeted_{'_'.join(weak_kinds[:2])}")
        return list(dict.fromkeys(flags))[:8]

    def _recovery_pass(self, domain_name: str, result: ModeResult[Any], hidden: Arena[Any], attack: Arena[Any]) -> Optional[RecoveryReport]:
        pack = self.bundles[domain_name]
        domain = pack["engine"].domain
        memory = pack["memory"]
        root_family = domain.classify_family(result.root)
        pathology = domain.detect_pathology(result.root, result.mode)
        ops = domain.operator_bank(result.mode, result.root, memory.relevant_promoted(domain.name, result.mode, root_family, pathology))
        continuation = memory.recommend_continuations(domain.name, result.mode, root_family, pathology, result.path[-2:])
        ordered = sorted(
            ops,
            key=lambda op: (
                -continuation.get(op.name, 0.0),
                -(statistics.mean(memory.cards[(domain.name, result.mode, root_family, pathology, op.name)].hidden_gain) if (domain.name, result.mode, root_family, pathology, op.name) in memory.cards and memory.cards[(domain.name, result.mode, root_family, pathology, op.name)].hidden_gain else 0.0),
                op.name,
            ),
        )

        base_hidden = domain.evaluate(result.champion, hidden, result.mode, result.root)
        base_attack = domain.evaluate(result.champion, attack, result.mode, result.root)
        base_score = base_hidden.scalar + 1.45 * base_attack.scalar
        best_cand = result.champion
        best_path: List[str] = []
        best_hidden = base_hidden
        best_attack = base_attack
        seen = {domain.candidate_hash(result.champion)}
        beam: List[Tuple[float, Any, List[str]]] = [(base_score, result.champion, [])]
        max_depth = 2 if result.mode != "repair" else 3
        beam_width = 5

        for _depth in range(1, max_depth + 1):
            pool: List[Tuple[float, Any, List[str]]] = []
            for _, cand, path in beam:
                for op in ordered[: max(6, beam_width + 2)]:
                    child = op.fn(cand)
                    sig = domain.candidate_hash(child)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    h = domain.evaluate(child, hidden, result.mode, result.root)
                    a = domain.evaluate(child, attack, result.mode, result.root)
                    if h.correct < h.total or a.correct < a.total:
                        continue
                    score = h.scalar + 1.45 * a.scalar + 0.12 * len(path)
                    pool.append((score, child, path + [op.name]))
                    if score < (best_hidden.scalar + 1.45 * best_attack.scalar) - 1e-9:
                        best_cand = child
                        best_path = path + [op.name]
                        best_hidden = h
                        best_attack = a
            if not pool:
                break
            pool.sort(key=lambda item: item[0])
            beam = pool[:beam_width]

        gain = base_score - (best_hidden.scalar + 1.45 * best_attack.scalar)
        if gain > 0.5:
            return RecoveryReport(
                attempted=True,
                recovered=True,
                path=best_path,
                hidden_scalar=best_hidden.scalar,
                attack_scalar=best_attack.scalar,
                gain=gain,
                verdict="recovered_sturdier_descendant",
            )
        return RecoveryReport(
            attempted=True,
            recovered=False,
            path=[],
            hidden_scalar=base_hidden.scalar,
            attack_scalar=base_attack.scalar,
            gain=0.0,
            verdict="no_recovery_found",
        )

    def attack_bundle(self, domain_name: str) -> Tuple[List[CounterCaseReport], str]:
        pack = self.bundles[domain_name]
        domain = pack["engine"].domain
        config = pack["engine"].config
        memory = pack["memory"]
        reports: List[CounterCaseReport] = []
        lines: List[str] = [f"## Counter-Forge / {domain_name}", "", "### Layer 1: targeted champion attack", ""]
        for result in sorted(pack["results"], key=lambda r: (r.mode, r.root_name)):
            seed = self._seed_for(domain, config, result.mode, result.root_name)
            _, _, hidden = domain.mode_arenas(result.mode, seed)
            weak_kinds = self._weak_kinds(domain, result, seed)
            attack = self._build_attack_arena(domain, result, weak_kinds, seed)
            root_hidden = domain.evaluate(result.root, hidden, result.mode, result.root)
            root_attack = domain.evaluate(result.root, attack, result.mode, result.root)
            champ_hidden = domain.evaluate(result.champion, hidden, result.mode, result.root)
            champ_attack = domain.evaluate(result.champion, attack, result.mode, result.root)
            degradation_ratio = champ_attack.scalar / max(1e-9, champ_hidden.scalar)
            if champ_attack.correct < champ_attack.total:
                verdict = "champion_broken_under_attack"
            elif degradation_ratio <= 1.10:
                verdict = "robust"
            elif degradation_ratio <= 1.30:
                verdict = "stressed_but_survives"
            else:
                verdict = "fragile"

            promoted = self._promoted_checks(domain, memory, result, hidden, attack, champ_hidden.scalar)
            seq_checks = self._sequence_checks(domain, memory, result, hidden, attack, champ_hidden.scalar)
            planner_flags = self._planner_flags(result, degradation_ratio, weak_kinds)
            recovery = self._recovery_pass(domain_name, result, hidden, attack) if verdict in {"fragile", "champion_broken_under_attack"} else None

            reports.append(CounterCaseReport(
                domain_name,
                result.mode,
                result.root_name,
                list(weak_kinds),
                root_hidden.scalar,
                champ_hidden.scalar,
                champ_attack.scalar,
                root_attack.scalar,
                degradation_ratio,
                verdict,
                promoted,
                seq_checks,
                planner_flags,
                recovery,
            ))

            lines.append(f"#### {result.mode} / {result.root_name}")
            lines.append(f"- weak_kinds: {', '.join(weak_kinds)}")
            lines.append(f"- hidden scalar: root={root_hidden.scalar:.4f}, champion={champ_hidden.scalar:.4f}")
            lines.append(f"- attack scalar: root={root_attack.scalar:.4f}, champion={champ_attack.scalar:.4f}")
            lines.append(f"- degradation_ratio: {degradation_ratio:.4f}")
            lines.append(f"- verdict: {verdict}")
            if recovery:
                lines.append(f"- recovery: {recovery.verdict}, gain={recovery.gain:.4f}, path={' -> '.join(recovery.path) if recovery.path else '(none)'}")
            lines.append("")
        lines.extend(["### Layer 2: promoted-operator, sequence, and planner audit", ""])
        for item in reports:
            lines.append(f"#### {item.mode} / {item.root_name}")
            if item.planner_flags:
                lines.append("- planner flags:")
                for flag in item.planner_flags:
                    lines.append(f"  - {flag}")
            if item.promoted_checks:
                lines.append("- promoted operator attack checks:")
                for check in item.promoted_checks:
                    lines.append(f"  - {check.op_name}: hidden={check.hidden_scalar:.4f} ({check.hidden_correct}), attack={check.attack_scalar:.4f} ({check.attack_correct}), verdict={check.verdict}")
            if item.sequence_checks:
                lines.append("- sequence attack checks:")
                for check in item.sequence_checks[:6]:
                    lines.append(f"  - {' -> '.join(check.sequence)}: hidden={check.hidden_scalar:.4f} ({check.hidden_correct}), attack={check.attack_scalar:.4f} ({check.attack_correct}), verdict={check.verdict}")
            if item.recovery:
                lines.append("- recovery:")
                lines.append(f"  - verdict={item.recovery.verdict}, gain={item.recovery.gain:.4f}, hidden={item.recovery.hidden_scalar:.4f}, attack={item.recovery.attack_scalar:.4f}, path={' -> '.join(item.recovery.path) if item.recovery.path else '(none)'}")
            lines.append("")
        return reports, "\n".join(lines)

    def run(self) -> Tuple[Dict[str, List[CounterCaseReport]], str]:
        sections = ["# Counter-Forge Report", "", "Counter-Forge attacks champions, promoted operators, promoted motif sequences, and planner assumptions with larger targeted arenas and recovery attempts.", ""]
        all_reports: Dict[str, List[CounterCaseReport]] = {}
        for domain_name in sorted(self.bundles):
            reports, section = self.attack_bundle(domain_name)
            all_reports[domain_name] = reports
            sections.append(section)
            sections.append("")
        return all_reports, "\n".join(sections)



def _append_unique_antipattern(memory: PatternMemory[Any], card: PatternCard) -> None:
    key = (card.domain_name, card.mode, card.root_family, card.pathology, card.motif)
    for existing in memory.anti_patterns:
        if (existing.domain_name, existing.mode, existing.root_family, existing.pathology, existing.motif) == key:
            return
    memory.anti_patterns.append(card)


def apply_counter_feedback(bundles: Dict[str, Any], counter_reports: Dict[str, List[CounterCaseReport]]) -> None:
    for domain_name, reports in counter_reports.items():
        pack = bundles[domain_name]
        domain = pack["engine"].domain
        memory = pack["memory"]
        results = {(r.mode, r.root_name): r for r in pack["results"]}
        for report in reports:
            result = results[(report.mode, report.root_name)]
            root_family = domain.classify_family(result.root)
            pathology = domain.detect_pathology(result.root, report.mode)
            context4 = (domain.name, report.mode, root_family, pathology)

            for check in report.promoted_checks:
                key = (domain.name, report.mode, root_family, pathology, check.op_name)
                card = memory.cards.setdefault(key, PatternCard(domain.name, check.op_name, report.mode, root_family, pathology))
                card.attack_support += 1
                if check.verdict != "stable":
                    card.attack_failures += 1
                if check.verdict == "broken_under_attack":
                    card.attack_breaks += 1
                    card.planner_only_traps += 1
                    _append_unique_antipattern(memory, card)
                elif check.verdict == "brittle":
                    card.planner_only_traps += 1
                    _append_unique_antipattern(memory, card)

                node = memory.nodes.get((domain.name, report.mode, root_family, pathology, check.op_name))
                if node:
                    if check.verdict != "stable":
                        node.attack_failures += 1
                    if check.verdict == "broken_under_attack":
                        node.attack_breaks += 1
                    if check.verdict in {"brittle", "broken_under_attack"}:
                        node.planner_traps += 1

            for check in report.sequence_checks:
                motif = "+".join(check.sequence)
                card = memory.cards.setdefault((domain.name, report.mode, root_family, pathology, motif), PatternCard(domain.name, motif, report.mode, root_family, pathology))
                card.attack_support += 1
                if check.verdict != "stable":
                    card.attack_failures += 1
                if check.verdict == "broken_under_attack":
                    card.attack_breaks += 1
                    card.planner_only_traps += 1
                    _append_unique_antipattern(memory, card)
                elif check.verdict == "brittle":
                    _append_unique_antipattern(memory, card)

                for op_name in check.sequence:
                    node = memory.nodes.get((domain.name, report.mode, root_family, pathology, op_name))
                    if node and check.verdict != "stable":
                        node.attack_failures += 1
                        if check.verdict == "broken_under_attack":
                            node.attack_breaks += 1
                for a, b in zip(check.sequence, check.sequence[1:]):
                    edge = memory.edges.get((domain.name, report.mode, root_family, pathology, a, b))
                    if edge and check.verdict != "stable":
                        edge.attack_failures += 1
                        if check.verdict == "broken_under_attack":
                            edge.attack_breaks += 1
                        if check.verdict in {"brittle", "broken_under_attack"}:
                            edge.planner_traps += 1

            if report.verdict in {"fragile", "champion_broken_under_attack"}:
                for op_name in result.path:
                    node = memory.nodes.get((domain.name, report.mode, root_family, pathology, op_name))
                    if node:
                        node.attack_failures += 1
                        if report.verdict == "champion_broken_under_attack":
                            node.attack_breaks += 1
                for a, b in zip(result.path, result.path[1:]):
                    edge = memory.edges.get((domain.name, report.mode, root_family, pathology, a, b))
                    if edge:
                        edge.attack_failures += 1
                        edge.planner_traps += 1
                        if report.verdict == "champion_broken_under_attack":
                            edge.attack_breaks += 1

            if report.recovery and report.recovery.recovered:
                for op_name in report.recovery.path:
                    key = (domain.name, report.mode, root_family, pathology, op_name)
                    card = memory.cards.setdefault(key, PatternCard(domain.name, op_name, report.mode, root_family, pathology))
                    card.support += 1
                    card.proof_gain.append(max(0.0, report.champion_hidden_scalar - report.recovery.hidden_scalar))
                    card.hidden_gain.append(max(0.0, report.champion_attack_scalar - report.recovery.attack_scalar))
                    card.hidden_passes += 1
                    node = memory.nodes.setdefault((domain.name, report.mode, root_family, pathology, op_name), MotifNode(domain.name, op_name, report.mode, root_family, pathology))
                    node.support += 1
                    node.proof_gain.append(max(0.0, report.champion_hidden_scalar - report.recovery.hidden_scalar))
                    node.hidden_gain.append(max(0.0, report.champion_attack_scalar - report.recovery.attack_scalar))
                for a, b in zip(report.recovery.path, report.recovery.path[1:]):
                    edge = memory.edges.setdefault((domain.name, report.mode, root_family, pathology, a, b), MotifEdge(domain.name, report.mode, root_family, pathology, a, b))
                    edge.support += 1
                    edge.proof_gain.append(max(0.0, report.champion_hidden_scalar - report.recovery.hidden_scalar))
                    edge.hidden_gain.append(max(0.0, report.champion_attack_scalar - report.recovery.attack_scalar))
                if len(report.recovery.path) >= 2:
                    seq = tuple(report.recovery.path)
                    if seq not in memory.promoted_sequences[context4]:
                        memory.promoted_sequences[context4].append(seq)

            if report.verdict in {"fragile", "champion_broken_under_attack"}:
                for flag in report.planner_flags:
                    memory.record_hypothesis(flag, "wrong")
            elif report.verdict == "stressed_but_survives":
                for flag in report.planner_flags:
                    memory.record_hypothesis(flag, "unresolved")
            else:
                for flag in report.planner_flags:
                    memory.record_hypothesis(flag, "confirmed")

        for key, ops in list(memory.promoted_ops.items()):
            survivors: List[DomainOperator[Any]] = []
            for op in ops:
                card = memory.cards.get((key[0], key[1], key[2], key[3], op.name))
                node = memory.nodes.get((key[0], key[1], key[2], key[3], op.name))
                fail_rate = 0.0
                break_rate = 0.0
                if card and card.attack_support:
                    fail_rate = max(fail_rate, card.attack_failures / max(1, card.attack_support))
                    break_rate = max(break_rate, card.attack_breaks / max(1, card.attack_support))
                if node and node.support:
                    fail_rate = max(fail_rate, node.attack_failures / max(1, node.support))
                    break_rate = max(break_rate, node.attack_breaks / max(1, node.support))
                if break_rate >= 0.35 or fail_rate >= 0.6:
                    if card:
                        _append_unique_antipattern(memory, card)
                    continue
                survivors.append(op)
            memory.promoted_ops[key] = survivors

        for key, seqs in list(memory.promoted_sequences.items()):
            kept: List[Tuple[str, ...]] = []
            for seq in seqs:
                seq_card = memory.cards.get((key[0], key[1], key[2], key[3], "+".join(seq)))
                edge_scores = []
                for a, b in zip(seq, seq[1:]):
                    edge = memory.edges.get((key[0], key[1], key[2], key[3], a, b))
                    if edge:
                        fail_rate = edge.attack_failures / max(1, edge.support)
                        break_rate = edge.attack_breaks / max(1, edge.support)
                        edge_scores.append((fail_rate, break_rate, edge.confidence))
                seq_fail = seq_card.attack_failures / max(1, seq_card.attack_support) if seq_card and seq_card.attack_support else 0.0
                seq_break = seq_card.attack_breaks / max(1, seq_card.attack_support) if seq_card and seq_card.attack_support else 0.0
                if seq_break >= 0.35 or seq_fail >= 0.6:
                    if seq_card:
                        _append_unique_antipattern(memory, seq_card)
                    continue
                if edge_scores and (max(x[1] for x in edge_scores) >= 0.4 or max(x[0] for x in edge_scores) >= 0.65):
                    continue
                kept.append(seq)
            memory.promoted_sequences[key] = kept

def run_domain(domain: DomainPack[Any], config: Optional[EngineConfig] = None):
    engine = ForgeEngine(domain, config or EngineConfig())
    return engine, *engine.run()


def run_engine_demo() -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    reports: List[str] = []

    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig())
    bundles["sorting"] = {
        "engine": sorting_engine,
        "results": sorting_results,
        "memory": sorting_memory,
        "forensics": sorting_forensics,
        "scores": sorting_scores,
    }

    path_cfg = EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=2, base_seed=41)
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), path_cfg)
    bundles["pathfinding"] = {
        "engine": path_engine,
        "results": path_results,
        "memory": path_memory,
        "forensics": path_forensics,
        "scores": path_scores,
    }

    counter = CounterForge(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)

    for domain_name in ("sorting", "pathfinding"):
        pack = bundles[domain_name]
        reports.append(pack["engine"].build_report(pack["results"], pack["memory"], pack["forensics"], pack["scores"]))

    print("SMART FORGE 1.7 core")
    print("counter-forge escalated with sequence attacks, planner-belief attacks, and recovery passes")
    for domain_name in ("sorting", "pathfinding"):
        pack = bundles[domain_name]
        print(f"\n[{domain_name}] meta-rounds")
        for k, v in pack["scores"].items():
            print(f"  {k}: {v:.4f}")
        print("  champions")
        for r in sorted(pack["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"    [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f} worst_hidden {r.hidden.worst_cost:.4f}")
        print("  counter-forge")
        for item in counter_reports[domain_name]:
            print(f"    {item.mode}/{item.root_name}: verdict={item.verdict} degradation={item.degradation_ratio:.3f} weak={','.join(item.weak_kinds)}")

    combined = "\n\n---\n\n".join(reports + [counter_md])
    with open("/mnt/data/smart_forge_1_6_core_report.md", "w", encoding="utf-8") as f:
        f.write(combined)
    bundles["counter_forge"] = {"reports": counter_reports, "report_markdown": counter_md}
    return bundles




import re
from inspect import signature
from pathlib import Path

@dataclass(frozen=True)
class QuickOperator(Generic[T]):
    name: str
    fn: Callable[[T], T]
    tags: Tuple[str, ...] = ()


@dataclass
class SimpleTask(Generic[T]):
    name: str
    initial: T
    score: Callable[..., float]
    operators: Sequence[QuickOperator[T]] | Dict[str, Callable[[T], T]]
    mode: str = "optimize"
    validate: Optional[Callable[..., bool]] = None
    cases: Optional[Sequence[Any]] = None
    proof_cases: Optional[Sequence[Any]] = None
    hidden_cases: Optional[Sequence[Any]] = None
    counter_cases: Optional[Sequence[Any]] = None
    summarize: Optional[Callable[[T], str]] = None
    classify: Optional[Callable[[T], str]] = None
    pathology: Optional[Callable[[T, str], str]] = None
    complexity: Optional[Callable[[T], int]] = None
    serialize: Optional[Callable[[T], str]] = None
    edit_distance: Optional[Callable[[Optional[T], T], int]] = None
    policy_hints: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TaskRunResult(Generic[T]):
    task: SimpleTask[T]
    champion: T
    root: T
    planner: EvalMetrics
    proof: EvalMetrics
    hidden: EvalMetrics
    path: List[str]
    report_markdown: str
    bundle: Dict[str, Any]
    counter: Optional[EvalMetrics] = None


class SimpleTaskDomainPack(DomainPack[T]):
    def __init__(self, task: SimpleTask[T]):
        self.task = task
        self.name = task.name
        if isinstance(task.operators, dict):
            self._operators = {name: DomainOperator(name, fn) for name, fn in task.operators.items()}
        else:
            self._operators = {op.name: DomainOperator(op.name, op.fn, op.tags) for op in task.operators}

    def _call(self, fn: Callable[..., Any], candidate: T, case: Any = None, fallback: Any = None) -> Any:
        if fn is None:
            return fallback
        try:
            params = len(signature(fn).parameters)
        except (TypeError, ValueError):
            params = 2
        if params <= 1:
            return fn(candidate)
        return fn(candidate, case)

    def _normalize_cases(self, seq: Optional[Sequence[Any]], default_name: str) -> Arena[Any]:
        seq = list(seq or [None])
        norm: List[Tuple[str, Any]] = []
        for idx, item in enumerate(seq):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                norm.append((item[0], item[1]))
            else:
                norm.append((f"{default_name}_{idx}", item))
        return Arena(default_name, norm)

    def root_sets(self) -> Dict[str, Dict[str, T]]:
        return {self.task.mode: {self.task.name: self.task.initial}}

    def candidate_hash(self, candidate: T) -> str:
        raw = self.task.serialize(candidate) if self.task.serialize else repr(candidate)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def summarize(self, candidate: T) -> str:
        return self.task.summarize(candidate) if self.task.summarize else repr(candidate)

    def complexity(self, candidate: T) -> int:
        if self.task.complexity:
            return max(1, int(self.task.complexity(candidate)))
        if isinstance(candidate, (str, bytes, list, tuple, dict, set)):
            return max(1, len(candidate))
        return 1

    def classify_family(self, candidate: T) -> str:
        return self.task.classify(candidate) if self.task.classify else "generic"

    def detect_pathology(self, candidate: T, mode: str) -> str:
        return self.task.pathology(candidate, mode) if self.task.pathology else "generic"

    def structural_edit_distance(self, root: Optional[T], candidate: T) -> int:
        if self.task.edit_distance:
            return max(0, int(self.task.edit_distance(root, candidate)))
        if root is None:
            return self.complexity(candidate)
        return 0 if self.candidate_hash(root) == self.candidate_hash(candidate) else abs(self.complexity(candidate) - self.complexity(root)) + 1

    def operator_bank(self, mode: str, root: T, promoted: Sequence[DomainOperator[T]]) -> List[DomainOperator[T]]:
        out: List[DomainOperator[T]] = []
        seen = set()
        for op in list(promoted) + list(self._operators.values()):
            if op.name in seen:
                continue
            seen.add(op.name)
            out.append(op)
        return out

    def operator_lookup(self) -> Dict[str, DomainOperator[T]]:
        return self._operators

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[Any], Arena[Any], Arena[Any]]:
        planner = self._normalize_cases(self.task.cases, f"{self.task.name}_{mode}_planner")
        proof = self._normalize_cases(self.task.proof_cases or self.task.cases, f"{self.task.name}_{mode}_proof")
        hidden = self._normalize_cases(self.task.hidden_cases or self.task.proof_cases or self.task.cases, f"{self.task.name}_{mode}_hidden")
        return planner, proof, hidden

    def evaluate(self, candidate: T, arena: Arena[Any], mode: str, root: Optional[T] = None) -> EvalMetrics:
        failures: Dict[str, int] = defaultdict(int)
        costs: List[float] = []
        correct = 0
        for _, case in arena.cases:
            try:
                valid = True if self.task.validate is None else bool(self._call(self.task.validate, candidate, case, True))
            except Exception:
                valid = False
            if not valid:
                failures["invalid"] += 1
                continue
            try:
                raw_cost = float(self._call(self.task.score, candidate, case, float("inf")))
            except Exception:
                failures["score_error"] += 1
                continue
            if not math.isfinite(raw_cost):
                failures["nonfinite"] += 1
                continue
            costs.append(raw_cost)
            correct += 1
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = float(self.complexity(candidate))
        edit_distance = float(self.structural_edit_distance(root, candidate) if root is not None else 0)
        failure_penalty = float((len(arena.cases) - correct) * 1000)
        scalar = failure_penalty
        if mode == "repair":
            scalar += avg_cost * 120.0 + worst_cost * 20.0 + edit_distance * 0.8 + complexity * 0.15
        elif mode == "harden":
            scalar += worst_cost * 120.0 + avg_cost * 35.0 + edit_distance * 0.25 + complexity * 0.1
        else:
            scalar += avg_cost * 100.0 + worst_cost * 20.0 + edit_distance * 0.2 + complexity * 0.1
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, dict(failures))

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        return self.task.policy_hints.get(pathology, {})

    def rediscovery_verdict(self, root: T, champion: T) -> str:
        if self.candidate_hash(root) == self.candidate_hash(champion):
            return "root retained"
        if self.classify_family(root) == self.classify_family(champion):
            return "refined same family"
        return "family shift"

    def suite_attribution(self, root: T, champion: T, arena: Arena[Any], mode: str) -> List[Tuple[str, float, float]]:
        rows: List[Tuple[str, float, float]] = []
        for label, case in arena.cases:
            try:
                root_ok = True if self.task.validate is None else bool(self._call(self.task.validate, root, case, True))
                champ_ok = True if self.task.validate is None else bool(self._call(self.task.validate, champion, case, True))
                root_cost = float(self._call(self.task.score, root, case, float("inf"))) if root_ok else float("inf")
                champ_cost = float(self._call(self.task.score, champion, case, float("inf"))) if champ_ok else float("inf")
            except Exception:
                continue
            cost_delta = 0.0 if (not math.isfinite(root_cost) or not math.isfinite(champ_cost)) else root_cost - champ_cost
            corr_delta = (1.0 if champ_ok else 0.0) - (1.0 if root_ok else 0.0)
            rows.append((label, cost_delta, corr_delta))
        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return rows


def run_task(task: SimpleTask[T], config: Optional[EngineConfig] = None, use_counter: bool = True) -> TaskRunResult[T]:
    domain = SimpleTaskDomainPack(task)
    engine = ForgeEngine(domain, config or EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=2, base_seed=73))
    results, memory, forensics, scores = engine.run()
    result = results[0]
    report = engine.build_report(results, memory, forensics, scores)
    counter_metrics = None
    if use_counter and task.counter_cases:
        attack_arena = domain._normalize_cases(task.counter_cases, f"{task.name}_{task.mode}_counter")
        counter_metrics = domain.evaluate(result.champion, attack_arena, task.mode, task.initial)
        report += "\n\n## Simple task counter check\n"
        report += f"- counter correctness: {counter_metrics.correct}/{counter_metrics.total}\n"
        report += f"- counter avg_cost: {counter_metrics.avg_cost:.4f}\n"
        report += f"- counter worst_cost: {counter_metrics.worst_cost:.4f}\n"
    bundle = {"engine": engine, "results": results, "memory": memory, "forensics": forensics, "scores": scores, "domain": domain}
    return TaskRunResult(task, result.champion, task.initial, result.planner, result.proof, result.hidden, result.path, report, bundle, counter_metrics)


def optimize(*, initial: T, operators: Sequence[QuickOperator[T]] | Dict[str, Callable[[T], T]], score: Callable[..., float], validate: Optional[Callable[..., bool]] = None, cases: Optional[Sequence[Any]] = None, proof_cases: Optional[Sequence[Any]] = None, hidden_cases: Optional[Sequence[Any]] = None, counter_cases: Optional[Sequence[Any]] = None, name: str = "task", summarize: Optional[Callable[[T], str]] = None, classify: Optional[Callable[[T], str]] = None, pathology: Optional[Callable[[T, str], str]] = None, complexity: Optional[Callable[[T], int]] = None, serialize: Optional[Callable[[T], str]] = None, edit_distance: Optional[Callable[[Optional[T], T], int]] = None, policy_hints: Optional[Dict[str, Dict[str, float]]] = None, config: Optional[EngineConfig] = None) -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="optimize", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    return run_task(task, config=config)


def repair(*, initial: T, operators: Sequence[QuickOperator[T]] | Dict[str, Callable[[T], T]], score: Callable[..., float], validate: Callable[..., bool], cases: Optional[Sequence[Any]] = None, proof_cases: Optional[Sequence[Any]] = None, hidden_cases: Optional[Sequence[Any]] = None, counter_cases: Optional[Sequence[Any]] = None, name: str = "task", summarize: Optional[Callable[[T], str]] = None, classify: Optional[Callable[[T], str]] = None, pathology: Optional[Callable[[T, str], str]] = None, complexity: Optional[Callable[[T], int]] = None, serialize: Optional[Callable[[T], str]] = None, edit_distance: Optional[Callable[[Optional[T], T], int]] = None, policy_hints: Optional[Dict[str, Dict[str, float]]] = None, config: Optional[EngineConfig] = None) -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="repair", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    return run_task(task, config=config)


def harden(*, initial: T, operators: Sequence[QuickOperator[T]] | Dict[str, Callable[[T], T]], score: Callable[..., float], validate: Callable[..., bool], cases: Optional[Sequence[Any]] = None, proof_cases: Optional[Sequence[Any]] = None, hidden_cases: Optional[Sequence[Any]] = None, counter_cases: Optional[Sequence[Any]] = None, name: str = "task", summarize: Optional[Callable[[T], str]] = None, classify: Optional[Callable[[T], str]] = None, pathology: Optional[Callable[[T, str], str]] = None, complexity: Optional[Callable[[T], int]] = None, serialize: Optional[Callable[[T], str]] = None, edit_distance: Optional[Callable[[Optional[T], T], int]] = None, policy_hints: Optional[Dict[str, Dict[str, float]]] = None, config: Optional[EngineConfig] = None) -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="harden", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    return run_task(task, config=config)


def _dedupe_regex_alternation(pattern: str) -> str:
    match = re.fullmatch(r"\(\?:([^()]+)\)\+", pattern)
    if not match:
        return pattern
    parts = match.group(1).split("|")
    deduped: List[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return f"(?:{'|'.join(deduped)})+"


def _sort_regex_alternation(pattern: str) -> str:
    match = re.fullmatch(r"\(\?:([^()]+)\)\+", pattern)
    if not match:
        return pattern
    parts = sorted(match.group(1).split("|"), key=lambda s: (len(s), s))
    return f"(?:{'|'.join(parts)})+"


def _compress_single_char_classes(pattern: str) -> str:
    return pattern.replace("(?:a|b)", "[ab]").replace("(?:b|a)", "[ab]")


def demo_regex_task() -> TaskRunResult[str]:
    initial = "(?:foo|bar|foo|bar)+"
    samples = [("txt0", "foo"), ("txt1", "barbar"), ("txt2", "foobar"), ("txt3", "baz"), ("txt4", "barfoofoo"), ("txt5", "quxbar")]
    hidden = [("hid0", "foofoo"), ("hid1", "barfoo"), ("hid2", ""), ("hid3", "foobarbar"), ("hid4", "zzz")]
    counter = [(f"ctr{i}", s * 2) for i, (_, s) in enumerate(hidden)]
    baseline = re.compile(initial)

    def validate(pattern: str, text: str) -> bool:
        try:
            candidate = re.compile(pattern)
        except re.error:
            return False
        return bool(candidate.fullmatch(text)) == bool(baseline.fullmatch(text))

    def score(pattern: str, text: str) -> float:
        return float(len(pattern))

    def classify(pattern: str) -> str:
        return "regex_alt" if "|" in pattern else "regex_compact"

    return optimize(
        name="regex_opt",
        initial=initial,
        operators={"dedupe_alt": _dedupe_regex_alternation, "sort_alt": _sort_regex_alternation, "compress_class": _compress_single_char_classes},
        score=score,
        validate=validate,
        cases=samples,
        hidden_cases=hidden,
        counter_cases=counter,
        classify=classify,
        complexity=len,
        serialize=str,
        summarize=lambda p: p,
        config=EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=2, base_seed=101),
    )


def build_api_section(regex_result: TaskRunResult[str]) -> str:
    return "\n".join([
        "# Smart Forge 1.7 adversarial escalation and recovery layer",
        "",
        "## What changed",
        "- Keeps the simple task API with optimize(...), repair(...), harden(...).",
        "- Added SimpleTask so users do not need a full DomainPack for every small task.",
        "- Added a generic SimpleTaskDomainPack wrapper that preserves thinking mode, pattern mining, reforge, and forensics.",
        "- Added a regex optimization demo using the high-level API.",
        "",
        "## Regex demo result",
        f"- root: `{regex_result.root}`",
        f"- champion: `{regex_result.champion}`",
        f"- proof: {regex_result.proof.correct}/{regex_result.proof.total}",
        f"- hidden: {regex_result.hidden.correct}/{regex_result.hidden.total}",
        f"- counter: {regex_result.counter.correct if regex_result.counter else 0}/{regex_result.counter.total if regex_result.counter else 0}",
        f"- path: {' -> '.join(regex_result.path) if regex_result.path else '(root kept)'}",
        "",
        "## Simple API example",
        "```python",
        "from smart_forge import optimize",
        "",
        "result = optimize(",
        "    name='regex_opt',",
        "    initial='(?:foo|bar|foo|bar)+',",
        "    operators={",
        "        'dedupe_alt': dedupe_alt,",
        "        'sort_alt': sort_alt,",
        "        'compress_class': compress_class,",
        "    },",
        "    score=score_regex,",
        "    validate=validate_regex,",
        "    cases=samples,",
        "    hidden_cases=hidden,",
        ")",
        "```",
        "",
        "## Why this matters",
        "- Small custom tasks no longer need a full 600-line domain implementation.",
        "- Power users can still use DomainPack directly.",
        "- The easy API keeps the engine pressure, planner, and forensics instead of collapsing into random local search.",
    ])


def run_demo() -> Dict[str, Any]:
    bundles = run_engine_demo()
    regex_result = demo_regex_task()
    bundles["simple_api_demo"] = regex_result
    api_md = build_api_section(regex_result)
    existing = Path("/mnt/data/smart_forge_1_7_core_report.md").read_text(encoding="utf-8") if Path("/mnt/data/smart_forge_1_7_core_report.md").exists() else ""
    combined = existing + "\n\n---\n\n" + api_md + "\n\n---\n\n" + regex_result.report_markdown
    Path("/mnt/data/smart_forge_1_7_report.md").write_text(combined, encoding="utf-8")
    print("\nSMART FORGE 1.7")
    print("counter-forge escalation and recovery loop hardened")
    print(f"regex demo: root={regex_result.root} champion={regex_result.champion} proof={regex_result.proof.correct}/{regex_result.proof.total} hidden={regex_result.hidden.correct}/{regex_result.hidden.total}")
    return bundles



__version__ = "o1"

@dataclass
class ForgePackageInfo:
    version: str
    script_name: str
    domains: List[str]
    commands: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "script_name": self.script_name,
            "domains": list(self.domains),
            "commands": list(self.commands),
        }


def package_info() -> ForgePackageInfo:
    return ForgePackageInfo(
        version=__version__,
        script_name="smart_forge_1_7.py",
        domains=["sorting", "pathfinding", "simple_task"],
        commands=["demo", "regex-demo", "list-domains", "write-examples", "run-regex-config"],
    )


def _metrics_to_dict(m: Optional[EvalMetrics]) -> Optional[Dict[str, Any]]:
    if m is None:
        return None
    return {
        "correct": m.correct,
        "total": m.total,
        "avg_cost": m.avg_cost,
        "worst_cost": m.worst_cost,
        "complexity": m.complexity,
        "scalar": m.scalar,
        "failures": dict(m.failures),
    }


def _mode_result_to_dict(r: Any) -> Dict[str, Any]:
    out = {
        "mode": r.mode,
        "root_name": r.root_name,
        "root_summary": getattr(r, 'root', None) if isinstance(getattr(r, 'root', None), (str, int, float, bool, type(None))) else None,
        "champion_summary": getattr(r, 'champion', None) if isinstance(getattr(r, 'champion', None), (str, int, float, bool, type(None))) else None,
        "path": list(getattr(r, 'path', [])),
        "planner": _metrics_to_dict(getattr(r, 'planner', None)),
        "proof": _metrics_to_dict(getattr(r, 'proof', None)),
        "hidden": _metrics_to_dict(getattr(r, 'hidden', None)),
        "reforge_gain": getattr(r, 'reforge_gain', 0.0),
    }
    if hasattr(r, 'root'):
        out["root_repr"] = repr(r.root)
    if hasattr(r, 'champion'):
        out["champion_repr"] = repr(r.champion)
    return out


def _task_result_to_dict(r: TaskRunResult[Any]) -> Dict[str, Any]:
    return {
        "task": r.task.name,
        "mode": r.task.mode,
        "root": repr(r.root),
        "champion": repr(r.champion),
        "path": list(r.path),
        "planner": _metrics_to_dict(r.planner),
        "proof": _metrics_to_dict(r.proof),
        "hidden": _metrics_to_dict(r.hidden),
        "counter": _metrics_to_dict(r.counter),
    }


def _counter_report_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "domain": item.domain_name,
        "mode": item.mode,
        "root_name": item.root_name,
        "verdict": item.verdict,
        "weak_kinds": list(item.weak_kinds),
        "degradation_ratio": item.degradation_ratio,
        "planner_flags": list(item.planner_flags),
        "promoted_checks": [
            {"op_name": c.op_name, "verdict": c.verdict, "degradation_ratio": (c.attack_scalar / max(1e-9, c.hidden_scalar)) if c.hidden_scalar else None}
            for c in item.promoted_checks
        ],
        "sequence_checks": [
            {"sequence": list(c.sequence), "verdict": c.verdict, "degradation_ratio": (c.attack_scalar / max(1e-9, c.hidden_scalar)) if c.hidden_scalar else None}
            for c in item.sequence_checks
        ],
        "recovery": None if item.recovery is None else {
            "attempted": item.recovery.attempted,
            "recovered": item.recovery.recovered,
            "path": list(item.recovery.path),
            "gain": item.recovery.gain,
            "verdict": item.recovery.verdict,
        },
    }


def build_release_report(bundles: Dict[str, Any], regex_result: TaskRunResult[str]) -> str:
    lines: List[str] = []
    info = package_info()
    lines.append(f"# Smart Forge {info.version}")
    lines.append("")
    lines.append("## What changed in 1.7")
    lines.append("- Counter-Forge now attacks motif sequences and planner beliefs, not just champions and single promoted operators.")
    lines.append("- Added post-attack recovery passes so fragile champions can be challenged and then locally reforged into sturdier descendants.")
    lines.append("- Attack feedback now demotes brittle promoted sequences and poisons misleading transition edges more aggressively.")
    lines.append("- The standalone CLI and simple task API remain intact.")
    lines.append("")
    lines.append("## Public commands")
    for cmd in info.commands:
        lines.append(f"- `{cmd}`")
    lines.append("")
    lines.append("## Standalone usage")
    lines.append("```bash")
    lines.append("python smart_forge_1_7.py demo")
    lines.append("python smart_forge_1_7.py regex-demo")
    lines.append("python smart_forge_1_7.py list-domains")
    lines.append("python smart_forge_1_7.py write-examples --dir examples")
    lines.append("python smart_forge_1_7.py run-regex-config examples/regex_optimize.json")
    lines.append("```")
    lines.append("")
    lines.append("## Regex demo")
    lines.append(f"- root: `{regex_result.root}`")
    lines.append(f"- champion: `{regex_result.champion}`")
    lines.append(f"- proof: {regex_result.proof.correct}/{regex_result.proof.total}")
    lines.append(f"- hidden: {regex_result.hidden.correct}/{regex_result.hidden.total}")
    if regex_result.counter:
        lines.append(f"- counter: {regex_result.counter.correct}/{regex_result.counter.total}")
    lines.append(f"- path: {' -> '.join(regex_result.path) if regex_result.path else '(root kept)'}")
    lines.append("")
    lines.append("## Domain bundles")
    for domain_name in ("sorting", "pathfinding"):
        pack = bundles[domain_name]
        lines.append(f"### {domain_name}")
        lines.append(f"- meta-round scores: {pack['scores']}")
        for r in sorted(pack['results'], key=lambda x: (x.mode, x.root_name)):
            lines.append(
                f"- [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total}, hidden {r.hidden.correct}/{r.hidden.total}, avg {r.proof.avg_cost:.4f}, hidden_worst {r.hidden.worst_cost:.4f}"
            )
        for cr in bundles["counter_forge"]["reports"].get(domain_name, [])[:6]:
            lines.append(f"- counter {cr.mode}/{cr.root_name}: verdict={cr.verdict}, degradation={cr.degradation_ratio:.3f}")
            if cr.recovery:
                lines.append(f"  - recovery: {cr.recovery.verdict}, gain={cr.recovery.gain:.4f}, path={' -> '.join(cr.recovery.path) if cr.recovery.path else '(none)'}")
        lines.append("")
    lines.append("## Why this matters")
    lines.append("- Forge no longer trusts strong motif sequences just because they once won.")
    lines.append("- Counter-Forge can now hunt sequence-level brittleness and challenge planner optimism directly.")
    lines.append("- Recovery passes create a build-break-recover loop, which is much closer to a real adaptive engine.")
    lines.append("")
    lines.append("## JSON shape")
    lines.append("```json")
    lines.append(json.dumps({
        "package": package_info().to_dict(),
        "regex_demo": _task_result_to_dict(regex_result),
        "counter_forge": {
            name: [_counter_report_to_dict(r) for r in bundles["counter_forge"]["reports"][name]]
            for name in bundles["counter_forge"]["reports"]
        },
    }, indent=2)[:2000])
    lines.append("```")
    return "\n".join(lines)


def write_json(path: str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def export_example_configs(target_dir: str) -> List[str]:
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "regex_optimize.json": {
            "name": "regex_opt",
            "mode": "optimize",
            "initial": "(?:foo|bar|foo|bar)+",
            "operators": ["dedupe_alt", "sort_alt", "compress_class"],
            "cases": ["foo", "bar", "foobar", "barfoo", "foofoo"],
            "hidden_cases": ["foo", "bar", "foobar", "barbar", "foofoo", "barfoofoo"],
            "counter_cases": ["foo", "bar", "foobarbar", "barfoofoofoo"],
            "max_depth": 2,
            "base_beam_width": 4,
            "meta_rounds": 2,
            "base_seed": 101,
        },
        "regex_repair.json": {
            "name": "regex_repair",
            "mode": "repair",
            "initial": "(?:foo|foo|bar)+",
            "operators": ["dedupe_alt", "sort_alt", "compress_class"],
            "cases": ["foo", "bar", "foofoo", "barbar"],
            "hidden_cases": ["foo", "bar", "foobar", "barfoo"],
            "counter_cases": ["foofoofoo", "barbarbar"],
            "max_depth": 2,
            "base_beam_width": 4,
            "meta_rounds": 2,
            "base_seed": 102,
        },
    }
    written = []
    for name, payload in examples.items():
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        written.append(str(path))
    return written


def _regex_operator_library() -> Dict[str, Callable[[str], str]]:
    return {
        "dedupe_alt": _dedupe_regex_alternation,
        "sort_alt": _sort_regex_alternation,
        "compress_class": _compress_single_char_classes,
    }


def _run_regex_config(config_path: str) -> TaskRunResult[str]:
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    ops = _regex_operator_library()
    chosen = {name: ops[name] for name in cfg["operators"]}
    def validate(pattern: str, text: str) -> bool:
        try:
            baseline = re.compile(cfg["initial"])
            candidate = re.compile(pattern)
        except re.error:
            return False
        return bool(candidate.fullmatch(text)) == bool(baseline.fullmatch(text))
    def score(pattern: str, text: str) -> float:
        return float(len(pattern))
    def classify(pattern: str) -> str:
        return "regex_alt" if "|" in pattern else "regex_compact"
    task = SimpleTask(
        name=cfg.get("name", "regex_task"),
        initial=cfg["initial"],
        score=score,
        operators=chosen,
        mode=cfg.get("mode", "optimize"),
        validate=validate,
        cases=cfg.get("cases"),
        hidden_cases=cfg.get("hidden_cases"),
        counter_cases=cfg.get("counter_cases"),
        classify=classify,
        complexity=len,
        serialize=str,
        summarize=lambda p: p,
    )
    return run_task(
        task,
        EngineConfig(
            max_depth=int(cfg.get("max_depth", 2)),
            base_beam_width=int(cfg.get("base_beam_width", 4)),
            meta_rounds=int(cfg.get("meta_rounds", 2)),
            base_seed=int(cfg.get("base_seed", 101)),
        ),
    )


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles = run_engine_demo()
    regex_result = demo_regex_task()
    bundles["simple_api_demo"] = regex_result
    release_report = build_release_report(bundles, regex_result)
    if write_report:
        Path("/mnt/data/smart_forge_1_7_report.md").write_text(release_report, encoding="utf-8")
    print("\nSMART FORGE 1.7")
    print("counter-forge escalation and recovery loop hardened")
    print(f"regex demo: root={regex_result.root} champion={regex_result.champion} proof={regex_result.proof.correct}/{regex_result.proof.total} hidden={regex_result.hidden.correct}/{regex_result.hidden.total}")
    return bundles


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smart_forge_1_7", description="Smart Forge 1.7 standalone engine and CLI")
    sub = parser.add_subparsers(dest="command", required=False)

    demo = sub.add_parser("demo", help="run full engine demo with built-in domains and write the markdown report")
    demo.add_argument("--json-out", dest="json_out")
    demo.add_argument("--stdout-json", action="store_true")

    rx = sub.add_parser("regex-demo", help="run the simple regex optimization demo only")
    rx.add_argument("--json-out", dest="json_out")
    rx.add_argument("--stdout-json", action="store_true")

    sub.add_parser("list-domains", help="show built-in domains and commands")

    ex = sub.add_parser("write-examples", help="write example config files for the simple regex task runner")
    ex.add_argument("--dir", default="examples")

    rc = sub.add_parser("run-regex-config", help="run a simple regex task from JSON config")
    rc.add_argument("config")
    rc.add_argument("--json-out", dest="json_out")
    rc.add_argument("--stdout-json", action="store_true")

    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "demo"

    if command == "demo":
        bundles = run_release_demo(write_report=True)
        payload = {
            "package": package_info().to_dict(),
            "domains": {
                name: {
                    "scores": bundle["scores"],
                    "results": [_mode_result_to_dict(r) for r in bundle["results"]],
                }
                for name, bundle in bundles.items() if name in {"sorting", "pathfinding"}
            },
            "regex_demo": _task_result_to_dict(bundles["simple_api_demo"]),
            "counter_forge": {
                name: [_counter_report_to_dict(r) for r in bundles["counter_forge"]["reports"][name]]
                for name in bundles["counter_forge"]["reports"]
            },
        }
        if getattr(args, "json_out", None):
            write_json(args.json_out, payload)
        if getattr(args, "stdout_json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if command == "regex-demo":
        result = demo_regex_task()
        payload = {"package": package_info().to_dict(), "result": _task_result_to_dict(result)}
        print(f"SMART FORGE 1.7 regex-demo\nroot={result.root}\nchampion={result.champion}\nproof={result.proof.correct}/{result.proof.total}\nhidden={result.hidden.correct}/{result.hidden.total}")
        if getattr(args, "json_out", None):
            write_json(args.json_out, payload)
        if getattr(args, "stdout_json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if command == "list-domains":
        info = package_info()
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
        return 0

    if command == "write-examples":
        written = export_example_configs(args.dir)
        print("wrote example configs:")
        for path in written:
            print(path)
        return 0

    if command == "run-regex-config":
        result = _run_regex_config(args.config)
        payload = {"package": package_info().to_dict(), "result": _task_result_to_dict(result)}
        print(f"SMART FORGE 1.7 run-regex-config\nroot={result.root}\nchampion={result.champion}\nproof={result.proof.correct}/{result.proof.total}\nhidden={result.hidden.correct}/{result.hidden.total}")
        if getattr(args, "json_out", None):
            write_json(args.json_out, payload)
        if getattr(args, "stdout_json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


# ===== Smart Forge 1.89 additions merged standalone =====

import argparse
import ast
import copy
import difflib
import hashlib
import inspect
import json
import math
import random
import re
import signal
import sys
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union, get_args, get_origin
T = TypeVar("T")

__version__ = "o1"


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ----------------------------- Regex proof domain -----------------------------


@dataclass(frozen=True)
class RegexCandidate:
    pattern: str

    def complexity(self) -> int:
        return len(self.pattern)


@dataclass(frozen=True)
class RegexCase:
    text: str
    expected: bool


class RegexDomainPack(DomainPack[RegexCandidate]):
    name = "regex"

    def __init__(self) -> None:
        self._ops = self._build_operators()

    def root_sets(self) -> Dict[str, Dict[str, RegexCandidate]]:
        return {
            "optimize": {
                "dup_alt": RegexCandidate(r"(?:foo|bar|foo|bar)+"),
                "nested_groups": RegexCandidate(r"(?:(?:foo)|(?:bar))+"),
            },
            "repair": {
                "accepts_empty": RegexCandidate(r"(?:foo|bar)*"),
                "overmatches": RegexCandidate(r"(?:foo|bar|baz)+"),
                "undermatches": RegexCandidate(r"(?:foo|baar)+"),
            },
            "harden": {
                "ambiguous_1": RegexCandidate(r"(?:foo|f(?:oo)|bar|ba(?:r))+"),
                "ambiguous_2": RegexCandidate(r"(?:(?:foo)|(?:bar)|(?:fo(?:o))|(?:ba(?:r)))+"),
                "ambiguous_3": RegexCandidate(r"(?:(?:foo)|(?:bar)|(?:foo(?:))|(?:bar(?:)))+"),
            },
        }

    def candidate_hash(self, candidate: RegexCandidate) -> str:
        return _stable_hash(candidate.pattern)

    def summarize(self, candidate: RegexCandidate) -> str:
        return candidate.pattern

    def complexity(self, candidate: RegexCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: RegexCandidate) -> str:
        p = candidate.pattern
        if "foo|bar|foo|bar" in p:
            return "dup_alt"
        if "?:(" in p or "(?:" in p and p.count("(?:") > 2:
            return "nested_alt"
        if "|baz" in p or "|baar" in p or p.endswith("*"):
            return "sloppy_spec"
        if "f(?:oo)" in p or "ba(?:r)" in p:
            return "ambiguous_alt"
        if p == r"(?:foo|bar)+":
            return "canonical"
        return "mixed"

    def detect_pathology(self, candidate: RegexCandidate, mode: str) -> str:
        p = candidate.pattern
        if p.endswith("*"):
            return "accepts_empty"
        if "|baz" in p:
            return "overmatch"
        if "|baar" in p:
            return "undermatch"
        if "f(?:oo)" in p or "ba(?:r)" in p:
            return "prefix_ambiguity"
        if p.count("(?:") > 2:
            return "nested_group"
        if p.count("|foo") or p.count("|bar") > 1:
            return "duplicate_alt"
        return "generic"

    def structural_edit_distance(self, root: Optional[RegexCandidate], candidate: RegexCandidate) -> int:
        if root is None:
            return len(candidate.pattern)
        a = root.pattern
        b = candidate.pattern
        prefix = 0
        for x, y in zip(a, b):
            if x == y:
                prefix += 1
            else:
                break
        return (len(a) - prefix) + (len(b) - prefix)

    def _build_operators(self) -> Dict[str, DomainOperator[RegexCandidate]]:
        return {
            "dedupe_alt": DomainOperator("dedupe_alt", lambda c: RegexCandidate(self._dedupe_alt(c.pattern))),
            "flatten_groups": DomainOperator("flatten_groups", lambda c: RegexCandidate(self._flatten_groups(c.pattern))),
            "remove_empty": DomainOperator("remove_empty", lambda c: RegexCandidate(c.pattern.replace("(?:)", ""))),
            "canonicalize": DomainOperator("canonicalize", lambda c: RegexCandidate(r"(?:foo|bar)+")),
            "repair_empty_plus": DomainOperator("repair_empty_plus", lambda c: RegexCandidate(c.pattern[:-1] + "+") if c.pattern.endswith("*") else c),
            "repair_remove_baz": DomainOperator("repair_remove_baz", lambda c: RegexCandidate(c.pattern.replace("|baz", ""))),
            "repair_baar_to_bar": DomainOperator("repair_baar_to_bar", lambda c: RegexCandidate(c.pattern.replace("baar", "bar"))),
            "sort_alt": DomainOperator("sort_alt", lambda c: RegexCandidate(self._sort_alt(c.pattern))),
            "disambiguate_alt": DomainOperator("disambiguate_alt", lambda c: RegexCandidate(self._disambiguate(c.pattern))),
            "trim_redundant_suffix": DomainOperator("trim_redundant_suffix", lambda c: RegexCandidate(c.pattern.replace("(?:foo(?:))", "(?:foo)").replace("(?:bar(?:))", "(?:bar)"))),
        }

    def operator_lookup(self) -> Dict[str, DomainOperator[RegexCandidate]]:
        return self._ops

    def operator_bank(self, mode: str, root: RegexCandidate, promoted: Sequence[DomainOperator[RegexCandidate]]) -> List[DomainOperator[RegexCandidate]]:
        names = {
            "optimize": ["dedupe_alt", "flatten_groups", "sort_alt", "canonicalize"],
            "repair": ["repair_empty_plus", "repair_remove_baz", "repair_baar_to_bar", "dedupe_alt", "canonicalize", "flatten_groups"],
            "harden": ["disambiguate_alt", "trim_redundant_suffix", "canonicalize", "sort_alt", "flatten_groups"],
        }[mode]
        out: List[DomainOperator[RegexCandidate]] = []
        seen = set()
        for op in list(promoted) + [self._ops[n] for n in names]:
            if op.name not in seen:
                out.append(op)
                seen.add(op.name)
        return out

    def _target_expected(self, text: str) -> bool:
        if not text:
            return False
        i = 0
        while i < len(text):
            if text.startswith("foo", i):
                i += 3
            elif text.startswith("bar", i):
                i += 3
            else:
                return False
        return True

    def _base_cases(self, kinds: Sequence[str], seed: int, count: int) -> List[Tuple[str, RegexCase]]:
        rng = random.Random(seed)
        out: List[Tuple[str, RegexCase]] = []
        for _ in range(count):
            kind = rng.choice(list(kinds))
            if kind == "valid_short":
                txt = "".join(rng.choice(["foo", "bar"]) for _ in range(rng.randint(1, 3)))
            elif kind == "valid_long":
                txt = "".join(rng.choice(["foo", "bar"]) for _ in range(rng.randint(4, 12)))
            elif kind == "invalid_empty":
                txt = ""
            elif kind == "invalid_miss":
                txt = "".join(rng.choice(["foo", "bar", "baz", "baar"]) for _ in range(rng.randint(1, 4)))
                if self._target_expected(txt):
                    txt += "x"
            elif kind == "near_miss":
                txt = "".join(rng.choice(["foo", "bar"]) for _ in range(rng.randint(2, 6)))
                txt = txt[:-1] + rng.choice(["x", "z", "q"])
            elif kind == "attack_repetition":
                token = rng.choice(["foo", "bar"])
                txt = token * rng.randint(8, 24)
                if rng.random() < 0.5:
                    txt += "x"
            else:
                txt = "foo"
            out.append((kind, RegexCase(txt, self._target_expected(txt))))
        return out

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[RegexCase], Arena[RegexCase], Arena[RegexCase]]:
        if mode == "repair":
            planner = Arena("regex_repair_planner", self._base_cases(["valid_short", "invalid_empty", "invalid_miss"], seed + 1, 12))
            proof = Arena("regex_repair_proof", self._base_cases(["valid_short", "valid_long", "invalid_empty", "invalid_miss", "near_miss"], seed + 2, 28))
            hidden = Arena("regex_repair_hidden", self._base_cases(["valid_short", "valid_long", "invalid_empty", "invalid_miss", "near_miss", "attack_repetition"], seed + 3, 32))
        elif mode == "harden":
            planner = Arena("regex_harden_planner", self._base_cases(["valid_long", "near_miss", "attack_repetition"], seed + 11, 12))
            proof = Arena("regex_harden_proof", self._base_cases(["valid_short", "valid_long", "near_miss", "attack_repetition"], seed + 12, 28))
            hidden = Arena("regex_harden_hidden", self._base_cases(["valid_short", "valid_long", "invalid_miss", "near_miss", "attack_repetition"], seed + 13, 32))
        else:
            planner = Arena("regex_opt_planner", self._base_cases(["valid_short", "valid_long", "near_miss"], seed + 21, 12))
            proof = Arena("regex_opt_proof", self._base_cases(["valid_short", "valid_long", "invalid_miss", "near_miss"], seed + 22, 28))
            hidden = Arena("regex_opt_hidden", self._base_cases(["valid_short", "valid_long", "invalid_miss", "near_miss", "attack_repetition"], seed + 23, 32))
        return planner, proof, hidden

    def _compile(self, pattern: str) -> Optional[re.Pattern[str]]:
        try:
            return re.compile("^" + pattern + "$")
        except re.error:
            return None

    def _risk_score(self, pattern: str) -> float:
        score = 0.0
        if pattern.endswith("*"):
            score += 3.0
        score += max(0, pattern.count("(?:") - 1) * 0.8
        alts = self._extract_alts(pattern)
        if alts:
            seen = set()
            dup = 0
            for alt in alts:
                if alt in seen:
                    dup += 1
                seen.add(alt)
            score += dup * 1.2
            for i in range(len(alts)):
                for j in range(i + 1, len(alts)):
                    pref = 0
                    for a, b in zip(alts[i], alts[j]):
                        if a == b:
                            pref += 1
                        else:
                            break
                    if pref > 0:
                        score += 0.15 * pref
        if "f(?:oo)" in pattern or "ba(?:r)" in pattern:
            score += 2.0
        if "(?:foo(?:))" in pattern or "(?:bar(?:))" in pattern:
            score += 1.0
        return score

    def case_cost(self, candidate: RegexCandidate, case: RegexCase, mode: str) -> Tuple[bool, float]:
        comp = self._compile(candidate.pattern)
        if comp is None:
            return False, float("inf")
        def _timeout(signum, frame):
            raise TimeoutError("regex timeout")
        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        try:
            t0 = time.perf_counter_ns()
            ok = bool(comp.fullmatch(case.text)) == case.expected
            dt = (time.perf_counter_ns() - t0) / 1e3
        except TimeoutError:
            ok = False
            dt = 5000.0
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except TimeoutError:
                pass
            signal.signal(signal.SIGALRM, old_handler)
        cost = dt + 0.35 * len(candidate.pattern) + 5.0 * self._risk_score(candidate.pattern)
        return ok, cost

    def evaluate(self, candidate: RegexCandidate, arena: Arena[RegexCase], mode: str, root: Optional[RegexCandidate] = None) -> EvalMetrics:
        failures: Dict[str, int] = {}
        costs: List[float] = []
        correct = 0
        for _, case in arena.cases:
            ok, cost = self.case_cost(candidate, case, mode)
            if not ok or not math.isfinite(cost):
                failures["wrong"] = failures.get("wrong", 0) + 1
                continue
            correct += 1
            costs.append(cost)
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = float(len(candidate.pattern))
        edit_distance = float(self.structural_edit_distance(root, candidate) if root else 0.0)
        failure_penalty = (len(arena.cases) - correct) * 1000.0
        scalar = failure_penalty
        if mode == "repair":
            scalar += avg_cost * 14.0 + worst_cost * 4.0 + complexity * 0.2 + edit_distance * 0.15
        elif mode == "harden":
            scalar += worst_cost * 18.0 + avg_cost * 6.0 + self._risk_score(candidate.pattern) * 10.0 + complexity * 0.15
        else:
            scalar += avg_cost * 12.0 + worst_cost * 4.0 + complexity * 0.2
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, failures)

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        return {
            "duplicate_alt": {"dedupe_alt": 2.8, "canonicalize": 2.2},
            "nested_group": {"flatten_groups": 2.6, "canonicalize": 2.0},
            "accepts_empty": {"repair_empty_plus": 2.8, "canonicalize": 1.8},
            "overmatch": {"repair_remove_baz": 2.8, "canonicalize": 1.8},
            "undermatch": {"repair_baar_to_bar": 2.8, "canonicalize": 1.8},
            "prefix_ambiguity": {"disambiguate_alt": 2.6, "canonicalize": 2.0},
        }.get(pathology, {})

    def rediscovery_verdict(self, root: RegexCandidate, champion: RegexCandidate) -> str:
        if champion.pattern == r"(?:foo|bar)+":
            return "rediscovered canonical token regex"
        if self.classify_family(root) == self.classify_family(champion):
            return "refined same family"
        return "family shift"

    def suite_attribution(self, root: RegexCandidate, champion: RegexCandidate, arena: Arena[RegexCase], mode: str) -> List[Tuple[str, float, float]]:
        rows: Dict[str, List[float]] = {}
        corr: Dict[str, List[float]] = {}
        for label, case in arena.cases:
            rok, rc = self.case_cost(root, case, mode)
            cok, cc = self.case_cost(champion, case, mode)
            rows.setdefault(label, []).append((rc if math.isfinite(rc) else 1000.0) - (cc if math.isfinite(cc) else 1000.0))
            corr.setdefault(label, []).append((1.0 if cok else 0.0) - (1.0 if rok else 0.0))
        out = []
        for kind in rows:
            out.append((kind, sum(rows[kind]) / len(rows[kind]), sum(corr[kind]) / len(corr[kind])))
        out.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return out

    def attack_arena(self, mode: str, weak_kinds: Sequence[str], seed: int) -> Arena[RegexCase]:
        kinds = list(dict.fromkeys(list(weak_kinds) + ["attack_repetition", "near_miss", "valid_long"]))
        return Arena(f"regex_counter_{mode}", self._base_cases(kinds, seed + 5001, 36))

    def weak_kinds(self, champion: RegexCandidate, arena: Arena[RegexCase], mode: str) -> List[str]:
        buckets: Dict[str, List[float]] = {}
        fails: Dict[str, int] = {}
        for kind, case in arena.cases:
            ok, cost = self.case_cost(champion, case, mode)
            buckets.setdefault(kind, []).append(cost if math.isfinite(cost) else 1000.0)
            if not ok:
                fails[kind] = fails.get(kind, 0) + 1
        scored = []
        for kind, vals in buckets.items():
            scored.append((statistics.mean(vals) + 5 * fails.get(kind, 0), kind))
        scored.sort(reverse=True)
        return [k for _, k in scored[:3]]

    def _extract_alts(self, pattern: str) -> List[str]:
        m = re.fullmatch(r"\(\?:([^()]+)\)(?:\+|\*)", pattern)
        if not m:
            return []
        return m.group(1).split("|")

    def _dedupe_alt(self, pattern: str) -> str:
        alts = self._extract_alts(pattern)
        if not alts:
            return pattern
        seen, out = set(), []
        for alt in alts:
            if alt not in seen:
                seen.add(alt)
                out.append(alt)
        suffix = "+" if pattern.endswith("+") else "*" if pattern.endswith("*") else ""
        return f"(?:{'|'.join(out)}){suffix}"

    def _flatten_groups(self, pattern: str) -> str:
        p = pattern
        reps = [
            (r"\(\?:\(\?:foo\)\|\(\?:bar\)\)\+", r"(?:foo|bar)+"),
            (r"\(\?:\(\?:foo\)\|\(\?:bar\)\)\*", r"(?:foo|bar)*"),
            (r"\(\?:foo\(\?:\)\)", "foo"),
            (r"\(\?:bar\(\?:\)\)", "bar"),
        ]
        for a, b in reps:
            p = re.sub(a, b, p)
        p = p.replace("(?:(?:foo)|(?:bar))+", "(?:foo|bar)+")
        p = p.replace("(?:(?:foo)|(?:bar))*", "(?:foo|bar)*")
        return p

    def _sort_alt(self, pattern: str) -> str:
        alts = self._extract_alts(pattern)
        if not alts:
            return pattern
        suffix = "+" if pattern.endswith("+") else "*" if pattern.endswith("*") else ""
        alts = sorted(set(alts), key=lambda s: (len(s), s))
        return f"(?:{'|'.join(alts)}){suffix}"

    def _disambiguate(self, pattern: str) -> str:
        p = pattern.replace("f(?:oo)", "foo").replace("ba(?:r)", "bar")
        p = p.replace("(?:(?:foo)|(?:bar)|(?:fo(?:o))|(?:ba(?:r)))+", "(?:foo|bar)+")
        p = p.replace("(?:foo|f(?:oo)|bar|ba(?:r))+", "(?:foo|bar)+")
        return p


# ------------------------------ Autolift subsystem ----------------------------


class AutoLiftError(Exception):
    pass


class UnsupportedFunction(AutoLiftError):
    pass


class TimeoutExceeded(AutoLiftError):
    pass


class IRNode:
    pass


class IRExpr(IRNode):
    pass


class IRStmt(IRNode):
    pass


@dataclass(frozen=True)
class IRName(IRExpr):
    name: str


@dataclass(frozen=True)
class IRConst(IRExpr):
    value: Any


@dataclass(frozen=True)
class IRList(IRExpr):
    items: Tuple[IRExpr, ...]


@dataclass(frozen=True)
class IRSet(IRExpr):
    items: Tuple[IRExpr, ...]


@dataclass(frozen=True)
class IRTuple(IRExpr):
    items: Tuple[IRExpr, ...]


@dataclass(frozen=True)
class IRBinOp(IRExpr):
    op: str
    left: IRExpr
    right: IRExpr


@dataclass(frozen=True)
class IRCompare(IRExpr):
    left: IRExpr
    op: str
    right: IRExpr


@dataclass(frozen=True)
class IRBoolOp(IRExpr):
    op: str
    values: Tuple[IRExpr, ...]


@dataclass(frozen=True)
class IRUnaryOp(IRExpr):
    op: str
    value: IRExpr


@dataclass(frozen=True)
class IRCall(IRExpr):
    func: str
    args: Tuple[IRExpr, ...]


@dataclass(frozen=True)
class IRAttribute(IRExpr):
    value: IRExpr
    attr: str


@dataclass(frozen=True)
class IRSubscript(IRExpr):
    value: IRExpr
    index: IRExpr


@dataclass(frozen=True)
class IRAssign(IRStmt):
    target: str
    value: IRExpr


@dataclass(frozen=True)
class IRAugAssign(IRStmt):
    target: str
    op: str
    value: IRExpr


@dataclass(frozen=True)
class IRExprStmt(IRStmt):
    value: IRExpr


@dataclass(frozen=True)
class IRIf(IRStmt):
    test: IRExpr
    body: Tuple[IRStmt, ...]
    orelse: Tuple[IRStmt, ...]


@dataclass(frozen=True)
class IRForRange(IRStmt):
    target: str
    args: Tuple[IRExpr, ...]
    body: Tuple[IRStmt, ...]


@dataclass(frozen=True)
class IRForIter(IRStmt):
    target: str
    iterable: IRExpr
    body: Tuple[IRStmt, ...]


@dataclass(frozen=True)
class IRReturn(IRStmt):
    value: IRExpr


@dataclass(frozen=True)
class FunctionIR(IRNode):
    name: str
    params: Tuple[str, ...]
    body: Tuple[IRStmt, ...]


class FunctionLifter:
    ALLOWED_CALLS = {"range", "len", "set", "list", "tuple", "sum", "min", "max", "any", "all"}

    def lift(self, fn_def: ast.FunctionDef) -> FunctionIR:
        params = tuple(arg.arg for arg in fn_def.args.args)
        body = tuple(self.lift_stmt(stmt) for stmt in fn_def.body)
        return FunctionIR(fn_def.name, params, body)

    def lift_stmt(self, node: ast.stmt) -> IRStmt:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return IRAssign(node.targets[0].id, self.lift_expr(node.value))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            return IRAugAssign(node.target.id, type(node.op).__name__, self.lift_expr(node.value))
        if isinstance(node, ast.Return):
            return IRReturn(self.lift_expr(node.value))
        if isinstance(node, ast.Expr):
            return IRExprStmt(self.lift_expr(node.value))
        if isinstance(node, ast.If):
            return IRIf(self.lift_expr(node.test), tuple(self.lift_stmt(s) for s in node.body), tuple(self.lift_stmt(s) for s in node.orelse))
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                return IRForRange(node.target.id, tuple(self.lift_expr(a) for a in node.iter.args), tuple(self.lift_stmt(s) for s in node.body))
            if isinstance(node.target, ast.Name):
                return IRForIter(node.target.id, self.lift_expr(node.iter), tuple(self.lift_stmt(s) for s in node.body))
        raise UnsupportedFunction(f"unsupported statement: {type(node).__name__}")

    def lift_expr(self, node: ast.AST) -> IRExpr:
        if isinstance(node, ast.Name):
            return IRName(node.id)
        if isinstance(node, ast.Constant):
            return IRConst(node.value)
        if isinstance(node, ast.List):
            return IRList(tuple(self.lift_expr(e) for e in node.elts))
        if isinstance(node, ast.Tuple):
            return IRTuple(tuple(self.lift_expr(e) for e in node.elts))
        if isinstance(node, ast.Set):
            return IRSet(tuple(self.lift_expr(e) for e in node.elts))
        if isinstance(node, ast.BinOp):
            return IRBinOp(type(node.op).__name__, self.lift_expr(node.left), self.lift_expr(node.right))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            return IRCompare(self.lift_expr(node.left), type(node.ops[0]).__name__, self.lift_expr(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            return IRBoolOp(type(node.op).__name__, tuple(self.lift_expr(v) for v in node.values))
        if isinstance(node, ast.UnaryOp):
            return IRUnaryOp(type(node.op).__name__, self.lift_expr(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.ALLOWED_CALLS:
                return IRCall(node.func.id, tuple(self.lift_expr(a) for a in node.args))
            if isinstance(node.func, ast.Attribute):
                return IRCall(f"{self.lift_expr(node.func.value)!r}.{node.func.attr}", tuple(self.lift_expr(a) for a in node.args))
        if isinstance(node, ast.Attribute):
            return IRAttribute(self.lift_expr(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return IRSubscript(self.lift_expr(node.value), self.lift_expr(node.slice))
        raise UnsupportedFunction(f"unsupported expression: {type(node).__name__}")


def _iter_child_paths(node: ast.AST, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], ast.AST]]:
    out = [(prefix, node)]
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, ast.AST):
                    out.extend(_iter_child_paths(item, prefix + (field, idx)))
        elif isinstance(value, ast.AST):
            out.extend(_iter_child_paths(value, prefix + (field,)))
    return out


def _locate_by_path(root: ast.AST, path: Tuple[Any, ...]) -> Optional[ast.AST]:
    cur: Any = root
    i = 0
    try:
        while i < len(path):
            key = path[i]
            if isinstance(key, str):
                cur = getattr(cur, key)
                i += 1
            else:
                cur = cur[key]
                i += 1
        return cur if isinstance(cur, ast.AST) else None
    except Exception:
        return None


def _replace_by_path(root: ast.AST, path: Tuple[Any, ...], new_node: ast.AST) -> bool:
    if not path:
        return False
    parent_path = path[:-1]
    last = path[-1]
    parent = _locate_by_path(root, parent_path)
    if parent is None:
        return False
    try:
        if isinstance(last, str):
            setattr(parent, last, new_node)
        else:
            container = getattr(parent, path[-2]) if len(path) >= 2 and isinstance(path[-2], str) else None
            if isinstance(container, list):
                container[last] = new_node
            else:
                return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class ConstKnob:
    path: Tuple[Any, ...]
    value: int


@dataclass(frozen=True)
class CompareKnob:
    path: Tuple[Any, ...]
    op_name: str


@dataclass(frozen=True)
class MembershipKnob:
    assign_path: Tuple[Any, ...]
    var_name: str


@dataclass(frozen=True)
class EarlyReturnKnob:
    assign_path: Tuple[Any, ...]
    loop_if_path: Tuple[Any, ...]
    flag_name: str


def _clone_module(module: ast.Module) -> ast.Module:
    return copy.deepcopy(module)


def _get_func_def(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise UnsupportedFunction(f"function {name} not found")


def _allowed_builtin_env() -> Dict[str, Any]:
    return {
        "__builtins__": {
            "len": len,
            "range": range,
            "set": set,
            "list": list,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "any": any,
            "all": all,
            "abs": abs,
            "enumerate": enumerate,
            "sorted": sorted,
        }
    }


def _normalize_source(src: str) -> str:
    tree = ast.parse(src)
    return ast.unparse(tree)


@dataclass(frozen=True)
class AutoLiftCase:
    args: Tuple[Any, ...]
    kwargs: Tuple[Tuple[str, Any], ...]
    expected: Any
    kind: str

    @property
    def kwargs_dict(self) -> Dict[str, Any]:
        return dict(self.kwargs)


@dataclass(frozen=True)
class AutoLiftCandidate:
    func_name: str
    source: str
    ir_summary: str
    discovered_knobs: Tuple[str, ...] = ()

    def complexity(self) -> int:
        return len(self.source.splitlines()) + len(self.discovered_knobs)



class AutoLiftAnalyzer:
    SAFE_GLOBALS = {"True", "False", "None", "bool", "int", "str", "list", "dict", "tuple", "set", "List", "Dict", "Tuple", "Set", "Optional", "Sequence", "Any"}
    SAFE_METHODS = {"append", "add", "extend"}
    UNSUPPORTED_NODES = (
        ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.While, ast.Lambda, ast.ClassDef,
        ast.AsyncFunctionDef, ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal,
        ast.Delete, ast.Raise, ast.Assert, ast.Await, ast.ListComp, ast.SetComp,
        ast.DictComp, ast.GeneratorExp, ast.Match,
    )

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.func_name = func.__name__
        src = inspect.getsource(func)
        self.source = _normalize_source(src)
        self.module = ast.parse(self.source)
        self.fn_def = _get_func_def(self.module, self.func_name)
        self.audit = self._validate_subset()
        self.ir = FunctionLifter().lift(self.fn_def)

    def _validate_subset(self) -> Dict[str, Any]:
        if self.fn_def.decorator_list:
            raise UnsupportedFunction("decorators are not supported")
        if self.fn_def.args.posonlyargs or self.fn_def.args.kwonlyargs or self.fn_def.args.vararg or self.fn_def.args.kwarg:
            raise UnsupportedFunction("variadic, keyword-only, and positional-only parameters are not supported")
        params = {arg.arg for arg in self.fn_def.args.args}
        assigned: set[str] = set(params)
        called_names: set[str] = set()
        method_calls: set[str] = set()
        int_literals: set[int] = set()
        str_literals: set[str] = set()
        loop_count = 0
        branch_count = 0
        risk_flags: List[str] = []
        for node in ast.walk(self.fn_def):
            if isinstance(node, self.UNSUPPORTED_NODES):
                raise UnsupportedFunction(f"unsupported construct: {type(node).__name__}")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        raise UnsupportedFunction("only simple name assignments are supported")
                    assigned.add(target.id)
            if isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    raise UnsupportedFunction("only simple name augmented assignments are supported")
                assigned.add(node.target.id)
            if isinstance(node, ast.For):
                loop_count += 1
                if isinstance(node.target, ast.Name):
                    assigned.add(node.target.id)
            if isinstance(node, ast.If):
                branch_count += 1
            if isinstance(node, ast.Constant):
                if isinstance(node.value, int) and abs(node.value) <= 10000:
                    int_literals.add(int(node.value))
                if isinstance(node.value, str) and len(node.value) <= 32:
                    str_literals.add(node.value)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                    called_names.add(fname)
                    if fname in {"eval", "exec", "__import__", "open", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr"}:
                        raise UnsupportedFunction(f"unsafe call: {fname}")
                    if fname not in FunctionLifter.ALLOWED_CALLS and fname not in params and fname != self.func_name:
                        raise UnsupportedFunction(f"unsupported call target: {fname}")
                elif isinstance(node.func, ast.Attribute):
                    if not isinstance(node.func.value, ast.Name):
                        raise UnsupportedFunction("nested attribute calls are not supported")
                    method = node.func.attr
                    method_calls.add(method)
                    if method not in self.SAFE_METHODS:
                        raise UnsupportedFunction(f"unsupported method call: {method}")
                else:
                    raise UnsupportedFunction("dynamic call targets are not supported")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in self.SAFE_GLOBALS:
                    continue
                if any(node is arg.annotation for arg in self.fn_def.args.args if arg.annotation is not None):
                    continue
                if node is self.fn_def.returns:
                    continue
                if node.id not in assigned and node.id not in FunctionLifter.ALLOWED_CALLS and node.id != self.func_name:
                    raise UnsupportedFunction(f"free name not allowed: {node.id}")
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                raise UnsupportedFunction("attribute assignment is not supported")
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                raise UnsupportedFunction("subscript assignment is not supported")
        closure = inspect.getclosurevars(self.func)
        if closure.globals or closure.nonlocals:
            raise UnsupportedFunction("functions closing over external global or nonlocal state are not supported")
        if loop_count > 6:
            risk_flags.append("deep_loop_nesting")
        if branch_count > 10:
            risk_flags.append("branch_dense")
        if any(abs(v) > 1000 for v in int_literals):
            risk_flags.append("large_thresholds")
        return {
            "params": sorted(params),
            "called_names": sorted(called_names),
            "method_calls": sorted(method_calls),
            "int_literals": sorted(int_literals),
            "str_literals": sorted(str_literals),
            "loop_count": loop_count,
            "branch_count": branch_count,
            "risk_flags": risk_flags,
        }

    def summarize_ir(self) -> str:
        counts: Dict[str, int] = {}
        def walk(node: Any) -> None:
            name = type(node).__name__
            counts[name] = counts.get(name, 0) + 1
            if dataclasses.is_dataclass(node):
                for field in dataclasses.fields(node):
                    value = getattr(node, field.name)
                    if isinstance(value, tuple):
                        for item in value:
                            if isinstance(item, (IRNode,)):
                                walk(item)
                    elif isinstance(value, IRNode):
                        walk(value)
        import dataclasses
        walk(self.ir)
        pieces = [f"{k}:{counts[k]}" for k in sorted(counts)]
        pieces.append(f"loops:{self.audit['loop_count']}")
        pieces.append(f"branches:{self.audit['branch_count']}")
        return ", ".join(pieces)

    def make_candidate(self) -> AutoLiftCandidate:
        return AutoLiftCandidate(self.func_name, self.source, self.summarize_ir())

    def discover_knobs(self) -> Tuple[List[ConstKnob], List[CompareKnob], List[MembershipKnob], List[EarlyReturnKnob]]:
        consts: List[ConstKnob] = []
        comps: List[CompareKnob] = []
        membs: List[MembershipKnob] = []
        earlys: List[EarlyReturnKnob] = []
        assign_vars: Dict[str, Tuple[Tuple[Any, ...], ast.Assign]] = {}
        for path, node in _iter_child_paths(self.fn_def):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assign_vars[node.targets[0].id] = (path, node)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and abs(node.value) <= 64:
                consts.append(ConstKnob(path, int(node.value)))
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]).__name__ in {"Lt", "LtE", "Gt", "GtE"}:
                comps.append(CompareKnob(path, type(node.ops[0]).__name__))
        for path, node in _iter_child_paths(self.fn_def):
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name):
                    var = node.comparators[0].id
                    if var in assign_vars:
                        apath, assign = assign_vars[var]
                        if isinstance(assign.value, (ast.List, ast.Tuple, ast.Set)) and all(isinstance(e, ast.Constant) for e in assign.value.elts):
                            membs.append(MembershipKnob(apath, var))
        return_name = None
        for stmt in self.fn_def.body:
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                return_name = stmt.value.id
        if return_name:
            for path, node in _iter_child_paths(self.fn_def):
                if isinstance(node, ast.If):
                    for idx, stmt in enumerate(node.body):
                        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                            if stmt.targets[0].id == return_name and isinstance(stmt.value, ast.Constant) and stmt.value.value is True:
                                earlys.append(EarlyReturnKnob(path + ("body", idx), path, return_name))
        return consts[:14], comps[:10], membs[:6], earlys[:4]


class _CallTimeout:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.old = None

    def __enter__(self):
        def handler(signum, frame):
            raise TimeoutExceeded("call timeout")
        self.old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old)
        return False


def _compile_function_source(source: str, func_name: str) -> Callable[..., Any]:
    module = ast.parse(source)
    code = compile(ast.fix_missing_locations(module), "<forge-autolift>", "exec")
    env: Dict[str, Any] = _allowed_builtin_env()
    ns: Dict[str, Any] = {}
    exec(code, env, ns)
    fn = ns.get(func_name) or env.get(func_name)
    if not callable(fn):
        raise AutoLiftError(f"failed to compile function {func_name}")
    return fn


def _safe_call(fn: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any], timeout: float = 0.03) -> Any:
    with _CallTimeout(timeout):
        return fn(*args, **kwargs)


def _normalize_test_item(item: Any, baseline: Callable[..., Any]) -> AutoLiftCase:
    if isinstance(item, AutoLiftCase):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        spec, expected = item
        if isinstance(spec, tuple):
            return AutoLiftCase(tuple(spec), tuple(), expected, "user")
        if isinstance(spec, dict):
            return AutoLiftCase(tuple(), tuple(sorted(spec.items())), expected, "user")
    if isinstance(item, tuple) and len(item) == 3:
        args, kwargs, expected = item
        return AutoLiftCase(tuple(args), tuple(sorted(kwargs.items())), expected, "user")
    raise AutoLiftError("tests must be ((args_tuple), expected) or (args_tuple, kwargs_dict, expected)")


def _infer_generators(func: Callable[..., Any], cases: Sequence[AutoLiftCase]) -> List[Callable[[random.Random], Any]]:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    observed: List[List[Any]] = [[] for _ in params]
    for case in cases:
        for i, value in enumerate(case.args):
            if i < len(observed):
                observed[i].append(value)

    gens: List[Callable[[random.Random], Any]] = []
    for i, param in enumerate(params):
        ann = param.annotation
        vals = observed[i]

        def make_int(vals=vals):
            low = min(vals) if vals and all(isinstance(v, int) for v in vals) else -10
            high = max(vals) if vals and all(isinstance(v, int) for v in vals) else 10
            return lambda rng: rng.randint(low - 3, high + 3)

        def make_bool():
            return lambda rng: bool(rng.getrandbits(1))

        def make_str(vals=vals):
            tokens = []
            for v in vals:
                if isinstance(v, str):
                    tokens.extend([v, v + "x"])
            tokens.extend(["foo", "bar", "DROP", "DELETE", "TRUNCATE", "ALTER", "noop", "safe", "x"])
            uniq = list(dict.fromkeys(tokens)) or ["x", "y"]
            return lambda rng: rng.choice(uniq)

        def make_list(inner_gen: Callable[[random.Random], Any], max_len: int = 20):
            return lambda rng: [inner_gen(rng) for _ in range(rng.randint(0, max_len))]

        origin = get_origin(ann)
        args = get_args(ann)
        if ann is int:
            gens.append(make_int())
        elif ann is bool:
            gens.append(make_bool())
        elif ann is str:
            gens.append(make_str())
        elif origin in {list, List} and args:
            inner = args[0]
            if inner is int:
                gens.append(make_list(make_int(), 24))
            else:
                gens.append(make_list(make_str(), 24))
        else:
            if vals and all(isinstance(v, list) for v in vals):
                flat = [x for v in vals for x in v]
                if flat and all(isinstance(x, int) for x in flat):
                    gens.append(make_list(make_int(flat), 24))
                else:
                    gens.append(make_list(make_str(flat), 24))
            elif vals and all(isinstance(v, str) for v in vals):
                gens.append(make_str(vals))
            elif vals and all(isinstance(v, int) for v in vals):
                gens.append(make_int(vals))
            else:
                gens.append(make_int())
    return gens



def _boundary_variants(value: Any, rng: random.Random, int_literals: Sequence[int], str_literals: Sequence[str]) -> List[Any]:
    out: List[Any] = []
    if isinstance(value, int):
        seeds = list(dict.fromkeys([value, 0, 1, -1] + [v for lit in int_literals[:8] for v in (lit - 1, lit, lit + 1)]))
        out.extend(seeds[:8])
    elif isinstance(value, str):
        tokens = list(dict.fromkeys([value, "", value * 2, value[:1], "x"] + list(str_literals[:6])))
        out.extend(tokens[:8])
    elif isinstance(value, list):
        out.append([])
        out.append(value[:1])
        out.append(value[: min(4, len(value))])
        if value:
            out.append(value + value[: min(3, len(value))])
            if all(isinstance(x, str) for x in value):
                hostile = list(str_literals[:4]) or ["DROP", "ALTER", "safe"]
                out.append([rng.choice(hostile) for _ in range(min(12, max(4, len(value) * 2)))])
            elif all(isinstance(x, int) for x in value):
                ints = list(int_literals[:4]) or [0, 1, -1]
                out.append([rng.choice(ints) for _ in range(min(12, max(4, len(value) * 2)))])
    return out

def _generate_fuzz_cases(func: Callable[..., Any], analyzer: AutoLiftAnalyzer, user_cases: Sequence[AutoLiftCase], count: int = 60) -> List[AutoLiftCase]:
    rng = random.Random(913)
    gens = _infer_generators(func, user_cases)
    int_literals = analyzer.audit.get("int_literals", [])
    str_literals = analyzer.audit.get("str_literals", [])
    out: List[AutoLiftCase] = []
    seen = set()
    for _ in range(count):
        args = tuple(gen(rng) for gen in gens)
        try:
            expected = _safe_call(func, args, {}, 0.04)
        except Exception:
            continue
        key = repr(args)
        if key not in seen:
            seen.add(key)
            kind = "fuzz_long" if any(isinstance(a, list) and len(a) >= 12 for a in args) else "fuzz"
            out.append(AutoLiftCase(args, tuple(), expected, kind))
        for idx, value in enumerate(args):
            for variant in _boundary_variants(value, rng, int_literals, str_literals):
                alt = list(args)
                alt[idx] = variant
                alt_t = tuple(alt)
                key = repr(alt_t)
                if key in seen:
                    continue
                try:
                    expected = _safe_call(func, alt_t, {}, 0.04)
                except Exception:
                    continue
                seen.add(key)
                label = "boundary" if not (isinstance(variant, list) and len(variant) >= 12) else "boundary_long"
                out.append(AutoLiftCase(alt_t, tuple(), expected, label))
                if len(out) >= count:
                    return out[:count]
        if len(out) >= count:
            break
    return out[:count]

def _generate_attack_cases(func: Callable[..., Any], analyzer: AutoLiftAnalyzer, user_cases: Sequence[AutoLiftCase], count: int = 28) -> List[AutoLiftCase]:
    rng = random.Random(1911)
    gens = _infer_generators(func, user_cases)
    int_literals = analyzer.audit.get("int_literals", [])
    str_literals = analyzer.audit.get("str_literals", [])
    out: List[AutoLiftCase] = []
    seen = set()
    for _ in range(count * 2):
        args = []
        for gen in gens:
            value = gen(rng)
            if isinstance(value, list):
                if value and all(isinstance(x, str) for x in value):
                    hostile = list(dict.fromkeys(list(str_literals[:6]) + ["DROP", "DELETE", "TRUNCATE", "ALTER", "safe", "noop", "x"]))
                    value = [rng.choice(hostile) for _ in range(rng.randint(24, 96))]
                elif value and all(isinstance(x, int) for x in value):
                    ints = list(dict.fromkeys(list(int_literals[:6]) + [-200, -1, 0, 1, 2, 200]))
                    value = [rng.choice(ints) for _ in range(rng.randint(24, 96))]
                else:
                    value = [value[0] if value else 0 for _ in range(rng.randint(24, 96))]
            elif isinstance(value, int):
                value = rng.choice(list(dict.fromkeys(list(int_literals[:6]) + [0, 1, -1, 128, -128, 512, -512])))
            elif isinstance(value, str):
                tokens = list(dict.fromkeys(list(str_literals[:6]) + ["DROP", "DELETE", "TRUNCATE", "ALTER", "", "safe", "noop"]))
                value = rng.choice(tokens) * max(1, rng.randint(1, 6))
            args.append(value)
        args_t = tuple(args)
        key = repr(args_t)
        if key in seen:
            continue
        try:
            expected = _safe_call(func, args_t, {}, 0.05)
        except Exception:
            continue
        seen.add(key)
        out.append(AutoLiftCase(args_t, tuple(), expected, "attack"))
        if len(out) >= count:
            break
    return out



@dataclass(frozen=True)
class AutoLiftOperatorSpec:
    op: DomainOperator["AutoLiftCandidate"]
    risk: str
    rationale: str

def _make_autolift_operators(analyzer: AutoLiftAnalyzer) -> List[AutoLiftOperatorSpec]:
    consts, comps, membs, earlys = analyzer.discover_knobs()
    func_name = analyzer.func_name
    specs: List[AutoLiftOperatorSpec] = []

    def add(name: str, fn: Callable[[AutoLiftCandidate], AutoLiftCandidate], risk: str, rationale: str) -> None:
        specs.append(AutoLiftOperatorSpec(DomainOperator(name, fn), risk, rationale))

    for knob in consts[:8]:
        for delta in (-1, 1):
            def make_const_op(knob=knob, delta=delta):
                def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                    mod = ast.parse(c.source)
                    node = _locate_by_path(_get_func_def(mod, func_name), knob.path)
                    if not isinstance(node, ast.Constant) or not isinstance(node.value, int):
                        return c
                    node.value = node.value + delta
                    return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"const{delta}@{knob.path}",))
                return fn
            add(f"const_{len(specs)}_{'plus' if delta>0 else 'minus'}1", make_const_op(), "speculative", "adjust small integer threshold by one")

    flip = {"Lt": ast.LtE, "LtE": ast.Lt, "Gt": ast.GtE, "GtE": ast.Gt}
    for knob in comps[:6]:
        def make_cmp_op(knob=knob):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                node = _locate_by_path(_get_func_def(mod, func_name), knob.path)
                if not isinstance(node, ast.Compare) or not node.ops:
                    return c
                name = type(node.ops[0]).__name__
                if name not in flip:
                    return c
                node.ops[0] = flip[name]()
                return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"cmp@{knob.path}",))
            return fn
        add(f"flip_compare_{len(specs)}", make_cmp_op(), "speculative", "tweak inclusive or exclusive comparison boundary")

    for knob in membs[:4]:
        def make_setlift(knob=knob):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                node = _locate_by_path(fn_def, knob.assign_path)
                if not isinstance(node, ast.Assign):
                    return c
                value = node.value
                if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    return c
                node.value = ast.Set(elts=list(value.elts))
                for sub in ast.walk(fn_def):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name) and sub.func.value.id == knob.var_name and sub.func.attr == "append":
                        sub.func.attr = "add"
                return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"setlift:{knob.var_name}",))
            return fn
        add(f"setlift_{knob.var_name}", make_setlift(), "safe", "replace repeated linear membership with set membership")

    for knob in earlys[:3]:
        def make_early(knob=knob):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                node = _locate_by_path(fn_def, knob.assign_path)
                if not isinstance(node, ast.Assign):
                    return c
                new_node = ast.Return(value=ast.Constant(True))
                if not _replace_by_path(fn_def, knob.assign_path, new_node):
                    return c
                return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"early:{knob.flag_name}",))
            return fn
        add(f"early_return_{knob.flag_name}", make_early(), "aggressive", "convert boolean flag commit into early return")

    if not specs:
        add("identity", lambda c: c, "safe", "fallback identity operator")
    return specs[:24]



class AutoLiftDomainPack(DomainPack[AutoLiftCandidate]):
    name = "autolift"

    def __init__(self, func: Callable[..., Any], tests: Sequence[Any], *, root_override: Optional[AutoLiftCandidate] = None, extra_proof_cases: Optional[Sequence[AutoLiftCase]] = None, extra_hidden_cases: Optional[Sequence[AutoLiftCase]] = None, extra_counter_cases: Optional[Sequence[AutoLiftCase]] = None):
        self.func = func
        self.func_name = func.__name__
        self.analyzer = AutoLiftAnalyzer(func)
        self.root = root_override or self.analyzer.make_candidate()
        self.user_cases = [_normalize_test_item(t, func) for t in tests]
        self.fuzz_cases = _generate_fuzz_cases(func, self.analyzer, self.user_cases, 36)
        self.attack_cases = _generate_attack_cases(func, self.analyzer, self.user_cases, 20)
        self.extra_proof_cases = list(extra_proof_cases or [])
        self.extra_hidden_cases = list(extra_hidden_cases or [])
        self.extra_counter_cases = list(extra_counter_cases or [])
        self._op_specs = _make_autolift_operators(self.analyzer)
        self._ops = {spec.op.name: spec.op for spec in self._op_specs}
        self.operator_risks = {spec.op.name: {"risk": spec.risk, "rationale": spec.rationale} for spec in self._op_specs}
        self._compile_cache: Dict[str, Callable[..., Any]] = {}

    def root_sets(self) -> Dict[str, Dict[str, AutoLiftCandidate]]:
        return {"optimize": {self.func_name: self.root}}

    def candidate_hash(self, candidate: AutoLiftCandidate) -> str:
        return _stable_hash(candidate.source)

    def summarize(self, candidate: AutoLiftCandidate) -> str:
        return f"{candidate.func_name} knobs={list(candidate.discovered_knobs)}"

    def complexity(self, candidate: AutoLiftCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: AutoLiftCandidate) -> str:
        src = candidate.source
        if (" in " in src and ("set(" in src or "={" in src or "= {" in src)):
            return "set_membership"
        if "return True" in src and "found = True" not in src:
            return "early_return"
        if "for " in src and "if " in src:
            return "loop_filter"
        return "generic_python"

    def detect_pathology(self, candidate: AutoLiftCandidate, mode: str) -> str:
        src = candidate.source
        if "found = True" in src and "return found" in src:
            return "late_boolean_commit"
        if " in [" in src or " in (" in src:
            return "linear_membership_literal"
        if " in forbidden" in src and "forbidden = [" in src:
            return "linear_membership_state"
        if self.analyzer.audit["risk_flags"]:
            return "risky_structure"
        return "generic"

    def structural_edit_distance(self, root: Optional[AutoLiftCandidate], candidate: AutoLiftCandidate) -> int:
        if root is None:
            return len(candidate.source)
        return abs(len(root.source) - len(candidate.source)) + (0 if root.source == candidate.source else 1) + max(0, len(candidate.discovered_knobs) - len(root.discovered_knobs))

    def operator_bank(self, mode: str, root: AutoLiftCandidate, promoted: Sequence[DomainOperator[AutoLiftCandidate]]) -> List[DomainOperator[AutoLiftCandidate]]:
        out: List[DomainOperator[AutoLiftCandidate]] = []
        seen = set()
        def risk_rank(name: str) -> int:
            risk = self.operator_risks.get(name, {}).get("risk", "speculative")
            return {"safe": 0, "speculative": 1, "aggressive": 2}.get(risk, 1)
        all_ops = list(promoted) + list(self._ops.values())
        all_ops.sort(key=lambda op: (risk_rank(op.name), op.name))
        for op in all_ops:
            if op.name not in seen:
                out.append(op)
                seen.add(op.name)
        return out

    def operator_lookup(self) -> Dict[str, DomainOperator[AutoLiftCandidate]]:
        return self._ops

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[AutoLiftCase], Arena[AutoLiftCase], Arena[AutoLiftCase]]:
        planner = Arena("autolift_planner", [(c.kind, c) for c in self.user_cases + self.fuzz_cases[:14]])
        proof = Arena("autolift_proof", [(c.kind, c) for c in self.user_cases + self.fuzz_cases[:22] + self.extra_proof_cases])
        hidden = Arena("autolift_hidden", [(c.kind, c) for c in self.user_cases + self.fuzz_cases + self.extra_hidden_cases])
        return planner, proof, hidden

    def _get_compiled(self, candidate: AutoLiftCandidate) -> Callable[..., Any]:
        h = self.candidate_hash(candidate)
        if h not in self._compile_cache:
            self._compile_cache[h] = _compile_function_source(candidate.source, candidate.func_name)
        return self._compile_cache[h]

    def case_cost(self, candidate: AutoLiftCandidate, case: AutoLiftCase, mode: str) -> Tuple[bool, float]:
        fn = self._get_compiled(candidate)
        kwargs = dict(case.kwargs)
        reps = 2 if any(isinstance(a, list) and len(a) > 28 for a in case.args) else 4
        t0 = time.perf_counter_ns()
        out = None
        try:
            for _ in range(reps):
                out = _safe_call(fn, case.args, kwargs, 0.03 if case.kind != "attack" else 0.04)
        except Exception:
            return False, float("inf")
        dt = (time.perf_counter_ns() - t0) / max(1, reps) / 1e3
        ok = out == case.expected
        src_penalty = 0.015 * len(candidate.source)
        risk_penalty = 2.5 * len(self.analyzer.audit["risk_flags"])
        return ok, dt + src_penalty + risk_penalty

    def evaluate(self, candidate: AutoLiftCandidate, arena: Arena[AutoLiftCase], mode: str, root: Optional[AutoLiftCandidate] = None) -> EvalMetrics:
        failures: Dict[str, int] = {}
        costs: List[float] = []
        correct = 0
        for _, case in arena.cases:
            ok, cost = self.case_cost(candidate, case, mode)
            if not ok or not math.isfinite(cost):
                failures["wrong"] = failures.get("wrong", 0) + 1
                continue
            correct += 1
            costs.append(cost)
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = float(candidate.complexity())
        edit_distance = float(self.structural_edit_distance(root, candidate) if root else 0.0)
        failure_penalty = (len(arena.cases) - correct) * 1000.0
        scalar = failure_penalty + avg_cost * 14.0 + worst_cost * 4.0 + complexity * 0.22 + edit_distance * 0.12
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, failures)

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        return {
            "late_boolean_commit": {name: 2.4 for name in self._ops if name.startswith("early_return_")},
            "linear_membership_literal": {name: 2.6 for name in self._ops if name.startswith("setlift_")},
            "linear_membership_state": {name: 2.8 for name in self._ops if name.startswith("setlift_")},
            "risky_structure": {"identity": -0.2},
        }.get(pathology, {})

    def rediscovery_verdict(self, root: AutoLiftCandidate, champion: AutoLiftCandidate) -> str:
        if root.source == champion.source:
            return "root retained"
        if "return True" in champion.source and ("set(" in champion.source or "={" in champion.source):
            return "rediscovered early-return + set-membership optimization"
        return "refined function structure"

    def suite_attribution(self, root: AutoLiftCandidate, champion: AutoLiftCandidate, arena: Arena[AutoLiftCase], mode: str) -> List[Tuple[str, float, float]]:
        buckets: Dict[str, List[float]] = {}
        corr: Dict[str, List[float]] = {}
        for label, case in arena.cases:
            rok, rc = self.case_cost(root, case, mode)
            cok, cc = self.case_cost(champion, case, mode)
            buckets.setdefault(label, []).append((rc if math.isfinite(rc) else 1000.0) - (cc if math.isfinite(cc) else 1000.0))
            corr.setdefault(label, []).append((1.0 if cok else 0.0) - (1.0 if rok else 0.0))
        out = []
        for kind in buckets:
            out.append((kind, sum(buckets[kind]) / len(buckets[kind]), sum(corr[kind]) / len(corr[kind])))
        out.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return out

    def attack_arena(self, mode: str, weak_kinds: Sequence[str], seed: int) -> Arena[AutoLiftCase]:
        cases = self.attack_cases + self.extra_counter_cases + self.fuzz_cases[:12]
        return Arena("autolift_counter", [(c.kind, c) for c in cases])

    def weak_kinds(self, champion: AutoLiftCandidate, arena: Arena[AutoLiftCase], mode: str) -> List[str]:
        buckets: Dict[str, List[float]] = {}
        fails: Dict[str, int] = {}
        for kind, case in arena.cases:
            ok, cost = self.case_cost(champion, case, mode)
            buckets.setdefault(kind, []).append(cost if math.isfinite(cost) else 1000.0)
            if not ok:
                fails[kind] = fails.get(kind, 0) + 1
        scored = []
        for kind, vals in buckets.items():
            scored.append((sum(vals) / len(vals) + 5 * fails.get(kind, 0), kind))
        scored.sort(reverse=True)
        return [k for _, k in scored[:3]]



@dataclass
class ImproveResult:
    original_source: str
    improved_source: str
    improved_function: Callable[..., Any]
    planner: EvalMetrics
    proof: EvalMetrics
    hidden: EvalMetrics
    counter: Optional[EvalMetrics]
    path: List[str]
    report_markdown: str
    ir_summary: str
    discovered_operators: List[str]
    subset_audit: Dict[str, Any]
    operator_risks: Dict[str, Dict[str, str]]
    synthesized_case_counts: Dict[str, int]
    recovery_applied: bool
    diff_preview: str

def _diff_preview(original: str, improved: str, limit: int = 1200) -> str:
    diff = "\n".join(difflib.unified_diff(original.splitlines(), improved.splitlines(), fromfile="original", tofile="improved", lineterm=""))
    return diff[:limit]

def improve(func: Callable[..., Any], tests: Sequence[Any], *, config: Optional[EngineConfig] = None, use_counter: bool = True) -> ImproveResult:
    original_source = AutoLiftAnalyzer(func).source
    domain = AutoLiftDomainPack(func, tests)
    engine = ForgeEngine(domain, config or EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=137))
    results, memory, forensics, scores = engine.run()
    result = results[0]
    report = engine.build_report(results, memory, forensics, scores)
    counter_metric = None
    recovery_applied = False

    if use_counter:
        bundles = {"autolift": {"engine": engine, "results": results, "memory": memory, "forensics": forensics, "scores": scores}}
        counter = CounterForgePlus(bundles)
        counter_reports, counter_md = counter.run()
        apply_counter_feedback(bundles, counter_reports)
        attack_arena = domain.attack_arena("optimize", ["attack"], 0)
        counter_metric = domain.evaluate(result.champion, attack_arena, "optimize", domain.root)

        needs_recovery = counter_metric.correct < counter_metric.total or counter_metric.scalar > (result.hidden.scalar * 1.2)
        if needs_recovery:
            recovery_domain = AutoLiftDomainPack(func, tests, root_override=result.champion, extra_proof_cases=domain.attack_cases[:10], extra_hidden_cases=domain.attack_cases, extra_counter_cases=domain.attack_cases)
            recovery_engine = ForgeEngine(recovery_domain, config or EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=139))
            rec_results, rec_memory, rec_forensics, rec_scores = recovery_engine.run()
            rec_result = rec_results[0]
            rec_counter = recovery_domain.evaluate(rec_result.champion, recovery_domain.attack_arena("optimize", ["attack"], 0), "optimize", recovery_domain.root)
            current_score = result.proof.scalar + result.hidden.scalar + counter_metric.scalar
            recovery_score = rec_result.proof.scalar + rec_result.hidden.scalar + rec_counter.scalar
            if rec_counter.correct == rec_counter.total and recovery_score < current_score:
                domain = recovery_domain
                engine = recovery_engine
                results, memory, forensics, scores = rec_results, rec_memory, rec_forensics, rec_scores
                result = rec_result
                counter_metric = rec_counter
                recovery_applied = True
                report = engine.build_report(results, memory, forensics, scores)
            report += "\n\n---\n\n" + counter_md
        else:
            report += "\n\n---\n\n" + counter_md

    report += "\n\n## Autolift hardening\n"
    report += f"- subset audit: {json.dumps(domain.analyzer.audit, sort_keys=True)}\n"
    report += f"- synthesized cases: user={len(domain.user_cases)}, fuzz={len(domain.fuzz_cases)}, attack={len(domain.attack_cases)}, extra_hidden={len(domain.extra_hidden_cases)}\n"
    report += f"- recovery_applied: {recovery_applied}\n"
    report += "- operator risks:\n"
    for name, meta in sorted(domain.operator_risks.items()):
        report += f"  - {name}: {meta['risk']} | {meta['rationale']}\n"

    improved_fn = _compile_function_source(result.champion.source, result.champion.func_name)
    return ImproveResult(
        original_source=original_source,
        improved_source=result.champion.source,
        improved_function=improved_fn,
        planner=result.planner,
        proof=result.proof,
        hidden=result.hidden,
        counter=counter_metric,
        path=result.path,
        report_markdown=report,
        ir_summary=domain.root.ir_summary,
        discovered_operators=list(domain.operator_lookup().keys()),
        subset_audit=domain.analyzer.audit,
        operator_risks=domain.operator_risks,
        synthesized_case_counts={"user": len(domain.user_cases), "fuzz": len(domain.fuzz_cases), "attack": len(domain.attack_cases), "extra_hidden": len(domain.extra_hidden_cases)},
        recovery_applied=recovery_applied,
        diff_preview=_diff_preview(original_source, result.champion.source),
    )

# ----------------------------- Counter-Forge plus -----------------------------
# ----------------------------- Counter-Forge plus -----------------------------


class CounterForgePlus(CounterForge):
    def _weak_kinds(self, domain, result, seed):
        if hasattr(domain, "weak_kinds"):
            _, _, hidden = domain.mode_arenas(result.mode, seed)
            return domain.weak_kinds(result.champion, hidden, result.mode)
        return super()._weak_kinds(domain, result, seed)

    def _build_attack_arena(self, domain, result, weak_kinds, seed):
        if hasattr(domain, "attack_arena"):
            return domain.attack_arena(result.mode, weak_kinds, seed)
        return super()._build_attack_arena(domain, result, weak_kinds, seed)


# ---------------------------------- demos ------------------------------------


def run_domain(domain: DomainPack[Any], config: Optional[EngineConfig] = None):
    engine = ForgeEngine(domain, config or EngineConfig())
    return engine, *engine.run()


def demo_autolift_function(tokens: List[str]) -> bool:
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    found = False
    limit = len(tokens)
    for idx in range(limit):
        tok = tokens[idx]
        if tok in forbidden:
            found = True
    return found


def autolift_demo() -> ImproveResult:
    tests = [
        ((["safe", "noop"],), False),
        ((["DROP"],), True),
        ((["safe", "ALTER"],), True),
        ((["x", "y", "z"],), False),
        ((["DELETE", "x", "y"],), True),
    ]
    return improve(demo_autolift_function, tests)


def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    lines: List[str] = []
    lines.append("# Smart Forge 1.89")
    lines.append("")
    lines.append("## What changed")
    lines.append("- Added a real third proof domain: regex symbolic transformation.")
    lines.append("- Hardened Autolift: `improve(function, tests=...)` now rejects unsafe subsets, grades operator risk, synthesizes stronger probes, and can recover after attack.")
    lines.append("- Autolift parses Python AST, lifts to a deeper IR, discovers tunable knobs, synthesizes risk-graded operators, fuzzes and attacks inputs, and runs the normal Forge pipeline with recovery.")
    lines.append("- Counter-Forge now handles the regex proof domain and Autolift domains through domain-provided attack arenas.")
    lines.append("")
    for domain_name in ("sorting", "pathfinding", "regex"):
        pack = bundles[domain_name]
        lines.append(f"## {domain_name}")
        for k, v in pack["scores"].items():
            lines.append(f"- {k}: {v:.4f}")
        for r in sorted(pack["results"], key=lambda x: (x.mode, x.root_name)):
            lines.append(f"- [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total}, hidden {r.hidden.correct}/{r.hidden.total}, avg {r.proof.avg_cost:.4f}, hidden_worst {r.hidden.worst_cost:.4f}")
        lines.append("")
    lines.append("## Autolift hardened demo")
    lines.append(f"- IR summary: {auto.ir_summary}")
    lines.append(f"- discovered operators: {', '.join(auto.discovered_operators[:8])}")
    lines.append(f"- subset risk flags: {', '.join(auto.subset_audit.get('risk_flags', [])) if auto.subset_audit.get('risk_flags') else 'none'}")
    lines.append(f"- synthesized cases: {auto.synthesized_case_counts}")
    lines.append(f"- recovery applied: {auto.recovery_applied}")
    lines.append(f"- path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    lines.append(f"- proof: {auto.proof.correct}/{auto.proof.total}")
    lines.append(f"- hidden: {auto.hidden.correct}/{auto.hidden.total}")
    if auto.counter:
        lines.append(f"- counter: {auto.counter.correct}/{auto.counter.total}")
    lines.append("")
    lines.append("### Original")
    lines.append("```python")
    lines.extend(auto.original_source.splitlines())
    lines.append("```")
    lines.append("")
    lines.append("### Diff preview")
    lines.append("```diff")
    lines.extend(auto.diff_preview.splitlines())
    lines.append("```")
    lines.append("")
    lines.append("### Improved")
    lines.append("```python")
    lines.extend(auto.improved_source.splitlines())
    lines.append("```")
    lines.append("")
    lines.append("## Public release acceptance checklist")
    lines.append("- standalone script: yes")
    lines.append("- CLI entrypoints: yes")
    lines.append("- public API: optimize / repair / harden / improve")
    lines.append("- proof domains: sorting / pathfinding / regex")
    lines.append("- restricted autolift: enabled")
    lines.append("- planner + motif memory + counter-forge + recovery: enabled")
    return "\n".join(lines)


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19))
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41))
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_public_release_o1_report.md").write_text(combined, encoding="utf-8")
    print("FORGE PUBLIC RELEASE o1")
    print("public release active: three proof domains + restricted autolift")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift hardened]")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    print(f"  improved source hash: {_stable_hash(auto.improved_source)}")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto}


def package_info() -> Dict[str, Any]:
    return {
        "version": __version__,
        "script_name": "forge_public_release_o1.py",
        "commands": ["demo", "autolift-demo", "regex-domain-demo", "list-domains"],
        "domains": ["sorting", "pathfinding", "regex", "autolift_restricted"],
        "modes": ["optimize", "repair", "harden", "improve"],
        "entrypoints": ["CLI", "standalone script", "Python API"],
        "release": "public",
    }


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_public_release_o1", description="Smart Forge 1.89 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("list-domains")
    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE PUBLIC RELEASE o1 / autolift-demo")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
        print("FORGE PUBLIC RELEASE o1 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_public_release_o1_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1




# ===== Forge o1.1 Deep IR foundation =====

__version__ = "o1.1"


@dataclass(frozen=True)
class DeepLoopInfo:
    target: str
    iterator_kind: str
    iterable_repr: str
    depth: int
    guard_count: int
    reduction_targets: Tuple[str, ...]
    membership_checks: Tuple[str, ...]
    repeated_calls: Tuple[str, ...]


@dataclass(frozen=True)
class DeepGuardInfo:
    test_repr: str
    compare_ops: Tuple[str, ...]
    compared_names: Tuple[str, ...]
    constants: Tuple[str, ...]
    depth: int


@dataclass(frozen=True)
class DeepReductionInfo:
    target: str
    op: str
    depth: int
    update_repr: str


@dataclass(frozen=True)
class DeepMembershipInfo:
    element_repr: str
    container_repr: str
    container_kind: str
    depth: int
    materializable: bool


@dataclass(frozen=True)
class DeepCallInfo:
    call_repr: str
    func_name: str
    count: int
    depth: int


@dataclass(frozen=True)
class DeepDataStructureOpportunity:
    kind: str
    target: str
    recommendation: str
    reason: str


@dataclass(frozen=True)
class DeepFunctionIR:
    function_name: str
    loops: Tuple[DeepLoopInfo, ...]
    guards: Tuple[DeepGuardInfo, ...]
    reductions: Tuple[DeepReductionInfo, ...]
    memberships: Tuple[DeepMembershipInfo, ...]
    repeated_calls: Tuple[DeepCallInfo, ...]
    thresholds: Tuple[str, ...]
    opportunities: Tuple[DeepDataStructureOpportunity, ...]

    def summary(self) -> str:
        pieces = [
            f"loops:{len(self.loops)}",
            f"guards:{len(self.guards)}",
            f"reductions:{len(self.reductions)}",
            f"memberships:{len(self.memberships)}",
            f"repeated_calls:{len(self.repeated_calls)}",
            f"thresholds:{len(self.thresholds)}",
            f"opportunities:{len(self.opportunities)}",
        ]
        if self.opportunities:
            lead = ", ".join(f"{o.kind}->{o.recommendation}" for o in self.opportunities[:4])
            pieces.append(f"lead_opportunities:{lead}")
        return ", ".join(pieces)


@dataclass(frozen=True)
class DirectMembershipKnob:
    compare_path: Tuple[Any, ...]
    container_repr: str


@dataclass(frozen=True)
class LenCacheKnob:
    call_paths: Tuple[Tuple[Any, ...], ...]
    var_name: str


def _ast_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _contains_name(node: ast.AST, target: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == target for n in ast.walk(node))


def _loop_depth_by_path(fn_def: ast.FunctionDef) -> Dict[Tuple[Any, ...], int]:
    depth_map: Dict[Tuple[Any, ...], int] = {}

    def walk(node: ast.AST, prefix: Tuple[Any, ...], depth: int) -> None:
        depth_map[prefix] = depth
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        child_depth = depth + 1 if isinstance(item, ast.For) else depth
                        walk(item, prefix + (field, idx), child_depth)
            elif isinstance(value, ast.AST):
                child_depth = depth + 1 if isinstance(value, ast.For) else depth
                walk(value, prefix + (field,), child_depth)
    walk(fn_def, tuple(), 0)
    return depth_map


class DeepIRBuilder:
    def __init__(self, fn_def: ast.FunctionDef, audit: Dict[str, Any]):
        self.fn_def = fn_def
        self.audit = audit
        self.depth_map = _loop_depth_by_path(fn_def)

    def build(self) -> DeepFunctionIR:
        loops: List[DeepLoopInfo] = []
        guards: List[DeepGuardInfo] = []
        reductions: List[DeepReductionInfo] = []
        memberships: List[DeepMembershipInfo] = []
        thresholds: List[str] = []
        repeated_calls_raw: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        assigns: Dict[str, ast.AST] = {}

        for path, node in _iter_child_paths(self.fn_def):
            depth = self.depth_map.get(path, 0)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assigns[node.targets[0].id] = node.value

            if isinstance(node, ast.If):
                compare_ops: List[str] = []
                names: List[str] = []
                constants: List[str] = []
                for sub in ast.walk(node.test):
                    if isinstance(sub, ast.Compare):
                        compare_ops.extend(type(op).__name__ for op in sub.ops)
                    if isinstance(sub, ast.Name):
                        names.append(sub.id)
                    if isinstance(sub, ast.Constant):
                        constants.append(repr(sub.value))
                guards.append(
                    DeepGuardInfo(
                        test_repr=_ast_text(node.test),
                        compare_ops=tuple(sorted(dict.fromkeys(compare_ops))),
                        compared_names=tuple(sorted(dict.fromkeys(names))),
                        constants=tuple(sorted(dict.fromkeys(constants))),
                        depth=depth,
                    )
                )

            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                reductions.append(DeepReductionInfo(node.target.id, type(node.op).__name__, depth, _ast_text(node.value)))
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.BinOp):
                if _contains_name(node.value, node.targets[0].id):
                    reductions.append(DeepReductionInfo(node.targets[0].id, type(node.value.op).__name__, depth, _ast_text(node.value)))

            if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and len(node.comparators) == 1:
                comp = node.comparators[0]
                container_kind = type(comp).__name__
                materializable = False
                if isinstance(comp, ast.Name) and comp.id in assigns and isinstance(assigns[comp.id], (ast.List, ast.Tuple, ast.Set)):
                    container_kind = type(assigns[comp.id]).__name__
                    materializable = True
                elif isinstance(comp, (ast.List, ast.Tuple, ast.Set)):
                    materializable = True
                memberships.append(
                    DeepMembershipInfo(
                        element_repr=_ast_text(node.left),
                        container_repr=_ast_text(comp),
                        container_kind=container_kind,
                        depth=depth,
                        materializable=materializable,
                    )
                )

            if isinstance(node, ast.Compare):
                vals = []
                for side in [node.left] + list(node.comparators):
                    if isinstance(side, ast.Constant) and isinstance(side.value, (int, float, str)):
                        vals.append(repr(side.value))
                thresholds.extend(vals)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        thresholds.append(repr(arg.value))

            if isinstance(node, ast.Call):
                func_name = _ast_text(node.func)
                repeated_calls_raw[(func_name, depth)].append(_ast_text(node))

        repeated_calls: List[DeepCallInfo] = []
        for (func_name, depth), calls in repeated_calls_raw.items():
            if len(calls) >= 2:
                exemplar = calls[0]
                repeated_calls.append(DeepCallInfo(exemplar, func_name, len(calls), depth))

        for path, node in _iter_child_paths(self.fn_def):
            if isinstance(node, ast.For):
                target = node.target.id if isinstance(node.target, ast.Name) else _ast_text(node.target)
                iterator_kind = "range" if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range" else type(node.iter).__name__
                body_nodes = list(ast.walk(node))
                loop_guards = sum(1 for n in body_nodes if isinstance(n, ast.If))
                loop_reductions = sorted({r.target for r in reductions if r.depth > self.depth_map.get(path, 0)})
                loop_memberships = sorted({m.container_repr for m in memberships if m.depth > self.depth_map.get(path, 0)})
                loop_calls = sorted({c.func_name for c in repeated_calls if c.depth > self.depth_map.get(path, 0)})
                loops.append(
                    DeepLoopInfo(
                        target=target,
                        iterator_kind=iterator_kind,
                        iterable_repr=_ast_text(node.iter),
                        depth=self.depth_map.get(path, 0),
                        guard_count=loop_guards,
                        reduction_targets=tuple(loop_reductions),
                        membership_checks=tuple(loop_memberships),
                        repeated_calls=tuple(loop_calls),
                    )
                )

        opportunities: List[DeepDataStructureOpportunity] = []
        for m in memberships:
            if m.materializable and m.container_kind in {"List", "Tuple", "list", "tuple"}:
                opportunities.append(
                    DeepDataStructureOpportunity(
                        kind="membership_materialization",
                        target=m.container_repr,
                        recommendation="set",
                        reason="membership check over linear container is materializable to set"
                    )
                )
        for c in repeated_calls:
            if c.func_name == "len" or c.func_name.endswith(".__len__") or c.call_repr.startswith("len("):
                opportunities.append(
                    DeepDataStructureOpportunity(
                        kind="call_caching",
                        target=c.call_repr,
                        recommendation="cache",
                        reason="repeated size query appears in multiple sites"
                    )
                )
        for l in loops:
            if l.guard_count and l.reduction_targets:
                opportunities.append(
                    DeepDataStructureOpportunity(
                        kind="guarded_reduction",
                        target=l.target,
                        recommendation="early_exit_or_hoist",
                        reason="loop mixes guards and reductions and may support structural tightening"
                    )
                )

        # Deduplicate
        unique_opps: List[DeepDataStructureOpportunity] = []
        seen = set()
        for o in opportunities:
            key = (o.kind, o.target, o.recommendation)
            if key not in seen:
                seen.add(key)
                unique_opps.append(o)

        return DeepFunctionIR(
            function_name=self.fn_def.name,
            loops=tuple(loops),
            guards=tuple(guards),
            reductions=tuple(reductions),
            memberships=tuple(memberships),
            repeated_calls=tuple(repeated_calls),
            thresholds=tuple(sorted(dict.fromkeys(thresholds))),
            opportunities=tuple(unique_opps),
        )


_old_autolift_init = AutoLiftAnalyzer.__init__
def _autolift_init_o11(self, func: Callable[..., Any]):
    _old_autolift_init(self, func)
    self.deep_ir = DeepIRBuilder(self.fn_def, self.audit).build()
AutoLiftAnalyzer.__init__ = _autolift_init_o11


def _summarize_ir_o11(self) -> str:
    counts: Dict[str, int] = {}
    def walk(node: Any) -> None:
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
        if dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                value = getattr(node, field.name)
                if isinstance(value, tuple):
                    for item in value:
                        if isinstance(item, (IRNode,)):
                            walk(item)
                elif isinstance(value, IRNode):
                    walk(value)
    import dataclasses
    walk(self.ir)
    pieces = [f"{k}:{counts[k]}" for k in sorted(counts)]
    pieces.append(f"loops:{self.audit['loop_count']}")
    pieces.append(f"branches:{self.audit['branch_count']}")
    pieces.append(self.deep_ir.summary())
    return ", ".join(pieces)
AutoLiftAnalyzer.summarize_ir = _summarize_ir_o11


def _make_candidate_o11(self) -> AutoLiftCandidate:
    deep_knobs = tuple(
        f"{o.kind}:{o.recommendation}"
        for o in self.deep_ir.opportunities[:6]
    )
    return AutoLiftCandidate(self.func_name, self.source, self.summarize_ir(), deep_knobs)
AutoLiftAnalyzer.make_candidate = _make_candidate_o11


def _discover_direct_memberships(self) -> List[DirectMembershipKnob]:
    out: List[DirectMembershipKnob] = []
    for path, node in _iter_child_paths(self.fn_def):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and len(node.comparators) == 1:
            comp = node.comparators[0]
            if isinstance(comp, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) for e in comp.elts):
                out.append(DirectMembershipKnob(path, _ast_text(comp)))
    return out[:6]
AutoLiftAnalyzer.discover_direct_memberships = _discover_direct_memberships


def _discover_len_cache_sites(self) -> List[LenCacheKnob]:
    calls_by_var: Dict[str, List[Tuple[Any, ...]]] = defaultdict(list)
    assigned = set(self.audit.get("params", []))
    # names assigned in body invalidate simple cache-by-parameter assumption
    for node in ast.walk(self.fn_def):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)

    params = set(arg.arg for arg in self.fn_def.args.args)
    mutable_params = assigned - params
    for path, node in _iter_child_paths(self.fn_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            var = node.args[0].id
            if var in params and var not in mutable_params:
                calls_by_var[var].append(path)
    out = [LenCacheKnob(tuple(paths[:8]), var) for var, paths in calls_by_var.items() if len(paths) >= 2]
    return out[:4]
AutoLiftAnalyzer.discover_len_cache_sites = _discover_len_cache_sites


_old_discover_knobs = AutoLiftAnalyzer.discover_knobs
def _discover_knobs_o11(self):
    return _old_discover_knobs(self)
AutoLiftAnalyzer.discover_knobs = _discover_knobs_o11


_old_make_autolift_operators = _make_autolift_operators
def _make_autolift_operators(analyzer: AutoLiftAnalyzer) -> List[AutoLiftOperatorSpec]:
    specs = list(_old_make_autolift_operators(analyzer))
    existing = {spec.op.name for spec in specs}
    func_name = analyzer.func_name

    def add(name: str, fn: Callable[[AutoLiftCandidate], AutoLiftCandidate], risk: str, rationale: str) -> None:
        if name not in existing:
            specs.append(AutoLiftOperatorSpec(DomainOperator(name, fn), risk, rationale))
            existing.add(name)

    for knob in analyzer.discover_direct_memberships():
        op_name = f"direct_setlift_{len(existing)}"
        def make_op(knob=knob):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                node = _locate_by_path(_get_func_def(mod, func_name), knob.compare_path)
                if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
                    return c
                comp = node.comparators[0]
                if not isinstance(comp, (ast.List, ast.Tuple)):
                    return c
                node.comparators[0] = ast.Set(elts=list(comp.elts))
                return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"direct_setlift@{knob.compare_path}",))
            return fn
        add(op_name, make_op(), "safe", "materialize direct literal membership into a set")

    for knob in analyzer.discover_len_cache_sites():
        op_name = f"cache_len_{knob.var_name}"
        def make_len_cache(knob=knob):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                cache_name = f"_forge_len_{knob.var_name}"
                # avoid collisions
                existing_names = {n.id for n in ast.walk(fn_def) if isinstance(n, ast.Name)}
                if cache_name in existing_names:
                    return c
                inserted = ast.Assign(targets=[ast.Name(id=cache_name, ctx=ast.Store())], value=ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[ast.Name(id=knob.var_name, ctx=ast.Load())], keywords=[]))
                fn_def.body.insert(0, inserted)
                replaced = 0
                for path in knob.call_paths:
                    node = _locate_by_path(fn_def, path)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
                        _replace_by_path(fn_def, path, ast.Name(id=cache_name, ctx=ast.Load()))
                        replaced += 1
                if replaced == 0:
                    return c
                return AutoLiftCandidate(func_name, ast.unparse(ast.fix_missing_locations(mod)), c.ir_summary, c.discovered_knobs + (f"cache_len:{knob.var_name}",))
            return fn
        add(op_name, make_len_cache(), "safe", "cache repeated len(parameter) calls when the parameter is not reassigned")

    # Deep-IR-informed priority combo: if both opportunities exist, encourage a composite operator.
    opp_kinds = {o.kind for o in analyzer.deep_ir.opportunities}
    if "membership_materialization" in opp_kinds and "call_caching" in opp_kinds:
        def combo(c: AutoLiftCandidate) -> AutoLiftCandidate:
            cur = c
            for spec in list(specs):
                if spec.op.name.startswith("direct_setlift_") or spec.op.name.startswith("cache_len_"):
                    cur = spec.op.fn(cur)
            return cur
        add("deep_combo_materialize_cache", combo, "speculative", "apply two safe deep-IR opportunities together")
    return specs
globals()['_make_autolift_operators'] = _make_autolift_operators


@dataclass
class DeepIRDemo:
    source: str
    ir_summary: str
    opportunities: List[str]
    loops: List[str]
    guards: List[str]
    reductions: List[str]


def _deep_ir_sample(nums: List[int], forbidden: Tuple[int, ...], limit: int) -> bool:
    hits = 0
    size = len(nums)
    for i in range(len(nums)):
        x = nums[i]
        if x in forbidden and x > limit:
            hits += 1
    return hits > size - 1


def deep_ir_demo() -> DeepIRDemo:
    analyzer = AutoLiftAnalyzer(_deep_ir_sample)
    deep = analyzer.deep_ir
    return DeepIRDemo(
        source=analyzer.source,
        ir_summary=deep.summary(),
        opportunities=[f"{o.kind}:{o.target}->{o.recommendation}" for o in deep.opportunities],
        loops=[f"{l.iterator_kind}:{l.iterable_repr}:guards={l.guard_count}:reductions={list(l.reduction_targets)}" for l in deep.loops],
        guards=[g.test_repr for g in deep.guards[:6]],
        reductions=[f"{r.target}:{r.op}:{r.update_repr}" for r in deep.reductions[:6]],
    )


_old_build_release_report = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report(bundles, auto)
    demo = deep_ir_demo()
    extra = []
    extra.append("## Deep IR foundation (o1.1)")
    extra.append("- Autolift now builds a deeper normalized IR summary before operator synthesis.")
    extra.append("- The deeper IR tracks loops, guards, reductions, memberships, repeated calls, and structural opportunities.")
    extra.append("- New deep-IR-driven operators include direct literal set lifting and repeated len(...) caching.")
    extra.append(f"- Deep IR summary: {demo.ir_summary}")
    extra.append(f"- Deep opportunities: {', '.join(demo.opportunities[:8]) if demo.opportunities else 'none'}")
    extra.append(f"- Deep loops: {', '.join(demo.loops[:6]) if demo.loops else 'none'}")
    extra.append(f"- Deep reductions: {', '.join(demo.reductions[:6]) if demo.reductions else 'none'}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report


_old_package_info = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info())
    info["version"] = __version__
    info["script_name"] = "forge_o1_1.py"
    cmds = list(info.get("commands", []))
    if "deep-ir-demo" not in cmds:
        cmds.append("deep-ir-demo")
    info["commands"] = cmds
    info["deep_ir"] = "foundation"
    return info
globals()['package_info'] = package_info


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19))
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41))
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_1_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.1")
    print("deep IR foundation active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto}
globals()['run_release_demo'] = run_release_demo


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_1", description="Forge o1.1 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.1 / autolift-demo")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
        print("FORGE o1.1 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_1_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.1 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("loops:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli




# ===== Forge o1.2 Ultra deep IR =====

__version__ = "o1.2"


@dataclass(frozen=True)
class DeepIndexScanPattern:
    loop_path: Tuple[Any, ...]
    seq_name: str
    idx_var: str
    value_var: str
    assignment_path: Optional[Tuple[Any, ...]]
    condition_repr: str
    action_kind: str
    direct_iter_safe: bool


@dataclass(frozen=True)
class DeepBoolCommitPattern:
    flag_name: str
    init_path: Tuple[Any, ...]
    loop_path: Tuple[Any, ...]
    if_path: Tuple[Any, ...]
    assign_true_path: Tuple[Any, ...]
    return_path: Tuple[Any, ...]


@dataclass(frozen=True)
class DeepParamMembershipPattern:
    compare_path: Tuple[Any, ...]
    container_name: str
    negated: bool
    safe_materialize: bool


@dataclass(frozen=True)
class UltraDeepFunctionIR:
    function_name: str
    loops: Tuple[DeepLoopInfo, ...]
    guards: Tuple[DeepGuardInfo, ...]
    reductions: Tuple[DeepReductionInfo, ...]
    memberships: Tuple[DeepMembershipInfo, ...]
    repeated_calls: Tuple[DeepCallInfo, ...]
    thresholds: Tuple[str, ...]
    opportunities: Tuple[DeepDataStructureOpportunity, ...]
    index_scans: Tuple[DeepIndexScanPattern, ...]
    bool_commits: Tuple[DeepBoolCommitPattern, ...]
    param_memberships: Tuple[DeepParamMembershipPattern, ...]

    def summary(self) -> str:
        parts = [
            f"loops:{len(self.loops)}",
            f"guards:{len(self.guards)}",
            f"reductions:{len(self.reductions)}",
            f"memberships:{len(self.memberships)}",
            f"repeated_calls:{len(self.repeated_calls)}",
            f"thresholds:{len(self.thresholds)}",
            f"opportunities:{len(self.opportunities)}",
            f"index_scans:{len(self.index_scans)}",
            f"bool_commits:{len(self.bool_commits)}",
            f"param_memberships:{len(self.param_memberships)}",
        ]
        if self.opportunities:
            lead = ", ".join(f"{o.kind}->{o.recommendation}" for o in self.opportunities[:5])
            parts.append(f"lead:{lead}")
        return ", ".join(parts)


def _count_name_uses(node: ast.AST, name: str) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == name)


def _is_name_subscript_of(node: ast.AST, seq_name: str, idx_name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == seq_name
        and isinstance(node.slice, ast.Name)
        and node.slice.id == idx_name
    )


def _replace_index_subscripts(node: ast.AST, seq_name: str, idx_name: str, value_name: str) -> ast.AST:
    class Rewriter(ast.NodeTransformer):
        def visit_Subscript(self, sub: ast.Subscript):
            self.generic_visit(sub)
            if _is_name_subscript_of(sub, seq_name, idx_name):
                return ast.copy_location(ast.Name(id=value_name, ctx=ast.Load()), sub)
            return sub
    return Rewriter().visit(node)


def _delete_stmt_by_path(root: ast.AST, path: Tuple[Any, ...]) -> bool:
    if len(path) < 2:
        return False
    container_path = path[:-1]
    idx = path[-1]
    if not isinstance(idx, int):
        return False
    container = _locate_by_path(root, container_path[:-1]) if len(container_path) >= 1 and isinstance(container_path[-1], str) else None
    if container is None:
        return False
    field = container_path[-1]
    if not isinstance(field, str):
        return False
    seq = getattr(container, field, None)
    if not isinstance(seq, list) or idx >= len(seq):
        return False
    del seq[idx]
    return True


class UltraDeepIRBuilder:
    def __init__(self, fn_def: ast.FunctionDef, audit: Dict[str, Any]):
        self.fn_def = fn_def
        self.audit = audit
        self.base = DeepIRBuilder(fn_def, audit).build()

    def _assigned_names(self) -> set[str]:
        out: set[str] = set()
        for n in ast.walk(self.fn_def):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
        return out

    def _discover_bool_commits(self) -> List[DeepBoolCommitPattern]:
        patterns: List[DeepBoolCommitPattern] = []
        body = self.fn_def.body
        for i, stmt in enumerate(body):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Constant) and stmt.value.value is False):
                continue
            flag = stmt.targets[0].id
            return_path = None
            for j in range(i + 1, len(body)):
                tail = body[j]
                if isinstance(tail, ast.Return) and isinstance(tail.value, ast.Name) and tail.value.id == flag:
                    return_path = ('body', j)
                    break
            if return_path is None:
                continue
            for j in range(i + 1, len(body)):
                loop = body[j]
                if not isinstance(loop, ast.For):
                    continue
                loop_path = ('body', j)
                for k, inner in enumerate(loop.body):
                    if not isinstance(inner, ast.If):
                        continue
                    if len(inner.body) != 1:
                        continue
                    assign = inner.body[0]
                    if isinstance(assign, ast.Assign) and len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name) and assign.targets[0].id == flag and isinstance(assign.value, ast.Constant) and assign.value.value is True:
                        patterns.append(
                            DeepBoolCommitPattern(
                                flag_name=flag,
                                init_path=('body', i),
                                loop_path=loop_path,
                                if_path=('body', j, 'body', k),
                                assign_true_path=('body', j, 'body', k, 'body', 0),
                                return_path=return_path,
                            )
                        )
        return patterns

    def _discover_param_memberships(self) -> List[DeepParamMembershipPattern]:
        params = {arg.arg for arg in self.fn_def.args.args}
        assigned = self._assigned_names()
        patterns: List[DeepParamMembershipPattern] = []
        for path, node in _iter_child_paths(self.fn_def):
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name):
                name = node.comparators[0].id
                if name in params:
                    patterns.append(
                        DeepParamMembershipPattern(
                            compare_path=path,
                            container_name=name,
                            negated=isinstance(node.ops[0], ast.NotIn),
                            safe_materialize=name not in assigned,
                        )
                    )
        return patterns

    def _discover_index_scans(self) -> List[DeepIndexScanPattern]:
        out: List[DeepIndexScanPattern] = []
        for path, node in _iter_child_paths(self.fn_def):
            if not isinstance(node, ast.For):
                continue
            if not (isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range' and len(node.iter.args) == 1):
                continue
            arg = node.iter.args[0]
            if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'len' and len(arg.args) == 1 and isinstance(arg.args[0], ast.Name)):
                continue
            idx_var = node.target.id
            seq_name = arg.args[0].id
            value_var = f"{seq_name}_item"
            assignment_path = None
            condition_repr = "none"
            action_kind = "generic"
            direct_iter_safe = False

            if node.body and isinstance(node.body[0], ast.Assign) and len(node.body[0].targets) == 1 and isinstance(node.body[0].targets[0], ast.Name):
                first = node.body[0]
                if _is_name_subscript_of(first.value, seq_name, idx_var):
                    assignment_path = path + ('body', 0)
                    value_var = first.targets[0].id

            loop_name_uses = _count_name_uses(node, idx_var)
            subscript_uses = sum(1 for n in ast.walk(node) if _is_name_subscript_of(n, seq_name, idx_var))
            direct_iter_safe = loop_name_uses == subscript_uses
            for inner in ast.walk(node):
                if isinstance(inner, ast.If):
                    condition_repr = _ast_text(inner.test)
                    if any(isinstance(s, ast.Assign) for s in inner.body):
                        action_kind = "flag_or_assign"
                    elif any(isinstance(s, ast.AugAssign) for s in inner.body):
                        action_kind = "reduction"
                    elif any(isinstance(s, ast.Return) for s in inner.body):
                        action_kind = "return"
                    break
            out.append(
                DeepIndexScanPattern(
                    loop_path=path,
                    seq_name=seq_name,
                    idx_var=idx_var,
                    value_var=value_var,
                    assignment_path=assignment_path,
                    condition_repr=condition_repr,
                    action_kind=action_kind,
                    direct_iter_safe=direct_iter_safe,
                )
            )
        return out

    def build(self) -> UltraDeepFunctionIR:
        bool_commits = self._discover_bool_commits()
        param_memberships = self._discover_param_memberships()
        index_scans = self._discover_index_scans()
        opportunities = list(self.base.opportunities)

        for pat in index_scans:
            if pat.direct_iter_safe:
                opportunities.append(
                    DeepDataStructureOpportunity(
                        kind="indexed_scan",
                        target=f"{pat.idx_var} over {pat.seq_name}",
                        recommendation="direct_iteration",
                        reason="loop iterates over range(len(seq)) and only uses the index for seq[idx]"
                    )
                )
        for pat in bool_commits:
            opportunities.append(
                DeepDataStructureOpportunity(
                    kind="boolean_commit",
                    target=pat.flag_name,
                    recommendation="early_return",
                    reason="flag is only committed inside a loop and returned at the end"
                )
            )
        for pat in param_memberships:
            if pat.safe_materialize:
                opportunities.append(
                    DeepDataStructureOpportunity(
                        kind="parameter_membership",
                        target=pat.container_name,
                        recommendation="materialize_set",
                        reason="parameter membership checks can be lifted into a local set"
                    )
                )

        uniq: List[DeepDataStructureOpportunity] = []
        seen = set()
        for o in opportunities:
            key = (o.kind, o.target, o.recommendation)
            if key not in seen:
                seen.add(key)
                uniq.append(o)

        return UltraDeepFunctionIR(
            function_name=self.base.function_name,
            loops=self.base.loops,
            guards=self.base.guards,
            reductions=self.base.reductions,
            memberships=self.base.memberships,
            repeated_calls=self.base.repeated_calls,
            thresholds=self.base.thresholds,
            opportunities=tuple(uniq),
            index_scans=tuple(index_scans),
            bool_commits=tuple(bool_commits),
            param_memberships=tuple(param_memberships),
        )


_old_autolift_init_o12 = AutoLiftAnalyzer.__init__
def _autolift_init_o12(self, func: Callable[..., Any]):
    _old_autolift_init_o12(self, func)
    self.deep_ir = UltraDeepIRBuilder(self.fn_def, self.audit).build()
AutoLiftAnalyzer.__init__ = _autolift_init_o12


_old_make_candidate_o12 = AutoLiftAnalyzer.make_candidate
def _make_candidate_o12(self) -> AutoLiftCandidate:
    deep = self.deep_ir
    deep_knobs = tuple(
        f"{o.kind}:{o.recommendation}"
        for o in deep.opportunities[:10]
    )
    return AutoLiftCandidate(self.func_name, self.source, deep.summary(), deep_knobs)
AutoLiftAnalyzer.make_candidate = _make_candidate_o12


def _discover_deep_patterns(self):
    return self.deep_ir.index_scans, self.deep_ir.bool_commits, self.deep_ir.param_memberships
AutoLiftAnalyzer.discover_deep_patterns = _discover_deep_patterns


_old_make_autolift_operators_o12 = _make_autolift_operators
def _make_autolift_operators(analyzer: AutoLiftAnalyzer) -> List[AutoLiftOperatorSpec]:
    specs = list(_old_make_autolift_operators_o12(analyzer))
    existing = {spec.op.name for spec in specs}
    func_name = analyzer.func_name
    index_scans, bool_commits, param_memberships = analyzer.discover_deep_patterns()

    def add(name: str, fn: Callable[[AutoLiftCandidate], AutoLiftCandidate], risk: str, rationale: str) -> None:
        if name not in existing:
            specs.append(AutoLiftOperatorSpec(DomainOperator(name, fn), risk, rationale))
            existing.add(name)

    for pat in index_scans[:4]:
        if not pat.direct_iter_safe:
            continue
        def make_direct_iter(pat=pat):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                loop = _locate_by_path(fn_def, pat.loop_path)
                if not isinstance(loop, ast.For):
                    return c
                loop.iter = ast.Name(id=pat.seq_name, ctx=ast.Load())
                loop.target = ast.Name(id=pat.value_var, ctx=ast.Store())
                if pat.assignment_path is not None:
                    if not _delete_stmt_by_path(fn_def, pat.assignment_path):
                        return c
                for i, stmt in enumerate(loop.body):
                    loop.body[i] = _replace_index_subscripts(stmt, pat.seq_name, pat.idx_var, pat.value_var)
                src = ast.unparse(ast.fix_missing_locations(mod))
                return AutoLiftCandidate(func_name, src, c.ir_summary, c.discovered_knobs + (f"direct_iter:{pat.seq_name}",))
            return fn
        add(f"deep_direct_iter_{pat.seq_name}", make_direct_iter(), "safe", "replace range(len(seq)) index scan with direct iteration")

    for pat in bool_commits[:4]:
        def make_bool_commit(pat=pat):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                if_node = _locate_by_path(fn_def, pat.if_path)
                ret_node = _locate_by_path(fn_def, pat.return_path)
                if not isinstance(if_node, ast.If) or not isinstance(ret_node, ast.Return):
                    return c
                if len(if_node.body) != 1:
                    return c
                if_node.body[0] = ast.Return(value=ast.Constant(True))
                ret_node.value = ast.Constant(False)
                _delete_stmt_by_path(fn_def, pat.init_path)
                src = ast.unparse(ast.fix_missing_locations(mod))
                return AutoLiftCandidate(func_name, src, c.ir_summary, c.discovered_knobs + (f"bool_commit:{pat.flag_name}",))
            return fn
        add(f"deep_bool_commit_{pat.flag_name}", make_bool_commit(), "safe", "replace boolean flag commit with early return and terminal false")

    for pat in param_memberships[:4]:
        if not pat.safe_materialize:
            continue
        def make_param_setlift(pat=pat):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                cache_name = f"_forge_set_{pat.container_name}"
                existing_names = {n.id for n in ast.walk(fn_def) if isinstance(n, ast.Name)}
                if cache_name in existing_names:
                    return c
                fn_def.body.insert(
                    0,
                    ast.Assign(
                        targets=[ast.Name(id=cache_name, ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id='set', ctx=ast.Load()), args=[ast.Name(id=pat.container_name, ctx=ast.Load())], keywords=[]),
                    ),
                )
                class Rewriter(ast.NodeTransformer):
                    def visit_Compare(self, node: ast.Compare):
                        self.generic_visit(node)
                        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == pat.container_name:
                            node.comparators[0] = ast.Name(id=cache_name, ctx=ast.Load())
                        return node
                fn_def = Rewriter().visit(fn_def)
                src = ast.unparse(ast.fix_missing_locations(mod))
                return AutoLiftCandidate(func_name, src, c.ir_summary, c.discovered_knobs + (f"param_setlift:{pat.container_name}",))
            return fn
        add(f"deep_param_setlift_{pat.container_name}", make_param_setlift(), "safe", "materialize parameter membership checks into a local set")

    return specs
globals()['_make_autolift_operators'] = _make_autolift_operators


def demo_autolift_function_o12(tokens: List[str]) -> bool:
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    found = False
    limit = len(tokens)
    for idx in range(limit):
        tok = tokens[idx]
        if tok in forbidden:
            found = True
    return found


def autolift_demo() -> ImproveResult:
    tests = [
        ((["safe", "noop"],), False),
        ((["DROP"],), True),
        ((["safe", "ALTER"],), True),
        ((["x", "y", "z"],), False),
        ((["DELETE", "x", "y"],), True),
    ]
    return improve(demo_autolift_function_o12, tests)
globals()['autolift_demo'] = autolift_demo


def _deep_ir_sample_o12(nums: List[int], forbidden: Tuple[int, ...], limit: int) -> bool:
    found = False
    for idx in range(len(nums)):
        value = nums[idx]
        if value in forbidden and value > limit:
            found = True
    return found


def deep_ir_demo() -> DeepIRDemo:
    analyzer = AutoLiftAnalyzer(_deep_ir_sample_o12)
    deep = analyzer.deep_ir
    return DeepIRDemo(
        source=analyzer.source,
        ir_summary=deep.summary(),
        opportunities=[f"{o.kind}:{o.target}->{o.recommendation}" for o in deep.opportunities],
        loops=[f"{l.iterator_kind}:{l.iterable_repr}:guards={l.guard_count}:reductions={list(l.reduction_targets)}" for l in deep.loops],
        guards=[g.test_repr for g in deep.guards[:6]],
        reductions=[f"{r.target}:{r.op}:{r.update_repr}" for r in deep.reductions[:6]],
    )
globals()['deep_ir_demo'] = deep_ir_demo


_old_build_release_report_o12 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o12(bundles, auto)
    demo = deep_ir_demo()
    extra = []
    extra.append("## Ultra deep IR (o1.2)")
    extra.append("- Autolift now derives semantic patterns, not just structural counts.")
    extra.append("- Deep IR identifies indexed scans, boolean commit loops, and safe parameter-membership materialization sites.")
    extra.append("- New deep operators synthesize direct iteration, early-return boolean commit elimination, and parameter set lifting.")
    extra.append(f"- Ultra deep IR summary: {demo.ir_summary}")
    extra.append(f"- Ultra deep opportunities: {', '.join(demo.opportunities[:10]) if demo.opportunities else 'none'}")
    extra.append(f"- Ultra deep loops: {', '.join(demo.loops[:6]) if demo.loops else 'none'}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report


_old_package_info_o12 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o12())
    info["version"] = __version__
    info["script_name"] = "forge_o1_2.py"
    cmds = list(info.get("commands", []))
    if "deep-ir-demo" not in cmds:
        cmds.append("deep-ir-demo")
    info["commands"] = cmds
    info["deep_ir"] = "ultra"
    return info
globals()['package_info'] = package_info


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19))
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41))
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_2_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.2")
    print("ultra deep IR active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto}
globals()['run_release_demo'] = run_release_demo


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_2", description="Forge o1.2 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.2 / autolift-demo")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
        print("FORGE o1.2 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_2_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.2 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("loops:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli




# ===== Forge o1.3 Hyper deep IR =====

__version__ = "o1.3"

@dataclass(frozen=True)
class DeepExistentialPattern:
    flag_name: str
    init_index: int
    loop_index: int
    return_index: int
    iterable_expr: str
    bound_var: str
    predicate_repr: str
    source_kind: str

@dataclass(frozen=True)
class DeepUniversalPattern:
    counter_name: str
    init_index: int
    loop_index: int
    return_index: int
    iterable_expr: str
    bound_var: str
    predicate_repr: str
    source_kind: str
    compare_repr: str

@dataclass(frozen=True)
class HyperDeepFunctionIR:
    function_name: str
    base_summary: str
    opportunities: Tuple[DeepDataStructureOpportunity, ...]
    existentials: Tuple[DeepExistentialPattern, ...]
    universals: Tuple[DeepUniversalPattern, ...]

    def summary(self) -> str:
        parts = [
            self.base_summary,
            f"existentials:{len(self.existentials)}",
            f"universals:{len(self.universals)}",
            f"opportunities:{len(self.opportunities)}",
        ]
        if self.opportunities:
            parts.append("lead:" + ", ".join(f"{o.kind}->{o.recommendation}" for o in self.opportunities[:6]))
        return ", ".join(parts)

def _o13_is_bool_false_assign(stmt: ast.stmt) -> Optional[str]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Constant) and stmt.value.value is False:
        return stmt.targets[0].id
    return None

def _o13_is_zero_assign(stmt: ast.stmt) -> Optional[str]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Constant) and stmt.value.value == 0:
        return stmt.targets[0].id
    return None

def _o13_extract_bound_from_loop(loop: ast.For) -> Tuple[Optional[str], Optional[str], str]:
    if isinstance(loop.target, ast.Name):
        if isinstance(loop.iter, ast.Name):
            return loop.target.id, _ast_text(loop.iter), "direct_iter"
        if isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Name) and loop.iter.func.id == "range" and len(loop.iter.args) == 1:
            arg = loop.iter.args[0]
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "len" and len(arg.args) == 1 and isinstance(arg.args[0], ast.Name):
                seq_name = arg.args[0].id
                if loop.body and isinstance(loop.body[0], ast.Assign) and len(loop.body[0].targets) == 1 and isinstance(loop.body[0].targets[0], ast.Name):
                    first = loop.body[0]
                    if _is_name_subscript_of(first.value, seq_name, loop.target.id):
                        idx_name = loop.target.id
                        value_name = first.targets[0].id
                        uses = _count_name_uses(loop, idx_name)
                        sub_uses = sum(1 for n in ast.walk(loop) if _is_name_subscript_of(n, seq_name, idx_name))
                        if uses == sub_uses:
                            return value_name, seq_name, "index_scan"
    return None, None, "unknown"

def _o13_find_if_true_assign(loop: ast.For, name: str) -> Optional[ast.If]:
    for stmt in loop.body:
        if isinstance(stmt, ast.If) and len(stmt.body) == 1:
            inner = stmt.body[0]
            if isinstance(inner, ast.Assign) and len(inner.targets) == 1 and isinstance(inner.targets[0], ast.Name) and inner.targets[0].id == name and isinstance(inner.value, ast.Constant) and inner.value.value is True:
                return stmt
    return None

def _o13_find_if_counter_increment(loop: ast.For, name: str) -> Optional[ast.If]:
    for stmt in loop.body:
        if isinstance(stmt, ast.If) and len(stmt.body) == 1:
            inner = stmt.body[0]
            if isinstance(inner, ast.AugAssign) and isinstance(inner.target, ast.Name) and inner.target.id == name and isinstance(inner.op, ast.Add) and isinstance(inner.value, ast.Constant) and inner.value.value == 1:
                return stmt
            if isinstance(inner, ast.Assign) and len(inner.targets) == 1 and isinstance(inner.targets[0], ast.Name) and inner.targets[0].id == name and isinstance(inner.value, ast.BinOp) and isinstance(inner.value.op, ast.Add) and _contains_name(inner.value, name):
                return stmt
    return None

def _o13_compare_is_any(counter: str, node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name) or node.left.id != counter or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    comp = node.comparators[0]
    op = node.ops[0]
    return (
        isinstance(op, ast.Gt) and isinstance(comp, ast.Constant) and comp.value == 0
    ) or (
        isinstance(op, ast.NotEq) and isinstance(comp, ast.Constant) and comp.value == 0
    ) or (
        isinstance(op, ast.GtE) and isinstance(comp, ast.Constant) and comp.value == 1
    )

def _o13_compare_is_all(counter: str, node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name) or node.left.id != counter or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    comp = node.comparators[0]
    op = node.ops[0]
    if isinstance(op, ast.Eq):
        if isinstance(comp, ast.Call) and isinstance(comp.func, ast.Name) and comp.func.id == "len" and len(comp.args) == 1:
            return True
        if isinstance(comp, ast.Name):
            return True
    if isinstance(op, ast.Gt) and isinstance(comp, ast.BinOp) and isinstance(comp.op, ast.Sub):
        return True
    if isinstance(op, ast.GtE) and isinstance(comp, (ast.Call, ast.Name)):
        return True
    return False

class HyperDeepIRBuilder:
    def __init__(self, fn_def: ast.FunctionDef, audit: Dict[str, Any], base_ir: Optional[UltraDeepFunctionIR] = None):
        self.fn_def = fn_def
        self.audit = audit
        self.base = base_ir if base_ir is not None else UltraDeepIRBuilder(fn_def, audit).build()

    def _discover_existentials(self) -> List[DeepExistentialPattern]:
        out: List[DeepExistentialPattern] = []
        body = self.fn_def.body
        for i, stmt in enumerate(body):
            flag = _o13_is_bool_false_assign(stmt)
            if not flag:
                continue
            for j in range(i + 1, len(body)):
                loop = body[j]
                if not isinstance(loop, ast.For):
                    continue
                cond_if = _o13_find_if_true_assign(loop, flag)
                if cond_if is None:
                    continue
                for k in range(j + 1, len(body)):
                    ret = body[k]
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name) and ret.value.id == flag:
                        bound_var, iterable_expr, source_kind = _o13_extract_bound_from_loop(loop)
                        if bound_var and iterable_expr:
                            pred = cond_if.test
                            if source_kind == "index_scan" and isinstance(loop.target, ast.Name):
                                pred = _replace_index_subscripts(ast.parse(_ast_text(cond_if.test), mode='eval').body, iterable_expr, loop.target.id, bound_var)
                            out.append(DeepExistentialPattern(flag, i, j, k, iterable_expr, bound_var, _ast_text(pred), source_kind))
                        break
        return out

    def _discover_universals(self) -> List[DeepUniversalPattern]:
        out: List[DeepUniversalPattern] = []
        body = self.fn_def.body
        for i, stmt in enumerate(body):
            counter = _o13_is_zero_assign(stmt)
            if not counter:
                continue
            for j in range(i + 1, len(body)):
                loop = body[j]
                if not isinstance(loop, ast.For):
                    continue
                cond_if = _o13_find_if_counter_increment(loop, counter)
                if cond_if is None:
                    continue
                for k in range(j + 1, len(body)):
                    ret = body[k]
                    if not isinstance(ret, ast.Return):
                        continue
                    mode = None
                    if _o13_compare_is_any(counter, ret.value):
                        mode = "any"
                    elif _o13_compare_is_all(counter, ret.value):
                        mode = "all"
                    if mode is None:
                        continue
                    bound_var, iterable_expr, source_kind = _o13_extract_bound_from_loop(loop)
                    if bound_var and iterable_expr:
                        pred = cond_if.test
                        if source_kind == "index_scan" and isinstance(loop.target, ast.Name):
                            pred = _replace_index_subscripts(ast.parse(_ast_text(cond_if.test), mode='eval').body, iterable_expr, loop.target.id, bound_var)
                        out.append(DeepUniversalPattern(counter, i, j, k, iterable_expr, bound_var, _ast_text(pred), mode, _ast_text(ret.value)))
                    break
        return out

    def build(self) -> HyperDeepFunctionIR:
        existentials = self._discover_existentials()
        universals = self._discover_universals()
        opps = list(self.base.opportunities)
        for pat in existentials:
            opps.append(DeepDataStructureOpportunity("existential_scan", pat.flag_name, "any_generator", "loop implements existential search and can collapse to any(...)"))
        for pat in universals:
            opps.append(DeepDataStructureOpportunity("quantified_counter", pat.counter_name, "any_generator" if pat.source_kind == "any" else "all_generator", "counter-based quantification can collapse to a generator form"))
        uniq: List[DeepDataStructureOpportunity] = []
        seen = set()
        for o in opps:
            key = (o.kind, o.target, o.recommendation)
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        return HyperDeepFunctionIR(self.base.function_name, self.base.summary(), tuple(uniq), tuple(existentials), tuple(universals))

_old_o13_init = AutoLiftAnalyzer.__init__
def _o13_init(self, func: Callable[..., Any]):
    _old_o13_init(self, func)
    self.deep_base_ir = self.deep_ir
    self.deep_ir = HyperDeepIRBuilder(self.fn_def, self.audit, self.deep_base_ir).build()
AutoLiftAnalyzer.__init__ = _o13_init

def _o13_make_candidate(self) -> AutoLiftCandidate:
    deep = self.deep_ir
    deep_knobs = tuple(f"{o.kind}:{o.recommendation}" for o in deep.opportunities[:12])
    return AutoLiftCandidate(self.func_name, self.source, deep.summary(), deep_knobs)
AutoLiftAnalyzer.make_candidate = _o13_make_candidate

def _o13_discover_deep_patterns(self):
    base = getattr(self, "deep_base_ir", None)
    if base is None:
        return tuple(), tuple(), tuple()
    return base.index_scans, base.bool_commits, base.param_memberships
AutoLiftAnalyzer.discover_deep_patterns = _o13_discover_deep_patterns

def _o13_discover_hyper_patterns(self):
    return self.deep_ir.existentials, self.deep_ir.universals
AutoLiftAnalyzer.discover_hyper_patterns = _o13_discover_hyper_patterns

def _o13_replace_body_with_quantifier(fn_def: ast.FunctionDef, init_index: int, loop_index: int, return_index: int, new_return: ast.Return) -> None:
    rebuilt: List[ast.stmt] = []
    for idx, stmt in enumerate(fn_def.body):
        if idx == init_index:
            rebuilt.append(new_return)
            continue
        if idx in {loop_index, return_index}:
            continue
        rebuilt.append(stmt)
    fn_def.body = rebuilt

_old_o13_make_ops = _make_autolift_operators
def _make_autolift_operators(analyzer: AutoLiftAnalyzer) -> List[AutoLiftOperatorSpec]:
    specs = list(_old_o13_make_ops(analyzer))
    existing = {spec.op.name for spec in specs}
    func_name = analyzer.func_name
    existentials, universals = analyzer.discover_hyper_patterns()

    def add(name: str, fn: Callable[[AutoLiftCandidate], AutoLiftCandidate], risk: str, rationale: str) -> None:
        if name not in existing:
            specs.append(AutoLiftOperatorSpec(DomainOperator(name, fn), risk, rationale))
            existing.add(name)

    for pat in existentials[:4]:
        def make_any(pat=pat):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                predicate = ast.parse(pat.predicate_repr, mode='eval').body
                gen = ast.GeneratorExp(elt=predicate, generators=[ast.comprehension(target=ast.Name(id=pat.bound_var, ctx=ast.Store()), iter=ast.Name(id=pat.iterable_expr, ctx=ast.Load()), ifs=[], is_async=0)])
                new_return = ast.Return(value=ast.Call(func=ast.Name(id='any', ctx=ast.Load()), args=[gen], keywords=[]))
                _o13_replace_body_with_quantifier(fn_def, pat.init_index, pat.loop_index, pat.return_index, new_return)
                src = ast.unparse(ast.fix_missing_locations(mod))
                return AutoLiftCandidate(func_name, src, c.ir_summary, c.discovered_knobs + (f"any:{pat.bound_var}",))
            return fn
        add(f"deep_any_{pat.flag_name}", make_any(), "safe", "collapse existential loop into any(generator)")

    for pat in universals[:4]:
        def make_q(pat=pat):
            def fn(c: AutoLiftCandidate) -> AutoLiftCandidate:
                mod = ast.parse(c.source)
                fn_def = _get_func_def(mod, func_name)
                predicate = ast.parse(pat.predicate_repr, mode='eval').body
                gen = ast.GeneratorExp(elt=predicate, generators=[ast.comprehension(target=ast.Name(id=pat.bound_var, ctx=ast.Store()), iter=ast.Name(id=pat.iterable_expr, ctx=ast.Load()), ifs=[], is_async=0)])
                qname = 'any' if pat.source_kind == 'any' else 'all'
                new_return = ast.Return(value=ast.Call(func=ast.Name(id=qname, ctx=ast.Load()), args=[gen], keywords=[]))
                _o13_replace_body_with_quantifier(fn_def, pat.init_index, pat.loop_index, pat.return_index, new_return)
                src = ast.unparse(ast.fix_missing_locations(mod))
                return AutoLiftCandidate(func_name, src, c.ir_summary, c.discovered_knobs + (f"{qname}:{pat.bound_var}",))
            return fn
        add(f"deep_{pat.source_kind}_{pat.counter_name}", make_q(), "safe", f"collapse quantified counter into {pat.source_kind}(generator)")
    return specs
globals()['_make_autolift_operators'] = _make_autolift_operators

def demo_autolift_function_o13(tokens: List[str]) -> bool:
    forbidden = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']
    found = False
    limit = len(tokens)
    for idx in range(limit):
        tok = tokens[idx]
        if tok in forbidden:
            found = True
    return found

def autolift_demo() -> ImproveResult:
    tests = [
        ((['safe', 'noop'],), False),
        ((['DROP'],), True),
        ((['safe', 'ALTER'],), True),
        ((['x', 'y', 'z'],), False),
        ((['DELETE', 'x', 'y'],), True),
    ]
    return improve(demo_autolift_function_o13, tests)
globals()['autolift_demo'] = autolift_demo

def _deep_ir_sample_o13(nums: List[int], forbidden: Tuple[int, ...], limit: int) -> bool:
    found = False
    for idx in range(len(nums)):
        value = nums[idx]
        if value in forbidden and value > limit:
            found = True
    return found

def deep_ir_demo() -> DeepIRDemo:
    analyzer = AutoLiftAnalyzer(_deep_ir_sample_o13)
    deep = analyzer.deep_ir
    return DeepIRDemo(
        source=analyzer.source,
        ir_summary=deep.summary(),
        opportunities=[f"{o.kind}:{o.target}->{o.recommendation}" for o in deep.opportunities],
        loops=[f"{e.bound_var} in {e.iterable_expr}:{e.predicate_repr}" for e in deep.existentials] + [f"{u.source_kind}:{u.bound_var} in {u.iterable_expr}:{u.predicate_repr}" for u in deep.universals],
        guards=[u.compare_repr for u in deep.universals[:6]],
        reductions=[f"existentials={len(deep.existentials)}", f"universals={len(deep.universals)}"],
    )
globals()['deep_ir_demo'] = deep_ir_demo

_old_o13_build_release = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_o13_build_release(bundles, auto)
    demo = deep_ir_demo()
    extra = []
    extra.append("## Hyper deep IR (o1.3)")
    extra.append("- Autolift now recognizes quantifier semantics, not just loop shapes.")
    extra.append("- Deep IR detects existential scans and quantified scans that can collapse to any(...) or all(...).")
    extra.append("- New operators synthesize generator quantifier rewrites directly from inferred semantics.")
    extra.append(f"- Hyper deep IR summary: {demo.ir_summary}")
    extra.append(f"- Hyper deep opportunities: {', '.join(demo.opportunities[:12]) if demo.opportunities else 'none'}")
    extra.append(f"- Hyper deep quantifiers: {', '.join(demo.loops[:8]) if demo.loops else 'none'}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report

_old_o13_package = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_o13_package())
    info["version"] = __version__
    info["script_name"] = "forge_o1_3.py"
    info["deep_ir"] = "hyper"
    return info
globals()['package_info'] = package_info

def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19))
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41))
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_3_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.3")
    print("hyper deep IR active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto}
globals()['run_release_demo'] = run_release_demo

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_3", description="Forge o1.3 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli

def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.3 / autolift-demo")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211))
        print("FORGE o1.3 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_3_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.3 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.4 Constitution Layer =====

__version__ = "o1.4"

@dataclass(frozen=True)
class ImprovementConstitution:
    name: str
    correctness: str = "strict"
    failure_penalty_per_case: float = 1000.0
    avg_cost_weight: float = 100.0
    worst_cost_weight: float = 20.0
    complexity_weight: float = 0.1
    edit_distance_weight: float = 0.2
    semantic_risk_penalty: float = 0.0
    prefer_worst_case_over_average: bool = False
    require_attack_survival: bool = False
    description: str = ""

    def summary(self) -> str:
        return (
            f"{self.name}: correctness={self.correctness}, failure={self.failure_penalty_per_case:.1f}, "
            f"avg={self.avg_cost_weight:.2f}, worst={self.worst_cost_weight:.2f}, "
            f"complexity={self.complexity_weight:.2f}, edit={self.edit_distance_weight:.2f}, "
            f"semantic_risk={self.semantic_risk_penalty:.2f}, "
            f"prefer_worst={self.prefer_worst_case_over_average}, "
            f"require_attack_survival={self.require_attack_survival}"
        )

def constitution_presets() -> Dict[str, ImprovementConstitution]:
    return {
        "optimize": ImprovementConstitution(
            name="optimize",
            correctness="strict",
            failure_penalty_per_case=1000.0,
            avg_cost_weight=100.0,
            worst_cost_weight=20.0,
            complexity_weight=0.10,
            edit_distance_weight=0.20,
            semantic_risk_penalty=0.10,
            description="Favor efficiency while preserving correctness."
        ),
        "repair": ImprovementConstitution(
            name="repair",
            correctness="strict",
            failure_penalty_per_case=1400.0,
            avg_cost_weight=120.0,
            worst_cost_weight=20.0,
            complexity_weight=0.15,
            edit_distance_weight=0.80,
            semantic_risk_penalty=0.25,
            description="Favor correctness recovery and low semantic drift."
        ),
        "harden": ImprovementConstitution(
            name="harden",
            correctness="strict",
            failure_penalty_per_case=1200.0,
            avg_cost_weight=38.0,
            worst_cost_weight=130.0,
            complexity_weight=0.10,
            edit_distance_weight=0.25,
            semantic_risk_penalty=0.35,
            prefer_worst_case_over_average=True,
            require_attack_survival=True,
            description="Favor worst-case robustness and adversarial survival."
        ),
        "balanced": ImprovementConstitution(
            name="balanced",
            correctness="strict",
            failure_penalty_per_case=1100.0,
            avg_cost_weight=75.0,
            worst_cost_weight=60.0,
            complexity_weight=0.12,
            edit_distance_weight=0.25,
            semantic_risk_penalty=0.20,
            description="Balanced improvement with moderate robustness pressure."
        ),
        "robust": ImprovementConstitution(
            name="robust",
            correctness="strict",
            failure_penalty_per_case=1250.0,
            avg_cost_weight=35.0,
            worst_cost_weight=155.0,
            complexity_weight=0.10,
            edit_distance_weight=0.30,
            semantic_risk_penalty=0.40,
            prefer_worst_case_over_average=True,
            require_attack_survival=True,
            description="Aggressively penalize brittle worst-case behavior."
        ),
    }

def forge_constitution(name: str = "balanced", **overrides: Any) -> ImprovementConstitution:
    presets = constitution_presets()
    base = presets.get(name, presets["balanced"])
    data = {
        "name": base.name,
        "correctness": base.correctness,
        "failure_penalty_per_case": base.failure_penalty_per_case,
        "avg_cost_weight": base.avg_cost_weight,
        "worst_cost_weight": base.worst_cost_weight,
        "complexity_weight": base.complexity_weight,
        "edit_distance_weight": base.edit_distance_weight,
        "semantic_risk_penalty": base.semantic_risk_penalty,
        "prefer_worst_case_over_average": base.prefer_worst_case_over_average,
        "require_attack_survival": base.require_attack_survival,
        "description": base.description,
    }
    data.update(overrides)
    return ImprovementConstitution(**data)

def _constitution_from_input(value: Any, mode: str) -> Optional[ImprovementConstitution]:
    presets = constitution_presets()
    if value is None:
        return None
    if isinstance(value, ImprovementConstitution):
        return value
    if isinstance(value, str):
        if value == "mode_default":
            return presets.get(mode, presets["balanced"])
        return presets.get(value, forge_constitution(value))
    if isinstance(value, dict):
        seed = value.get("name", "balanced")
        const = forge_constitution(seed)
        data = const.__dict__.copy()
        data.update(value)
        return ImprovementConstitution(**data)
    return None

def _domain_constitution(self: Any, mode: str) -> Optional[ImprovementConstitution]:
    value = getattr(self, "constitution", None)
    const = _constitution_from_input(value, mode)
    return const

def _constitution_semantic_risk(metrics: EvalMetrics) -> float:
    return max(0.0, metrics.complexity - 4.0)

def _rescore_with_constitution(metrics: EvalMetrics, mode: str, edit_distance: float, constitution: ImprovementConstitution) -> EvalMetrics:
    failures = max(0, metrics.total - metrics.correct)
    failure_multiplier = 1.25 if constitution.correctness == "strict" else 1.0
    failure_penalty = failures * constitution.failure_penalty_per_case * failure_multiplier
    semantic_risk = _constitution_semantic_risk(metrics) * constitution.semantic_risk_penalty
    scalar = (
        failure_penalty
        + metrics.avg_cost * constitution.avg_cost_weight
        + metrics.worst_cost * constitution.worst_cost_weight
        + metrics.complexity * constitution.complexity_weight
        + edit_distance * constitution.edit_distance_weight
        + semantic_risk
    )
    return EvalMetrics(metrics.correct, metrics.total, metrics.avg_cost, metrics.worst_cost, metrics.complexity, scalar, dict(metrics.failures))

def _wrap_domain_evaluate(cls: Any) -> None:
    old = cls.evaluate
    def wrapped(self, candidate, arena, mode, root=None):
        metrics = old(self, candidate, arena, mode, root)
        constitution = _domain_constitution(self, mode)
        if constitution is None:
            return metrics
        edit_distance = self.structural_edit_distance(root, candidate) if root is not None else 0.0
        return _rescore_with_constitution(metrics, mode, float(edit_distance), constitution)
    cls.evaluate = wrapped

for _cls in (SortingDomainPack, PathfindingDomainPack, RegexDomainPack, SimpleTaskDomainPack, AutoLiftDomainPack):
    _wrap_domain_evaluate(_cls)

_old_simple_init_o14 = SimpleTaskDomainPack.__init__
def _simple_init_o14(self, task):
    _old_simple_init_o14(self, task)
    self.constitution = getattr(task, "constitution", None)
SimpleTaskDomainPack.__init__ = _simple_init_o14

_old_autolift_domain_init_o14 = AutoLiftDomainPack.__init__
def _autolift_domain_init_o14(self, *args, **kwargs):
    _old_autolift_domain_init_o14(self, *args, **kwargs)
    if not hasattr(self, "constitution"):
        self.constitution = None
AutoLiftDomainPack.__init__ = _autolift_domain_init_o14

_old_run_task_o14 = run_task
def run_task(task: SimpleTask[T], config: Optional[EngineConfig] = None, use_counter: bool = True) -> TaskRunResult[T]:
    result = _old_run_task_o14(task, config=config, use_counter=use_counter)
    constitution = getattr(task, "constitution", None)
    if constitution is not None:
        const = _constitution_from_input(constitution, task.mode)
        result.report_markdown += "\n\n## Constitution\n"
        result.report_markdown += f"- {const.summary()}\n"
        setattr(result, "constitution", const)
    return result
globals()['run_task'] = run_task

_old_optimize_o14 = optimize
def optimize(*, initial: T, operators, score, validate=None, cases=None, proof_cases=None, hidden_cases=None, counter_cases=None, name="task", summarize=None, classify=None, pathology=None, complexity=None, serialize=None, edit_distance=None, policy_hints=None, config=None, constitution: Optional[Any] = "mode_default") -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="optimize", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    setattr(task, "constitution", _constitution_from_input(constitution, "optimize"))
    return run_task(task, config=config)
globals()['optimize'] = optimize

def repair(*, initial: T, operators, score, validate, cases=None, proof_cases=None, hidden_cases=None, counter_cases=None, name="task", summarize=None, classify=None, pathology=None, complexity=None, serialize=None, edit_distance=None, policy_hints=None, config=None, constitution: Optional[Any] = "mode_default") -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="repair", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    setattr(task, "constitution", _constitution_from_input(constitution, "repair"))
    return run_task(task, config=config)
globals()['repair'] = repair

def harden(*, initial: T, operators, score, validate, cases=None, proof_cases=None, hidden_cases=None, counter_cases=None, name="task", summarize=None, classify=None, pathology=None, complexity=None, serialize=None, edit_distance=None, policy_hints=None, config=None, constitution: Optional[Any] = "mode_default") -> TaskRunResult[T]:
    task = SimpleTask(name=name, initial=initial, operators=operators, score=score, validate=validate, cases=cases, proof_cases=proof_cases, hidden_cases=hidden_cases, counter_cases=counter_cases, mode="harden", summarize=summarize, classify=classify, pathology=pathology, complexity=complexity, serialize=serialize, edit_distance=edit_distance, policy_hints=policy_hints or {})
    setattr(task, "constitution", _constitution_from_input(constitution, "harden"))
    return run_task(task, config=config)
globals()['harden'] = harden

def improve(func: Callable[..., Any], tests: Sequence[Any], *, config: Optional[EngineConfig] = None, use_counter: bool = True, constitution: Optional[Any] = "mode_default") -> ImproveResult:
    original_source = AutoLiftAnalyzer(func).source
    domain = AutoLiftDomainPack(func, tests)
    domain.constitution = _constitution_from_input(constitution, "optimize")
    engine = ForgeEngine(domain, config or EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=137))
    results, memory, forensics, scores = engine.run()
    result = results[0]
    report = engine.build_report(results, memory, forensics, scores)
    counter_metric = None
    recovery_applied = False

    if use_counter:
        bundles = {"autolift": {"engine": engine, "results": results, "memory": memory, "forensics": forensics, "scores": scores}}
        counter = CounterForgePlus(bundles)
        counter_reports, counter_md = counter.run()
        apply_counter_feedback(bundles, counter_reports)
        attack_arena = domain.attack_arena("optimize", ["attack"], 0)
        counter_metric = domain.evaluate(result.champion, attack_arena, "optimize", domain.root)

        threshold = result.hidden.scalar * (1.15 if (domain.constitution and domain.constitution.require_attack_survival) else 1.2)
        needs_recovery = counter_metric.correct < counter_metric.total or counter_metric.scalar > threshold
        if needs_recovery:
            recovery_domain = AutoLiftDomainPack(func, tests, root_override=result.champion, extra_proof_cases=domain.attack_cases[:10], extra_hidden_cases=domain.attack_cases, extra_counter_cases=domain.attack_cases)
            recovery_domain.constitution = domain.constitution
            recovery_engine = ForgeEngine(recovery_domain, config or EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=139))
            rec_results, rec_memory, rec_forensics, rec_scores = recovery_engine.run()
            rec_result = rec_results[0]
            rec_counter = recovery_domain.evaluate(rec_result.champion, recovery_domain.attack_arena("optimize", ["attack"], 0), "optimize", recovery_domain.root)
            current_score = result.proof.scalar + result.hidden.scalar + counter_metric.scalar
            recovery_score = rec_result.proof.scalar + rec_result.hidden.scalar + rec_counter.scalar
            if rec_counter.correct == rec_counter.total and recovery_score < current_score:
                domain = recovery_domain
                engine = recovery_engine
                results, memory, forensics, scores = rec_results, rec_memory, rec_forensics, rec_scores
                result = rec_result
                counter_metric = rec_counter
                recovery_applied = True
                report = engine.build_report(results, memory, forensics, scores)
            report += "\n\n---\n\n" + counter_md
        else:
            report += "\n\n---\n\n" + counter_md

    report += "\n\n## Constitution\n"
    if domain.constitution is not None:
        report += f"- {domain.constitution.summary()}\n"
        if domain.constitution.description:
            report += f"- description: {domain.constitution.description}\n"
    report += "\n## Autolift hardening\n"
    report += f"- subset audit: {json.dumps(domain.analyzer.audit, sort_keys=True)}\n"
    report += f"- synthesized cases: user={len(domain.user_cases)}, fuzz={len(domain.fuzz_cases)}, attack={len(domain.attack_cases)}, extra_hidden={len(domain.extra_hidden_cases)}\n"
    report += f"- recovery_applied: {recovery_applied}\n"
    report += "- operator risks:\n"
    for name, meta in sorted(domain.operator_risks.items()):
        report += f"  - {name}: {meta['risk']} | {meta['rationale']}\n"

    improved_fn = _compile_function_source(result.champion.source, result.champion.func_name)
    out = ImproveResult(
        original_source=original_source,
        improved_source=result.champion.source,
        improved_function=improved_fn,
        planner=result.planner,
        proof=result.proof,
        hidden=result.hidden,
        counter=counter_metric,
        path=result.path,
        report_markdown=report,
        ir_summary=domain.root.ir_summary,
        discovered_operators=list(domain.operator_lookup().keys()),
        subset_audit=domain.analyzer.audit,
        operator_risks=domain.operator_risks,
        synthesized_case_counts={"user": len(domain.user_cases), "fuzz": len(domain.fuzz_cases), "attack": len(domain.attack_cases), "extra_hidden": len(domain.extra_hidden_cases)},
        recovery_applied=recovery_applied,
        diff_preview=_diff_preview(original_source, result.champion.source),
    )
    setattr(out, "constitution", domain.constitution)
    return out
globals()['improve'] = improve

_old_run_domain_o14 = run_domain
def run_domain(domain: DomainPack[Any], config: Optional[EngineConfig] = None, constitution: Optional[Any] = "mode_default"):
    if getattr(domain, "constitution", None) is None:
        domain.constitution = constitution
    engine = ForgeEngine(domain, config or EngineConfig())
    return engine, *engine.run()
globals()['run_domain'] = run_domain

def constitution_demo() -> str:
    lines = ["Forge Constitution Layer", ""]
    for name, const in constitution_presets().items():
        lines.append(f"- {const.summary()}")
    return "\n".join(lines)

_old_build_release_report_o14 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o14(bundles, auto)
    extra = []
    extra.append("## Constitution layer (o1.4)")
    extra.append("- Forge is now defined as an adversarial structural improvement engine governed by an explicit constitution.")
    extra.append("- Modes are no longer just labels. They compile to constitutions that define what counts as better.")
    extra.append("- Constitutions control correctness pressure, average-cost pressure, worst-case pressure, semantic-risk pressure, and adversarial expectations.")
    for name, const in constitution_presets().items():
        extra.append(f"- {const.summary()}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report

_old_package_info_o14 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o14())
    info["version"] = __version__
    info["script_name"] = "forge_o1_4.py"
    cmds = list(info.get("commands", []))
    if "constitution-demo" not in cmds:
        cmds.append("constitution-demo")
    info["commands"] = cmds
    info["definition"] = "adversarial structural improvement engine governed by an explicit constitution"
    return info
globals()['package_info'] = package_info

def demo_autolift_function_o14(tokens: List[str]) -> bool:
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    found = False
    limit = len(tokens)
    for idx in range(limit):
        tok = tokens[idx]
        if tok in forbidden:
            found = True
    return found

def autolift_demo() -> ImproveResult:
    tests = [
        ((["safe", "noop"],), False),
        ((["DROP"],), True),
        ((["safe", "ALTER"],), True),
        ((["x", "y", "z"],), False),
        ((["DELETE", "x", "y"],), True),
    ]
    return improve(demo_autolift_function_o14, tests, constitution="robust")
globals()['autolift_demo'] = autolift_demo

def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_4_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.4")
    print("constitution layer active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  constitution: {auto.constitution.summary() if hasattr(auto, 'constitution') and auto.constitution else 'none'}")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto}
globals()['run_release_demo'] = run_release_demo

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_4", description="Forge o1.4 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli

def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.4 / autolift-demo")
        print("constitution:", res.constitution.summary() if hasattr(res, "constitution") and res.constitution else "none")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
        print("FORGE o1.4 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_4_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.4 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "constitution-demo":
        print(constitution_demo())
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.5 Meaning State Foundation =====

import time as _forge_time

__version__ = "o1.5"

@dataclass(frozen=True)
class EmotionVector:
    warmth: float
    ache: float
    drift: float
    tension: float
    glow: float
    calm: float
    intimacy: float
    endlessness: float

    def clamp(self) -> "EmotionVector":
        return EmotionVector(
            warmth=max(0.0, min(1.0, self.warmth)),
            ache=max(0.0, min(1.0, self.ache)),
            drift=max(0.0, min(1.0, self.drift)),
            tension=max(0.0, min(1.0, self.tension)),
            glow=max(0.0, min(1.0, self.glow)),
            calm=max(0.0, min(1.0, self.calm)),
            intimacy=max(0.0, min(1.0, self.intimacy)),
            endlessness=max(0.0, min(1.0, self.endlessness)),
        )

    def blend(self, other: "EmotionVector", weight: float = 0.5) -> "EmotionVector":
        w = max(0.0, min(1.0, weight))
        return EmotionVector(
            warmth=(1 - w) * self.warmth + w * other.warmth,
            ache=(1 - w) * self.ache + w * other.ache,
            drift=(1 - w) * self.drift + w * other.drift,
            tension=(1 - w) * self.tension + w * other.tension,
            glow=(1 - w) * self.glow + w * other.glow,
            calm=(1 - w) * self.calm + w * other.calm,
            intimacy=(1 - w) * self.intimacy + w * other.intimacy,
            endlessness=(1 - w) * self.endlessness + w * other.endlessness,
        ).clamp()

    def to_dict(self) -> Dict[str, float]:
        return {
            "warmth": self.warmth,
            "ache": self.ache,
            "drift": self.drift,
            "tension": self.tension,
            "glow": self.glow,
            "calm": self.calm,
            "intimacy": self.intimacy,
            "endlessness": self.endlessness,
        }


@dataclass(frozen=True)
class BondState:
    trust: float
    familiarity: float
    tenderness: float
    playfulness: float
    protectiveness: float

    def clamp(self) -> "BondState":
        return BondState(
            trust=max(0.0, min(1.0, self.trust)),
            familiarity=max(0.0, min(1.0, self.familiarity)),
            tenderness=max(0.0, min(1.0, self.tenderness)),
            playfulness=max(0.0, min(1.0, self.playfulness)),
            protectiveness=max(0.0, min(1.0, self.protectiveness)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "trust": self.trust,
            "familiarity": self.familiarity,
            "tenderness": self.tenderness,
            "playfulness": self.playfulness,
            "protectiveness": self.protectiveness,
        }


@dataclass(frozen=True)
class MemoryAtom:
    id: str
    kind: str
    timestamp: float
    tags: Tuple[str, ...]
    emotion: EmotionVector
    intimacy: float
    recurrence: float
    payload_ref: str
    linked_words: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SharedSymbol:
    name: str
    linked_words: Tuple[str, ...]
    linked_memory_ids: Tuple[str, ...]
    emotion: EmotionVector
    strength: float


@dataclass(frozen=True)
class CompanionState:
    emotion: EmotionVector
    bond: BondState
    memories: Tuple[MemoryAtom, ...]
    shared_symbols: Tuple[SharedSymbol, ...]
    constitution_name: str


def companion_constitutions() -> Dict[str, ImprovementConstitution]:
    return {
        "tender": ImprovementConstitution(
            name="tender",
            correctness="strict",
            failure_penalty_per_case=1300.0,
            avg_cost_weight=70.0,
            worst_cost_weight=50.0,
            complexity_weight=0.08,
            edit_distance_weight=0.18,
            semantic_risk_penalty=0.35,
            description="Favor emotional continuity, soft expression, and grounded warmth.",
        ),
        "nocturnal": ImprovementConstitution(
            name="nocturnal",
            correctness="strict",
            failure_penalty_per_case=1280.0,
            avg_cost_weight=60.0,
            worst_cost_weight=70.0,
            complexity_weight=0.08,
            edit_distance_weight=0.18,
            semantic_risk_penalty=0.32,
            description="Favor hush, drift, dim tenderness, and reflective uncertainty.",
        ),
        "endless": ImprovementConstitution(
            name="endless",
            correctness="strict",
            failure_penalty_per_case=1250.0,
            avg_cost_weight=55.0,
            worst_cost_weight=95.0,
            complexity_weight=0.06,
            edit_distance_weight=0.16,
            semantic_risk_penalty=0.30,
            description="Favor suspended resolution, continuity, and lingering affect.",
        ),
    }


_old_constitution_presets_o15 = constitution_presets
def constitution_presets() -> Dict[str, ImprovementConstitution]:
    merged = dict(_old_constitution_presets_o15())
    merged.update(companion_constitutions())
    return merged
globals()['constitution_presets'] = constitution_presets


def _zero_emotion() -> EmotionVector:
    return EmotionVector(0.5, 0.0, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0)


def _zero_bond() -> BondState:
    return BondState(0.0, 0.0, 0.0, 0.0, 0.0)


def _emotion_average(items: Sequence[EmotionVector]) -> EmotionVector:
    if not items:
        return _zero_emotion()
    n = float(len(items))
    return EmotionVector(
        warmth=sum(x.warmth for x in items) / n,
        ache=sum(x.ache for x in items) / n,
        drift=sum(x.drift for x in items) / n,
        tension=sum(x.tension for x in items) / n,
        glow=sum(x.glow for x in items) / n,
        calm=sum(x.calm for x in items) / n,
        intimacy=sum(x.intimacy for x in items) / n,
        endlessness=sum(x.endlessness for x in items) / n,
    ).clamp()


def _simple_tokenize(text: str) -> Tuple[str, ...]:
    cleaned = re.sub(r"[^a-zA-Z0-9' ]+", " ", text.lower())
    toks = [t for t in cleaned.split() if t]
    return tuple(toks[:24])


def infer_emotion_from_text(text: str, tags: Sequence[str] = ()) -> EmotionVector:
    text_l = text.lower()
    bag = set(_simple_tokenize(text) + tuple(t.lower() for t in tags))
    warmth = 0.35
    ache = 0.0
    drift = 0.0
    tension = 0.0
    glow = 0.15
    calm = 0.35
    intimacy = 0.0
    endlessness = 0.0

    if {"warm", "soft", "tender", "gentle", "held", "safe", "care", "kind"} & bag:
        warmth += 0.35
        calm += 0.15
    if {"miss", "ache", "hurt", "longing", "lonely", "late", "night", "empty"} & bag:
        ache += 0.40
        drift += 0.20
    if {"drift", "blue", "dusk", "sea", "rain", "window", "far", "away"} & bag:
        drift += 0.35
    if {"tense", "fear", "sharp", "break", "panic", "storm"} & bag:
        tension += 0.45
        calm -= 0.15
    if {"glow", "light", "gold", "sun", "bright", "home"} & bag:
        glow += 0.40
        warmth += 0.10
    if {"calm", "slow", "hush", "quiet", "still", "rest"} & bag:
        calm += 0.35
        tension -= 0.10
    if {"you", "us", "ours", "remember", "together", "close"} & bag:
        intimacy += 0.40
        warmth += 0.15
    if {"endless", "linger", "again", "always", "still", "forever"} & bag:
        endlessness += 0.40
        drift += 0.10

    if "?" in text:
        tension += 0.05
    if any(word in text_l for word in ("maybe", "perhaps", "not sure", "i think")):
        calm += 0.08
        drift += 0.06

    return EmotionVector(warmth, ache, drift, tension, glow, calm, intimacy, endlessness).clamp()


class CompanionOrchestrator:
    def __init__(self, constitution_name: str = "tender"):
        consts = constitution_presets()
        if constitution_name not in consts:
            constitution_name = "tender"
        self.constitution_name = constitution_name
        self.state = CompanionState(
            emotion=_zero_emotion(),
            bond=_zero_bond(),
            memories=tuple(),
            shared_symbols=tuple(),
            constitution_name=constitution_name,
        )

    def _next_memory_id(self, kind: str, payload_ref: str, words: Sequence[str]) -> str:
        raw = f"{kind}|{payload_ref}|{'|'.join(words)}|{len(self.state.memories)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _update_bond(self, emotion: EmotionVector, intimacy: float) -> BondState:
        prev = self.state.bond
        next_bond = BondState(
            trust=prev.trust + 0.05 + 0.08 * intimacy,
            familiarity=prev.familiarity + 0.06,
            tenderness=prev.tenderness + 0.07 * emotion.warmth + 0.05 * intimacy,
            playfulness=prev.playfulness + 0.03 * max(0.0, emotion.glow - emotion.ache * 0.4),
            protectiveness=prev.protectiveness + 0.04 * max(emotion.ache, emotion.tension),
        )
        return next_bond.clamp()

    def _update_emotion(self, emotion: EmotionVector) -> EmotionVector:
        return self.state.emotion.blend(emotion, weight=0.42)

    def _memory_recurrence(self, linked_words: Sequence[str]) -> float:
        prior_words = [w for m in self.state.memories for w in m.linked_words]
        if not linked_words or not prior_words:
            return 0.0
        overlap = sum(1 for w in linked_words if w in prior_words)
        return min(1.0, overlap / max(1, len(linked_words)))

    def ingest_event(
        self,
        *,
        kind: str,
        payload: str,
        tags: Sequence[str] = (),
        linked_words: Optional[Sequence[str]] = None,
        payload_ref: Optional[str] = None,
        emotion: Optional[EmotionVector] = None,
        intimacy: Optional[float] = None,
    ) -> MemoryAtom:
        linked = tuple(linked_words or _simple_tokenize(payload))
        emo = emotion or infer_emotion_from_text(payload, tags)
        intimacy_value = max(0.0, min(1.0, intimacy if intimacy is not None else 0.25 + 0.45 * emo.intimacy + 0.20 * emo.warmth))
        recurrence = self._memory_recurrence(linked)
        ref = payload_ref or f"{kind}:{hashlib.sha1(payload.encode()).hexdigest()[:10]}"
        atom = MemoryAtom(
            id=self._next_memory_id(kind, ref, linked),
            kind=kind,
            timestamp=float(_forge_time.time()),
            tags=tuple(tags),
            emotion=emo,
            intimacy=intimacy_value,
            recurrence=recurrence,
            payload_ref=ref,
            linked_words=linked,
        )
        new_memories = tuple(list(self.state.memories) + [atom])
        self.state = CompanionState(
            emotion=self._update_emotion(emo),
            bond=self._update_bond(emo, intimacy_value),
            memories=new_memories,
            shared_symbols=self.state.shared_symbols,
            constitution_name=self.constitution_name,
        )
        self.update_symbols()
        return atom

    def ingest_text(self, text: str, *, tags: Sequence[str] = (), linked_words: Optional[Sequence[str]] = None) -> MemoryAtom:
        return self.ingest_event(kind="text", payload=text, tags=tags, linked_words=linked_words)

    def ingest_song(self, title: str, *, tags: Sequence[str] = (), linked_words: Optional[Sequence[str]] = None) -> MemoryAtom:
        all_tags = tuple(tags) + ("song",)
        return self.ingest_event(kind="song", payload=title, tags=all_tags, linked_words=linked_words or _simple_tokenize(title))

    def update_symbols(self) -> Tuple[SharedSymbol, ...]:
        word_to_memories: Dict[str, List[MemoryAtom]] = defaultdict(list)
        for memory in self.state.memories:
            for word in memory.linked_words:
                if len(word) >= 3:
                    word_to_memories[word].append(memory)

        symbols: List[SharedSymbol] = []
        for word, memories in word_to_memories.items():
            if len(memories) < 2:
                continue
            emotions = _emotion_average([m.emotion for m in memories])
            strength = min(1.0, 0.22 * len(memories) + 0.30 * max(m.recurrence for m in memories) + 0.25 * emotions.intimacy)
            linked_words = tuple(sorted({w for m in memories for w in m.linked_words if w == word or w in memory.tags}))
            symbols.append(
                SharedSymbol(
                    name=word,
                    linked_words=tuple(sorted({word} | set(linked_words))),
                    linked_memory_ids=tuple(m.id for m in memories[-6:]),
                    emotion=emotions,
                    strength=strength,
                )
            )

        symbols.sort(key=lambda s: (s.strength, len(s.linked_memory_ids), s.name), reverse=True)
        self.state = CompanionState(
            emotion=self.state.emotion,
            bond=self.state.bond,
            memories=self.state.memories,
            shared_symbols=tuple(symbols[:24]),
            constitution_name=self.constitution_name,
        )
        return self.state.shared_symbols

    def anchor(self) -> Dict[str, Any]:
        if self.state.shared_symbols:
            sym = max(self.state.shared_symbols, key=lambda s: (s.strength, len(s.linked_memory_ids)))
            return {
                "kind": "symbol",
                "name": sym.name,
                "strength": sym.strength,
                "linked_memory_ids": list(sym.linked_memory_ids),
                "emotion": sym.emotion.to_dict(),
            }
        if self.state.memories:
            mem = max(self.state.memories, key=lambda m: (m.intimacy + m.recurrence, m.timestamp))
            return {
                "kind": "memory",
                "id": mem.id,
                "memory_kind": mem.kind,
                "payload_ref": mem.payload_ref,
                "linked_words": list(mem.linked_words),
                "emotion": mem.emotion.to_dict(),
            }
        return {"kind": "none"}

    def snapshot_for_ios(self) -> Dict[str, Any]:
        return {
            "constitution_name": self.state.constitution_name,
            "emotion": self.state.emotion.to_dict(),
            "bond": self.state.bond.to_dict(),
            "anchor": self.anchor(),
            "memories": [
                {
                    "id": m.id,
                    "kind": m.kind,
                    "timestamp": m.timestamp,
                    "tags": list(m.tags),
                    "intimacy": m.intimacy,
                    "recurrence": m.recurrence,
                    "payload_ref": m.payload_ref,
                    "linked_words": list(m.linked_words),
                    "emotion": m.emotion.to_dict(),
                }
                for m in self.state.memories[-12:]
            ],
            "shared_symbols": [
                {
                    "name": s.name,
                    "linked_words": list(s.linked_words),
                    "linked_memory_ids": list(s.linked_memory_ids),
                    "strength": s.strength,
                    "emotion": s.emotion.to_dict(),
                }
                for s in self.state.shared_symbols[:12]
            ],
        }


def companion_demo() -> Dict[str, Any]:
    companion = CompanionOrchestrator("tender")
    companion.ingest_text("blue again at the window, soft and late, I remember us", tags=("late", "blue", "memory"))
    companion.ingest_song("Blue Window Demo", tags=("blue", "night", "song"))
    companion.ingest_text("still blue, still warm, still close", tags=("blue", "warm", "recurrence"))
    return companion.snapshot_for_ios()


_old_build_release_report_o15 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o15(bundles, auto)
    snap = companion_demo()
    extra = []
    extra.append("## Meaning state foundation (o1.5)")
    extra.append("- Added EmotionVector, BondState, MemoryAtom, SharedSymbol, and CompanionState.")
    extra.append("- Added a first CompanionOrchestrator with ingest, memory accumulation, symbol clustering, and iOS snapshot export.")
    extra.append("- Prime rule added: outputs should be able to anchor in one shared memory or symbol instead of answering from nowhere.")
    extra.append(f"- Demo anchor: {json.dumps(snap.get('anchor', {}), sort_keys=True)}")
    extra.append(f"- Demo shared symbols: {', '.join(sym['name'] for sym in snap.get('shared_symbols', [])[:8]) or 'none'}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report


_old_package_info_o15 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o15())
    info["version"] = __version__
    info["script_name"] = "forge_o1_5.py"
    cmds = list(info.get("commands", []))
    if "companion-demo" not in cmds:
        cmds.append("companion-demo")
    info["commands"] = cmds
    info["meaning_state"] = "foundation"
    return info
globals()['package_info'] = package_info


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    companion_snapshot = companion_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        build_release_report(bundles, auto),
        "## Companion snapshot demo\n- " + json.dumps(companion_snapshot, sort_keys=True),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_5_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.5")
    print("meaning state foundation active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  constitution: {auto.constitution.summary() if hasattr(auto, 'constitution') and auto.constitution else 'none'}")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    print("\n[companion]")
    print("  anchor:", json.dumps(companion_snapshot["anchor"], sort_keys=True))
    print("  symbols:", ", ".join(sym["name"] for sym in companion_snapshot["shared_symbols"][:8]) or "none")
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto, "companion_snapshot": companion_snapshot}
globals()['run_release_demo'] = run_release_demo


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_5", description="Forge o1.5 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("companion-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.5 / autolift-demo")
        print("constitution:", res.constitution.summary() if hasattr(res, "constitution") and res.constitution else "none")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
        print("FORGE o1.5 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_5_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.5 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "constitution-demo":
        print(constitution_demo())
        print()
        for name, const in companion_constitutions().items():
            print(f"- {const.summary()}")
        return 0
    if cmd == "companion-demo":
        print(json.dumps(companion_demo(), indent=2, sort_keys=True))
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.6 Melody Motif Domain =====

__version__ = "o1.6"

@dataclass(frozen=True)
class MelodyCandidate:
    scale: str
    tonic: int
    contour: Tuple[int, ...]
    rhythm: Tuple[float, ...]
    note_count: int
    register: str
    sustain: float
    repetition: float
    cadence_strength: float
    rest_ratio: float
    atmosphere: Tuple[str, ...]

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.scale,
            self.tonic,
            self.contour,
            self.rhythm,
            self.note_count,
            self.register,
            round(self.sustain, 3),
            round(self.repetition, 3),
            round(self.cadence_strength, 3),
            round(self.rest_ratio, 3),
            self.atmosphere,
        )

    def complexity(self) -> int:
        return (
            1
            + len(set(self.contour))
            + len(set(self.rhythm))
            + len(self.atmosphere)
            + (1 if self.register != "mid" else 0)
            + (1 if self.scale != "minor" else 0)
        )


@dataclass(frozen=True)
class SongDescriptor:
    energy: float
    brightness: float
    density: float
    rhythmic_stability: float
    melancholy: float
    warmth: float
    tension: float
    endlessness: float

    def clamp(self) -> "SongDescriptor":
        return SongDescriptor(
            energy=max(0.0, min(1.0, self.energy)),
            brightness=max(0.0, min(1.0, self.brightness)),
            density=max(0.0, min(1.0, self.density)),
            rhythmic_stability=max(0.0, min(1.0, self.rhythmic_stability)),
            melancholy=max(0.0, min(1.0, self.melancholy)),
            warmth=max(0.0, min(1.0, self.warmth)),
            tension=max(0.0, min(1.0, self.tension)),
            endlessness=max(0.0, min(1.0, self.endlessness)),
        )


def _descriptor_average(items: Sequence[SongDescriptor]) -> SongDescriptor:
    if not items:
        return SongDescriptor(0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.2, 0.5)
    n = float(len(items))
    return SongDescriptor(
        energy=sum(x.energy for x in items) / n,
        brightness=sum(x.brightness for x in items) / n,
        density=sum(x.density for x in items) / n,
        rhythmic_stability=sum(x.rhythmic_stability for x in items) / n,
        melancholy=sum(x.melancholy for x in items) / n,
        warmth=sum(x.warmth for x in items) / n,
        tension=sum(x.tension for x in items) / n,
        endlessness=sum(x.endlessness for x in items) / n,
    ).clamp()


def infer_song_descriptor(text: str, tags: Sequence[str] = ()) -> SongDescriptor:
    emo = infer_emotion_from_text(text, tags)
    bag = set(_simple_tokenize(text) + tuple(t.lower() for t in tags))
    energy = 0.35 + 0.25 * emo.glow + 0.20 * emo.tension - 0.10 * emo.calm
    brightness = 0.30 + 0.35 * emo.glow + 0.18 * emo.warmth - 0.18 * emo.ache
    density = 0.28 + 0.25 * emo.tension + 0.18 * emo.drift
    rhythmic_stability = 0.60 + 0.12 * emo.calm - 0.10 * emo.drift
    melancholy = 0.15 + 0.65 * emo.ache + 0.20 * emo.drift
    warmth = 0.18 + 0.70 * emo.warmth + 0.10 * emo.intimacy
    tension = 0.10 + 0.70 * emo.tension + (0.10 if {"storm", "sharp", "panic", "break"} & bag else 0.0)
    endlessness = 0.08 + 0.78 * emo.endlessness + 0.16 * emo.drift + 0.08 * emo.calm
    if {"dance", "neon", "rush", "fast"} & bag:
        energy += 0.18
        density += 0.12
    if {"hush", "slow", "late", "blue", "window", "night"} & bag:
        endlessness += 0.14
        warmth += 0.05
        brightness -= 0.05
    return SongDescriptor(energy, brightness, density, rhythmic_stability, melancholy, warmth, tension, endlessness).clamp()


def melody_descriptor_from_texts(items: Sequence[Tuple[str, Sequence[str]]]) -> SongDescriptor:
    return _descriptor_average([infer_song_descriptor(text, tags) for text, tags in items])


def melody_candidate_descriptor(c: MelodyCandidate) -> SongDescriptor:
    avg_abs_interval = sum(abs(x) for x in c.contour) / max(1, len(c.contour))
    density = max(0.0, min(1.0, (1.0 - c.rest_ratio) * (0.35 + 0.08 * len(c.rhythm))))
    cadence_pull = c.cadence_strength
    endlessness = max(0.0, min(1.0, 0.55 * c.sustain + 0.35 * c.repetition + 0.25 * (1.0 - cadence_pull)))
    warmth = 0.30 + (0.25 if c.scale in {"minor", "dorian"} else 0.10) + (0.12 if "warm" in c.atmosphere else 0.0) + 0.18 * c.sustain
    melancholy = 0.20 + (0.35 if c.scale in {"minor", "dorian"} else 0.08) + (0.12 if "dusk" in c.atmosphere or "blue" in c.atmosphere else 0.0)
    brightness = 0.25 + (0.22 if c.scale in {"major", "mixolydian"} else 0.0) + (0.12 if "glow" in c.atmosphere or "glass" in c.atmosphere else 0.0)
    energy = 0.20 + min(0.45, avg_abs_interval * 0.08) + (0.10 if c.register == "high" else 0.0) + (0.08 if c.rest_ratio < 0.15 else 0.0)
    rhythmic_stability = max(0.0, min(1.0, 0.70 - 0.20 * abs(len(set(c.rhythm)) - 2) / max(1, len(c.rhythm)) + 0.15 * c.repetition))
    tension = max(0.0, min(1.0, 0.20 + 0.35 * cadence_pull + 0.15 * avg_abs_interval / 4.0 - 0.10 * c.sustain))
    return SongDescriptor(energy, brightness, density, rhythmic_stability, melancholy, warmth, tension, endlessness).clamp()


class MelodyDomainPack(DomainPack[MelodyCandidate]):
    name = "melody"

    def __init__(self, descriptor: SongDescriptor, anchor_symbol: Optional[str] = None):
        self.target = descriptor
        self.anchor_symbol = anchor_symbol or ""
        self.constitution = "endless" if descriptor.endlessness >= 0.62 else "tender"
        self._operators = self._build_operators()

    def root_sets(self) -> Dict[str, Dict[str, MelodyCandidate]]:
        base = MelodyCandidate(
            scale="minor",
            tonic=0,
            contour=(1, -1, 0, -2),
            rhythm=(1.0, 0.5, 1.0, 1.5),
            note_count=5,
            register="mid",
            sustain=0.45,
            repetition=0.30,
            cadence_strength=0.55,
            rest_ratio=0.12,
            atmosphere=("blue", "warm") if self.target.warmth >= 0.55 else ("glass",),
        )
        return {"optimize": {"motif_seed": base}}

    def candidate_hash(self, candidate: MelodyCandidate) -> str:
        return hashlib.sha1(repr(candidate.signature()).encode()).hexdigest()[:12]

    def summarize(self, candidate: MelodyCandidate) -> str:
        return (
            f"scale={candidate.scale}, tonic={candidate.tonic}, contour={candidate.contour}, rhythm={candidate.rhythm}, "
            f"register={candidate.register}, sustain={candidate.sustain:.2f}, repetition={candidate.repetition:.2f}, "
            f"cadence={candidate.cadence_strength:.2f}, rest={candidate.rest_ratio:.2f}, atmosphere={candidate.atmosphere}"
        )

    def complexity(self, candidate: MelodyCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: MelodyCandidate) -> str:
        if candidate.cadence_strength <= 0.30 and candidate.sustain >= 0.60:
            return "endless_motif"
        if candidate.scale in {"minor", "dorian"}:
            return "soft_minor"
        if candidate.scale in {"major", "mixolydian"}:
            return "glow_major"
        return "neutral_motif"

    def detect_pathology(self, candidate: MelodyCandidate, mode: str) -> str:
        if candidate.cadence_strength > 0.72 and self.target.endlessness > 0.55:
            return "too_resolved"
        if candidate.rest_ratio < 0.05 and self.target.warmth > 0.55:
            return "too_busy"
        if len(set(candidate.contour)) <= 1:
            return "too_flat"
        if max(abs(x) for x in candidate.contour) >= 6:
            return "too_jittery"
        return "generic"

    def structural_edit_distance(self, root: Optional[MelodyCandidate], candidate: MelodyCandidate) -> int:
        if root is None:
            return candidate.complexity()
        return sum(x != y for x, y in zip(root.signature(), candidate.signature()))

    def _clip(self, c: MelodyCandidate) -> MelodyCandidate:
        contour = tuple(max(-7, min(7, x)) for x in c.contour)
        rhythm = tuple(max(0.25, min(2.0, x)) for x in c.rhythm)
        note_count = max(3, min(8, c.note_count))
        sustain = max(0.0, min(1.0, c.sustain))
        repetition = max(0.0, min(1.0, c.repetition))
        cadence_strength = max(0.0, min(1.0, c.cadence_strength))
        rest_ratio = max(0.0, min(0.8, c.rest_ratio))
        return MelodyCandidate(
            scale=c.scale,
            tonic=c.tonic % 12,
            contour=contour,
            rhythm=rhythm,
            note_count=note_count,
            register=c.register,
            sustain=sustain,
            repetition=repetition,
            cadence_strength=cadence_strength,
            rest_ratio=rest_ratio,
            atmosphere=tuple(dict.fromkeys(c.atmosphere))[:4],
        )

    def _with_atmo(self, c: MelodyCandidate, token: str) -> MelodyCandidate:
        return self._clip(replace(c, atmosphere=tuple(dict.fromkeys(c.atmosphere + (token,)))))

    def _build_operators(self) -> Dict[str, DomainOperator[MelodyCandidate]]:
        return {
            "increase_sustain": DomainOperator("increase_sustain", lambda c: self._clip(replace(c, sustain=c.sustain + 0.15))),
            "reduce_cadence": DomainOperator("reduce_cadence", lambda c: self._clip(replace(c, cadence_strength=c.cadence_strength - 0.18))),
            "increase_repetition": DomainOperator("increase_repetition", lambda c: self._clip(replace(c, repetition=c.repetition + 0.16))),
            "add_rest_space": DomainOperator("add_rest_space", lambda c: self._clip(replace(c, rest_ratio=c.rest_ratio + 0.10))),
            "descend_tail": DomainOperator("descend_tail", lambda c: self._clip(replace(c, contour=c.contour[:-1] + (min(-1, c.contour[-1] - 1),)))),
            "ascend_lift": DomainOperator("ascend_lift", lambda c: self._clip(replace(c, contour=(max(1, c.contour[0] + 1),) + c.contour[1:]))),
            "minor_color": DomainOperator("minor_color", lambda c: self._clip(replace(c, scale="minor"))),
            "major_glow": DomainOperator("major_glow", lambda c: self._clip(replace(c, scale="major"))),
            "narrow_range": DomainOperator("narrow_range", lambda c: self._clip(replace(c, contour=tuple(int(round(x * 0.5)) for x in c.contour)))),
            "widen_range": DomainOperator("widen_range", lambda c: self._clip(replace(c, contour=tuple(int(round(x * 1.5)) if x != 0 else 0 for x in c.contour)))),
            "lower_register": DomainOperator("lower_register", lambda c: self._clip(replace(c, register="low"))),
            "raise_register": DomainOperator("raise_register", lambda c: self._clip(replace(c, register="high"))),
            "warm_atmosphere": DomainOperator("warm_atmosphere", lambda c: self._with_atmo(c, "warm")),
            "endless_atmosphere": DomainOperator("endless_atmosphere", lambda c: self._with_atmo(c, "endless")),
            "blue_atmosphere": DomainOperator("blue_atmosphere", lambda c: self._with_atmo(c, "blue")),
            "glass_atmosphere": DomainOperator("glass_atmosphere", lambda c: self._with_atmo(c, "glass")),
        }

    def operator_lookup(self) -> Dict[str, DomainOperator[MelodyCandidate]]:
        return self._operators

    def operator_bank(self, mode: str, root: MelodyCandidate, promoted: Sequence[DomainOperator[MelodyCandidate]]) -> List[DomainOperator[MelodyCandidate]]:
        out: List[DomainOperator[MelodyCandidate]] = []
        seen = set()
        for op in list(promoted) + list(self._operators.values()):
            if op.name in seen:
                continue
            seen.add(op.name)
            out.append(op)
        return out

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[SongDescriptor], Arena[SongDescriptor], Arena[SongDescriptor]]:
        rng = random.Random(seed)
        def vary(base: SongDescriptor, count: int, name: str) -> Arena[SongDescriptor]:
            cases = []
            for i in range(count):
                wiggle = lambda x, s: max(0.0, min(1.0, x + rng.uniform(-s, s)))
                d = SongDescriptor(
                    wiggle(base.energy, 0.08),
                    wiggle(base.brightness, 0.08),
                    wiggle(base.density, 0.08),
                    wiggle(base.rhythmic_stability, 0.08),
                    wiggle(base.melancholy, 0.08),
                    wiggle(base.warmth, 0.08),
                    wiggle(base.tension, 0.08),
                    wiggle(base.endlessness, 0.10),
                )
                cases.append((f"{name}_{i}", d))
            return Arena(name, cases)
        return vary(self.target, 12, f"{mode}_planner"), vary(self.target, 28, f"{mode}_proof"), vary(self.target, 32, f"{mode}_hidden")

    def _coherence_penalty(self, c: MelodyCandidate) -> float:
        jumps = [abs(x) for x in c.contour]
        jump_pen = 0.08 * sum(max(0, j - 4) for j in jumps)
        busy_pen = 0.40 if c.rest_ratio < 0.05 and c.sustain < 0.35 else 0.0
        cadence_pen = 0.55 * max(0.0, c.cadence_strength - (0.35 if self.target.endlessness > 0.55 else 0.75))
        flat_pen = 0.28 if len(set(c.contour)) <= 1 else 0.0
        generic_pen = 0.20 if c.contour in {(1, -1, 1, -1), (0, 0, 0, 0)} else 0.0
        return jump_pen + busy_pen + cadence_pen + flat_pen + generic_pen

    def evaluate(self, candidate: MelodyCandidate, arena: Arena[SongDescriptor], mode: str, root: Optional[MelodyCandidate] = None) -> EvalMetrics:
        desc = melody_candidate_descriptor(candidate)
        costs: List[float] = []
        for _, target in arena.cases:
            dist = (
                1.30 * abs(desc.energy - target.energy)
                + 1.05 * abs(desc.brightness - target.brightness)
                + 1.15 * abs(desc.density - target.density)
                + 0.85 * abs(desc.rhythmic_stability - target.rhythmic_stability)
                + 1.35 * abs(desc.melancholy - target.melancholy)
                + 1.40 * abs(desc.warmth - target.warmth)
                + 1.20 * abs(desc.tension - target.tension)
                + 1.60 * abs(desc.endlessness - target.endlessness)
            )
            costs.append(dist + self._coherence_penalty(candidate))
        avg_cost = statistics.mean(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = candidate.complexity()
        edit_distance = self.structural_edit_distance(root, candidate) if root else 0.0
        scalar = avg_cost * 100.0 + worst_cost * (90.0 if self.target.endlessness > 0.55 else 55.0) + complexity * 0.15 + edit_distance * 0.20
        return EvalMetrics(len(arena.cases), len(arena.cases), avg_cost, worst_cost, complexity, scalar, {})

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        hints = {
            "too_resolved": {"reduce_cadence": 2.5, "increase_sustain": 1.8, "endless_atmosphere": 1.6},
            "too_busy": {"add_rest_space": 2.4, "increase_sustain": 1.4, "narrow_range": 1.3},
            "too_flat": {"ascend_lift": 1.2, "descend_tail": 1.2, "widen_range": 1.0},
            "too_jittery": {"narrow_range": 2.0, "increase_repetition": 1.1, "lower_register": 0.8},
        }
        if self.target.endlessness > 0.60:
            hints.setdefault("generic", {}).update({"reduce_cadence": 1.4, "increase_sustain": 1.2, "add_rest_space": 0.9})
        if self.target.warmth > 0.58:
            hints.setdefault("generic", {}).update({"warm_atmosphere": 1.1, "blue_atmosphere": 0.8})
        return hints.get(pathology, hints.get("generic", {}))

    def rediscovery_verdict(self, root: MelodyCandidate, champion: MelodyCandidate) -> str:
        if self.classify_family(root) == self.classify_family(champion):
            return "refined same motif family"
        return "family shift toward target affect"

    def suite_attribution(self, root: MelodyCandidate, champion: MelodyCandidate, arena: Arena[SongDescriptor], mode: str) -> List[Tuple[str, float, float]]:
        root_desc = melody_candidate_descriptor(root)
        champ_desc = melody_candidate_descriptor(champion)
        rows = []
        for label, target in arena.cases:
            root_cost = (
                abs(root_desc.energy - target.energy)
                + abs(root_desc.brightness - target.brightness)
                + abs(root_desc.density - target.density)
                + abs(root_desc.rhythmic_stability - target.rhythmic_stability)
                + abs(root_desc.melancholy - target.melancholy)
                + abs(root_desc.warmth - target.warmth)
                + abs(root_desc.tension - target.tension)
                + abs(root_desc.endlessness - target.endlessness)
                + self._coherence_penalty(root)
            )
            champ_cost = (
                abs(champ_desc.energy - target.energy)
                + abs(champ_desc.brightness - target.brightness)
                + abs(champ_desc.density - target.density)
                + abs(champ_desc.rhythmic_stability - target.rhythmic_stability)
                + abs(champ_desc.melancholy - target.melancholy)
                + abs(champ_desc.warmth - target.warmth)
                + abs(champ_desc.tension - target.tension)
                + abs(champ_desc.endlessness - target.endlessness)
                + self._coherence_penalty(champion)
            )
            rows.append((label, root_cost - champ_cost, 0.0))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows


def melody_demo() -> Dict[str, Any]:
    companion = CompanionOrchestrator("endless")
    companion.ingest_song("Blue Window Demo", tags=("blue", "late", "warm"))
    companion.ingest_song("Glass Dusk Loop", tags=("glass", "dusk", "endless"))
    companion.ingest_text("still blue at the window, soft and unended", tags=("blue", "endless", "memory"))
    memory_inputs = [(m.payload_ref.replace("song:", "").replace("text:", ""), m.tags) for m in companion.state.memories[-4:]]
    target = melody_descriptor_from_texts([(m.payload_ref + " " + " ".join(m.tags), m.tags) for m in companion.state.memories[-4:]])
    domain = MelodyDomainPack(target, anchor_symbol=companion.anchor().get("name"))
    engine, results, memory, forensics, scores = run_domain(domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=503), constitution="endless")
    result = results[0]
    snapshot = companion.snapshot_for_ios()
    snapshot["melody"] = {
        "target_descriptor": target.__dict__,
        "anchor_symbol": domain.anchor_symbol,
        "champion_summary": domain.summarize(result.champion),
        "path": list(result.path),
        "proof": _metrics_to_dict(result.proof),
        "hidden": _metrics_to_dict(result.hidden),
    }
    return {
        "snapshot": snapshot,
        "report": engine.build_report(results, memory, forensics, scores),
        "scores": scores,
    }


_old_build_release_report_o16 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o16(bundles, auto)
    mel = melody_demo()
    extra = []
    extra.append("## Melody motif domain (o1.6)")
    extra.append("- Added MelodyCandidate, SongDescriptor, and MelodyDomainPack.")
    extra.append("- Melody search now targets warmth, melancholy, tension, and endlessness without attempting full-song generation.")
    extra.append("- Scoring rewards emotional similarity, motif coherence, and suspended cadence when the target asks for endlessness.")
    extra.append("- Prime anchoring rule applied: melody demo inherits the strongest shared symbol from the companion memory state.")
    extra.append(f"- Melody anchor symbol: {json.dumps(mel['snapshot']['melody']['anchor_symbol'])}")
    extra.append(f"- Melody champion: {mel['snapshot']['melody']['champion_summary']}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report


_old_package_info_o16 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o16())
    info["version"] = __version__
    info["script_name"] = "forge_o1_6.py"
    cmds = list(info.get("commands", []))
    if "melody-demo" not in cmds:
        cmds.append("melody-demo")
    info["commands"] = cmds
    info["melody_domain"] = "motif"
    return info
globals()['package_info'] = package_info


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    companion_snapshot = companion_demo()
    mel = melody_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        mel["report"],
        build_release_report(bundles, auto),
        "## Companion snapshot demo\n- " + json.dumps(companion_snapshot, sort_keys=True),
        "## Melody snapshot demo\n- " + json.dumps(mel["snapshot"], sort_keys=True),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_6_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.6")
    print("melody motif domain active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  constitution: {auto.constitution.summary() if hasattr(auto, 'constitution') and auto.constitution else 'none'}")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    print("\n[companion]")
    print("  anchor:", json.dumps(companion_snapshot["anchor"], sort_keys=True))
    print("  symbols:", ", ".join(sym["name"] for sym in companion_snapshot["shared_symbols"][:8]) or "none")
    print("\n[melody]")
    print("  anchor_symbol:", json.dumps(mel["snapshot"]["melody"]["anchor_symbol"]))
    print("  champion:", mel["snapshot"]["melody"]["champion_summary"])
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto, "companion_snapshot": companion_snapshot, "melody_snapshot": mel["snapshot"]}
globals()['run_release_demo'] = run_release_demo


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_6", description="Forge o1.6 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("companion-demo")
    sub.add_parser("melody-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.6 / autolift-demo")
        print("constitution:", res.constitution.summary() if hasattr(res, "constitution") and res.constitution else "none")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
        print("FORGE o1.6 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_6_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.6 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "constitution-demo":
        print(constitution_demo())
        print()
        for name, const in companion_constitutions().items():
            print(f"- {const.summary()}")
        return 0
    if cmd == "companion-demo":
        print(json.dumps(companion_demo(), indent=2, sort_keys=True))
        return 0
    if cmd == "melody-demo":
        print(json.dumps(melody_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.7 Grounded Language Domain =====

__version__ = "o1.7"

@dataclass(frozen=True)
class LexiconEntry:
    token: str
    base_tags: Tuple[str, ...]
    personal_tags: Tuple[str, ...]
    linked_memory_ids: Tuple[str, ...]
    confidence: float
    warmth_bias: float
    ambiguity: float


@dataclass(frozen=True)
class ResponseCandidate:
    intent: str
    stance: str
    anchor_tokens: Tuple[str, ...]
    recalled_memory_ids: Tuple[str, ...]
    sentence_shape: str
    softness: float
    certainty: float
    metaphor: float
    warmth: float
    text: str

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.intent,
            self.stance,
            self.anchor_tokens,
            self.recalled_memory_ids,
            self.sentence_shape,
            round(self.softness, 3),
            round(self.certainty, 3),
            round(self.metaphor, 3),
            round(self.warmth, 3),
            self.text,
        )

    def complexity(self) -> int:
        return (
            1
            + len(self.anchor_tokens)
            + len(self.recalled_memory_ids)
            + len(set(_simple_tokenize(self.text)))
            + (1 if self.metaphor > 0.45 else 0)
            + (1 if self.certainty < 0.45 else 0)
        )


@dataclass(frozen=True)
class LanguagePromptCase:
    prompt: str
    intent: str
    desired_emotion: EmotionVector
    desired_warmth: float
    desired_softness: float
    desired_certainty: float
    require_anchor: bool
    preferred_tokens: Tuple[str, ...]


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_lexicon_from_companion(companion: CompanionOrchestrator) -> Dict[str, LexiconEntry]:
    memory_map: Dict[str, List[MemoryAtom]] = defaultdict(list)
    for memory in companion.state.memories:
        for token in memory.linked_words:
            memory_map[token].append(memory)

    symbol_names = {s.name for s in companion.state.shared_symbols}
    lexicon: Dict[str, LexiconEntry] = {}
    for token, memories in memory_map.items():
        emo = _emotion_average([m.emotion for m in memories])
        base_tags = tuple(sorted({tag for m in memories for tag in m.tags if len(tag) >= 3}))[:6]
        personal_tags = []
        if token in symbol_names:
            personal_tags.append("shared_symbol")
        if emo.warmth > 0.58:
            personal_tags.append("warm")
        if emo.ache > 0.30:
            personal_tags.append("aching")
        if emo.endlessness > 0.28:
            personal_tags.append("lingering")
        confidence = min(1.0, 0.30 + 0.16 * len(memories))
        warmth_bias = max(0.0, min(1.0, 0.55 * emo.warmth + 0.25 * emo.intimacy + (0.15 if token in symbol_names else 0.0)))
        ambiguity = max(0.0, min(1.0, 0.20 + 0.18 * len({tuple(m.tags) for m in memories}) + 0.15 * emo.drift))
        lexicon[token] = LexiconEntry(
            token=token,
            base_tags=base_tags,
            personal_tags=tuple(personal_tags),
            linked_memory_ids=tuple(m.id for m in memories[-8:]),
            confidence=confidence,
            warmth_bias=warmth_bias,
            ambiguity=ambiguity,
        )
    return lexicon


def _language_prompt_descriptor(prompt: str, intent: str) -> LanguagePromptCase:
    emo = infer_emotion_from_text(prompt, (intent,))
    desired_warmth = max(0.35, emo.warmth)
    desired_softness = max(0.30, 0.25 + 0.55 * emo.calm + 0.18 * emo.drift)
    desired_certainty = max(0.18, min(0.82, 0.55 - 0.25 * emo.drift - 0.20 * emo.ache + 0.15 * emo.glow))
    preferred = tuple(t for t in _simple_tokenize(prompt) if len(t) >= 4)[:6]
    return LanguagePromptCase(
        prompt=prompt,
        intent=intent,
        desired_emotion=emo,
        desired_warmth=desired_warmth,
        desired_softness=desired_softness,
        desired_certainty=desired_certainty,
        require_anchor=True,
        preferred_tokens=preferred,
    )


def _base_response_text(
    prompt_case: LanguagePromptCase,
    companion: CompanionOrchestrator,
    lexicon: Dict[str, LexiconEntry],
) -> ResponseCandidate:
    anchor = companion.anchor()
    anchor_tokens: Tuple[str, ...] = ()
    memory_ids: Tuple[str, ...] = ()
    if anchor["kind"] == "symbol":
        anchor_tokens = (anchor["name"],)
        memory_ids = tuple(anchor["linked_memory_ids"][:3])
        text = f"I keep coming back to {anchor['name']}. It still feels close."
    elif anchor["kind"] == "memory":
        anchor_tokens = tuple(anchor.get("linked_words", [])[:2])
        memory_ids = (anchor["id"],)
        text = "This feels tied to something I remember, softly."
    else:
        prompt_tokens = prompt_case.preferred_tokens[:2]
        anchor_tokens = tuple(prompt_tokens)
        text = "I want to answer gently and stay close to what you meant."

    stance = {
        "comfort": "gentle",
        "reflect": "uncertain",
        "remember": "tender",
        "wonder": "open",
    }.get(prompt_case.intent, "soft")
    return ResponseCandidate(
        intent=prompt_case.intent,
        stance=stance,
        anchor_tokens=anchor_tokens,
        recalled_memory_ids=memory_ids,
        sentence_shape="balanced",
        softness=0.52,
        certainty=0.58,
        metaphor=0.18,
        warmth=0.55,
        text=text,
    )


def _render_response(
    candidate: ResponseCandidate,
    prompt_case: LanguagePromptCase,
    companion: CompanionOrchestrator,
    lexicon: Dict[str, LexiconEntry],
) -> str:
    anchor_word = candidate.anchor_tokens[0] if candidate.anchor_tokens else None
    memory_phrase = ""
    if candidate.recalled_memory_ids:
        memory_phrase = " I am anchoring to something we have already touched."
    if candidate.sentence_shape == "short":
        text = candidate.text
    elif candidate.sentence_shape == "question":
        text = candidate.text + " Do you want me to stay with that feeling a little longer?"
    elif candidate.sentence_shape == "image":
        image_word = anchor_word or (prompt_case.preferred_tokens[0] if prompt_case.preferred_tokens else "this")
        text = candidate.text + f" It sits like {image_word} in the room, not fully leaving."
    else:
        text = candidate.text + memory_phrase

    if candidate.softness > 0.68 and not text.lower().startswith("maybe"):
        text = "Maybe " + text[0].lower() + text[1:]
    if candidate.certainty < 0.38:
        if "I don't know" not in text and "I'm not sure" not in text:
            text += " I'm not fully sure, but that is the shape I get."
    if candidate.metaphor > 0.55:
        metaphor_word = anchor_word or "it"
        text += f" {metaphor_word.capitalize()} feels more like weather than an answer."
    if candidate.warmth > 0.70 and "close" not in text.lower():
        text += " I want to keep it close and gentle."
    text = _truncate_words(text.strip(), 32)
    if not text.endswith((".", "?", "!")):
        text += "."
    return text


class LanguageDomainPack(DomainPack[ResponseCandidate]):
    name = "language"

    def __init__(self, companion: CompanionOrchestrator, prompt_case: LanguagePromptCase):
        self.companion = companion
        self.prompt_case = prompt_case
        self.lexicon = build_lexicon_from_companion(companion)
        self.constitution = "tender" if prompt_case.desired_warmth >= 0.55 else "balanced"
        self._operators = self._build_operators()

    def root_sets(self) -> Dict[str, Dict[str, ResponseCandidate]]:
        root = _base_response_text(self.prompt_case, self.companion, self.lexicon)
        return {"optimize": {"response_seed": replace(root, text=_render_response(root, self.prompt_case, self.companion, self.lexicon))}}

    def candidate_hash(self, candidate: ResponseCandidate) -> str:
        return hashlib.sha1(repr(candidate.signature()).encode()).hexdigest()[:12]

    def summarize(self, candidate: ResponseCandidate) -> str:
        return (
            f"intent={candidate.intent}, stance={candidate.stance}, anchor_tokens={candidate.anchor_tokens}, "
            f"softness={candidate.softness:.2f}, certainty={candidate.certainty:.2f}, metaphor={candidate.metaphor:.2f}, "
            f"warmth={candidate.warmth:.2f}, text={candidate.text}"
        )

    def complexity(self, candidate: ResponseCandidate) -> int:
        return candidate.complexity()

    def classify_family(self, candidate: ResponseCandidate) -> str:
        if candidate.sentence_shape == "image" and candidate.metaphor >= 0.45:
            return "image_lingering"
        if candidate.certainty < 0.40:
            return "uncertain_tender"
        if candidate.sentence_shape == "short":
            return "brief_warm"
        return "grounded_reply"

    def detect_pathology(self, candidate: ResponseCandidate, mode: str) -> str:
        tokens = _simple_tokenize(candidate.text)
        if self.prompt_case.require_anchor and not candidate.anchor_tokens:
            return "ungrounded"
        if candidate.warmth < 0.30:
            return "too_cold"
        if len(tokens) > 30:
            return "overexplained"
        if candidate.certainty > 0.82 and ("maybe" not in candidate.text.lower() and "not sure" not in candidate.text.lower()):
            return "too_confident"
        if candidate.metaphor > 0.72 and not candidate.anchor_tokens:
            return "too_floaty"
        return "generic"

    def structural_edit_distance(self, root: Optional[ResponseCandidate], candidate: ResponseCandidate) -> int:
        if root is None:
            return candidate.complexity()
        return sum(x != y for x, y in zip(root.signature(), candidate.signature()))

    def _render(self, c: ResponseCandidate) -> ResponseCandidate:
        return replace(c, text=_render_response(c, self.prompt_case, self.companion, self.lexicon))

    def _clip(self, c: ResponseCandidate) -> ResponseCandidate:
        return replace(
            c,
            softness=max(0.0, min(1.0, c.softness)),
            certainty=max(0.0, min(1.0, c.certainty)),
            metaphor=max(0.0, min(1.0, c.metaphor)),
            warmth=max(0.0, min(1.0, c.warmth)),
        )

    def _pick_memory(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        anchor = self.companion.anchor()
        if anchor["kind"] == "symbol":
            return (anchor["name"],), tuple(anchor["linked_memory_ids"][:3])
        if anchor["kind"] == "memory":
            linked = tuple(anchor.get("linked_words", [])[:2])
            return linked, (anchor["id"],)
        return tuple(), tuple()

    def _build_operators(self) -> Dict[str, DomainOperator[ResponseCandidate]]:
        def op(name: str, fn):
            return DomainOperator(name, lambda c: self._render(self._clip(fn(c))))
        return {
            "soften_claim": op("soften_claim", lambda c: replace(c, softness=c.softness + 0.16, certainty=c.certainty - 0.12, stance="gentle")),
            "admit_unknown": op("admit_unknown", lambda c: replace(c, certainty=c.certainty - 0.20, softness=c.softness + 0.08)),
            "ask_gently": op("ask_gently", lambda c: replace(c, sentence_shape="question", softness=c.softness + 0.12)),
            "bind_to_memory": op("bind_to_memory", lambda c: replace(c, anchor_tokens=self._pick_memory()[0], recalled_memory_ids=self._pick_memory()[1], warmth=c.warmth + 0.08)),
            "recall_shared_symbol": op("recall_shared_symbol", lambda c: replace(c, anchor_tokens=self._pick_memory()[0] or c.anchor_tokens, warmth=c.warmth + 0.10, metaphor=c.metaphor + 0.05)),
            "reduce_literalness": op("reduce_literalness", lambda c: replace(c, metaphor=c.metaphor + 0.16, certainty=c.certainty - 0.06, sentence_shape="image")),
            "increase_tenderness": op("increase_tenderness", lambda c: replace(c, warmth=c.warmth + 0.16, softness=c.softness + 0.10)),
            "shorten_reply": op("shorten_reply", lambda c: replace(c, sentence_shape="short", certainty=c.certainty - 0.04)),
            "linger_on_image": op("linger_on_image", lambda c: replace(c, sentence_shape="image", metaphor=c.metaphor + 0.20, softness=c.softness + 0.04)),
            "echo_user_phrase": op("echo_user_phrase", lambda c: replace(c, anchor_tokens=self.prompt_case.preferred_tokens[:2] or c.anchor_tokens, warmth=c.warmth + 0.05)),
        }

    def operator_lookup(self) -> Dict[str, DomainOperator[ResponseCandidate]]:
        return self._operators

    def operator_bank(self, mode: str, root: ResponseCandidate, promoted: Sequence[DomainOperator[ResponseCandidate]]) -> List[DomainOperator[ResponseCandidate]]:
        out: List[DomainOperator[ResponseCandidate]] = []
        seen = set()
        for op in list(promoted) + list(self._operators.values()):
            if op.name in seen:
                continue
            seen.add(op.name)
            out.append(op)
        return out

    def mode_arenas(self, mode: str, seed: int) -> Tuple[Arena[LanguagePromptCase], Arena[LanguagePromptCase], Arena[LanguagePromptCase]]:
        base = self.prompt_case
        planner = Arena(f"{mode}_planner", [(f"{mode}_p0", base)])
        proof = Arena(f"{mode}_proof", [(f"{mode}_pf{i}", base) for i in range(6)])
        hidden_cases = [base]
        hidden_cases.append(LanguagePromptCase(base.prompt + " maybe", base.intent, base.desired_emotion, min(1.0, base.desired_warmth + 0.05), min(1.0, base.desired_softness + 0.10), max(0.18, base.desired_certainty - 0.10), True, base.preferred_tokens))
        hidden_cases.append(LanguagePromptCase(base.prompt + " softly", base.intent, base.desired_emotion, base.desired_warmth, base.desired_softness, base.desired_certainty, True, base.preferred_tokens))
        hidden = Arena(f"{mode}_hidden", [(f"{mode}_h{i}", case) for i, case in enumerate(hidden_cases)])
        return planner, proof, hidden

    def evaluate(self, candidate: ResponseCandidate, arena: Arena[LanguagePromptCase], mode: str, root: Optional[ResponseCandidate] = None) -> EvalMetrics:
        failures: Dict[str, int] = defaultdict(int)
        costs: List[float] = []
        correct = 0
        for _, prompt_case in arena.cases:
            text = candidate.text.lower()
            anchor_ok = (not prompt_case.require_anchor) or bool(candidate.anchor_tokens) or bool(candidate.recalled_memory_ids)
            if not anchor_ok:
                failures["ungrounded"] += 1
                continue
            generic_penalty = 0.0
            if "i understand" in text or "i am here for you" in text:
                generic_penalty += 0.45
            if len(set(_simple_tokenize(candidate.text))) <= 4:
                generic_penalty += 0.22
            unsupported_penalty = 0.0
            if candidate.certainty > 0.78 and ("not sure" not in text and "maybe" not in text):
                unsupported_penalty += 0.22
            token_match = 0.0
            if prompt_case.preferred_tokens:
                token_match = sum(1 for t in prompt_case.preferred_tokens if t in text) / max(1, len(prompt_case.preferred_tokens))
            cost = (
                1.35 * abs(candidate.warmth - prompt_case.desired_warmth)
                + 1.15 * abs(candidate.softness - prompt_case.desired_softness)
                + 1.05 * abs(candidate.certainty - prompt_case.desired_certainty)
                + 0.65 * max(0.0, 0.25 - token_match)
                + generic_penalty
                + unsupported_penalty
            )
            costs.append(cost)
            correct += 1
        avg_cost = sum(costs) / len(costs) if costs else float("inf")
        worst_cost = max(costs) if costs else float("inf")
        complexity = candidate.complexity()
        edit_distance = self.structural_edit_distance(root, candidate) if root else 0.0
        scalar = avg_cost * 100.0 + worst_cost * 65.0 + complexity * 0.08 + edit_distance * 0.18
        return EvalMetrics(correct, len(arena.cases), avg_cost, worst_cost, complexity, scalar, dict(failures))

    def policy_hints(self, mode: str, pathology: str) -> Dict[str, float]:
        hints = {
            "ungrounded": {"bind_to_memory": 2.6, "recall_shared_symbol": 2.2, "echo_user_phrase": 1.0},
            "too_cold": {"increase_tenderness": 2.3, "bind_to_memory": 1.2},
            "overexplained": {"shorten_reply": 2.4, "soften_claim": 0.8},
            "too_confident": {"admit_unknown": 2.5, "soften_claim": 1.8},
            "too_floaty": {"bind_to_memory": 2.1, "echo_user_phrase": 1.1},
        }
        if self.prompt_case.desired_warmth > 0.55:
            hints.setdefault("generic", {}).update({"increase_tenderness": 1.2, "bind_to_memory": 1.0})
        return hints.get(pathology, hints.get("generic", {}))

    def rediscovery_verdict(self, root: ResponseCandidate, champion: ResponseCandidate) -> str:
        if self.classify_family(root) == self.classify_family(champion):
            return "refined same reply family"
        return "shifted toward warmer grounded language"

    def suite_attribution(self, root: ResponseCandidate, champion: ResponseCandidate, arena: Arena[LanguagePromptCase], mode: str) -> List[Tuple[str, float, float]]:
        rows = []
        for label, case in arena.cases:
            root_cost = abs(root.warmth - case.desired_warmth) + abs(root.softness - case.desired_softness) + abs(root.certainty - case.desired_certainty)
            champ_cost = abs(champion.warmth - case.desired_warmth) + abs(champion.softness - case.desired_softness) + abs(champion.certainty - case.desired_certainty)
            rows.append((label, root_cost - champ_cost, 0.0))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows


def language_demo() -> Dict[str, Any]:
    companion = CompanionOrchestrator("tender")
    companion.ingest_text("blue again at the window, soft and late, I remember us", tags=("late", "blue", "memory"))
    companion.ingest_song("Blue Window Demo", tags=("blue", "night", "song"))
    companion.ingest_text("still blue, still warm, still close", tags=("blue", "warm", "recurrence"))
    prompt_case = _language_prompt_descriptor("I feel strange tonight and I don't really know why", "reflect")
    domain = LanguageDomainPack(companion, prompt_case)
    engine, results, memory, forensics, scores = run_domain(domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=607), constitution="tender")
    result = results[0]
    snapshot = companion.snapshot_for_ios()
    snapshot["response"] = {
        "prompt": prompt_case.prompt,
        "champion_summary": domain.summarize(result.champion),
        "text": result.champion.text,
        "path": list(result.path),
        "proof": _metrics_to_dict(result.proof),
        "hidden": _metrics_to_dict(result.hidden),
    }
    return {
        "snapshot": snapshot,
        "report": engine.build_report(results, memory, forensics, scores),
        "scores": scores,
    }


_old_build_release_report_o17 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o17(bundles, auto)
    lang = language_demo()
    extra = []
    extra.append("## Grounded language domain (o1.7)")
    extra.append("- Added LexiconEntry, ResponseCandidate, LanguagePromptCase, and LanguageDomainPack.")
    extra.append("- Responses are now scored for grounding, tenderness, uncertainty honesty, and anti-generic pressure.")
    extra.append("- Language evolution uses memory anchors and shared symbols instead of answering from nowhere.")
    extra.append(f"- Language reply: {lang['snapshot']['response']['text']}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report


_old_package_info_o17 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o17())
    info["version"] = __version__
    info["script_name"] = "forge_o1_7.py"
    cmds = list(info.get("commands", []))
    if "language-demo" not in cmds:
        cmds.append("language-demo")
    info["commands"] = cmds
    info["language_domain"] = "grounded"
    return info
globals()['package_info'] = package_info


def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    companion_snapshot = companion_demo()
    mel = melody_demo()
    lang = language_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        mel["report"],
        lang["report"],
        build_release_report(bundles, auto),
        "## Companion snapshot demo\n- " + json.dumps(companion_snapshot, sort_keys=True),
        "## Melody snapshot demo\n- " + json.dumps(mel["snapshot"], sort_keys=True),
        "## Language snapshot demo\n- " + json.dumps(lang["snapshot"], sort_keys=True),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path('/mnt/data/forge_o1_7_report.md').write_text(combined, encoding='utf-8')
    print("FORGE o1.7")
    print("grounded language domain active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[autolift restricted]")
    print(f"  constitution: {auto.constitution.summary() if hasattr(auto, 'constitution') and auto.constitution else 'none'}")
    print(f"  path: {' -> '.join(auto.path) if auto.path else '(root kept)'}")
    print(f"  proof {auto.proof.correct}/{auto.proof.total} hidden {auto.hidden.correct}/{auto.hidden.total}")
    print("\n[companion]")
    print("  anchor:", json.dumps(companion_snapshot["anchor"], sort_keys=True))
    print("  symbols:", ", ".join(sym["name"] for sym in companion_snapshot["shared_symbols"][:8]) or "none")
    print("\n[melody]")
    print("  anchor_symbol:", json.dumps(mel["snapshot"]["melody"]["anchor_symbol"]))
    print("  champion:", mel["snapshot"]["melody"]["champion_summary"])
    print("\n[language]")
    print("  reply:", lang["snapshot"]["response"]["text"])
    return {"bundles": bundles, "counter": counter_reports, "autolift": auto, "companion_snapshot": companion_snapshot, "melody_snapshot": mel["snapshot"], "language_snapshot": lang["snapshot"]}
globals()['run_release_demo'] = run_release_demo


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_7", description="Forge o1.7 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("companion-demo")
    sub.add_parser("melody-demo")
    sub.add_parser("language-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.7 / autolift-demo")
        print("constitution:", res.constitution.summary() if hasattr(res, "constitution") and res.constitution else "none")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
        print("FORGE o1.7 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path('/mnt/data/forge_o1_7_regex_report.md').write_text(engine.build_report(results, memory, forensics, scores), encoding='utf-8')
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.7 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "constitution-demo":
        print(constitution_demo())
        print()
        for name, const in companion_constitutions().items():
            print(f"- {const.summary()}")
        return 0
    if cmd == "companion-demo":
        print(json.dumps(companion_demo(), indent=2, sort_keys=True))
        return 0
    if cmd == "melody-demo":
        print(json.dumps(melody_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "language-demo":
        print(json.dumps(language_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.8 Warmth Unification + Autocorrect =====

__version__ = "o1.8"

@dataclass(frozen=True)
class WarmthProfile:
    memory_recall_bias: float
    uncertainty_honesty: float
    tenderness_bias: float
    symbolic_reuse: float
    restraint: float
    playfulness: float

    def clamp(self) -> "WarmthProfile":
        return WarmthProfile(
            memory_recall_bias=max(0.0, min(1.0, self.memory_recall_bias)),
            uncertainty_honesty=max(0.0, min(1.0, self.uncertainty_honesty)),
            tenderness_bias=max(0.0, min(1.0, self.tenderness_bias)),
            symbolic_reuse=max(0.0, min(1.0, self.symbolic_reuse)),
            restraint=max(0.0, min(1.0, self.restraint)),
            playfulness=max(0.0, min(1.0, self.playfulness)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "memory_recall_bias": self.memory_recall_bias,
            "uncertainty_honesty": self.uncertainty_honesty,
            "tenderness_bias": self.tenderness_bias,
            "symbolic_reuse": self.symbolic_reuse,
            "restraint": self.restraint,
            "playfulness": self.playfulness,
        }


def warmth_profile_presets() -> Dict[str, WarmthProfile]:
    return {
        "tender": WarmthProfile(0.84, 0.78, 0.90, 0.74, 0.72, 0.28),
        "nocturnal": WarmthProfile(0.76, 0.82, 0.74, 0.66, 0.88, 0.14),
        "endless": WarmthProfile(0.80, 0.76, 0.72, 0.86, 0.78, 0.18),
        "balanced": WarmthProfile(0.52, 0.62, 0.50, 0.42, 0.58, 0.30),
        "robust": WarmthProfile(0.44, 0.70, 0.40, 0.30, 0.68, 0.20),
        "optimize": WarmthProfile(0.36, 0.56, 0.32, 0.22, 0.54, 0.24),
        "repair": WarmthProfile(0.44, 0.64, 0.42, 0.34, 0.60, 0.18),
        "harden": WarmthProfile(0.34, 0.72, 0.30, 0.20, 0.70, 0.14),
    }

def warmth_profile_for(name: str) -> WarmthProfile:
    return warmth_profile_presets().get(name, warmth_profile_presets()["balanced"]).clamp()

def _levenshtein(a: str, b: str, limit: int = 2) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            row_min = min(row_min, val)
        if row_min > limit:
            return limit + 1
        prev = cur
    return prev[-1]

def _normalize_token_alias(token: str, known: Sequence[str]) -> str:
    token = token.lower().strip()
    if len(token) < 3:
        return token
    exact = set(known)
    if token in exact:
        return token
    best = token
    best_dist = 3
    for cand in exact:
        if abs(len(cand) - len(token)) > 1:
            continue
        d = _levenshtein(token, cand, limit=2)
        if d < best_dist:
            best_dist = d
            best = cand
            if d == 1:
                break
    return best if best_dist <= 1 else token

_old_companion_init_o18 = CompanionOrchestrator.__init__
def _companion_init_o18(self, constitution_name: str = "tender"):
    _old_companion_init_o18(self, constitution_name)
    self.warmth_profile = warmth_profile_for(self.constitution_name)
    self.autocorrect_aliases: Dict[str, str] = {}
CompanionOrchestrator.__init__ = _companion_init_o18

def _known_tokens_o18(self) -> Tuple[str, ...]:
    known = []
    for memory in self.state.memories:
        known.extend(memory.linked_words)
        known.extend(tag.lower() for tag in memory.tags if isinstance(tag, str))
    for sym in self.state.shared_symbols:
        known.append(sym.name)
        known.extend(sym.linked_words)
    return tuple(sorted(set(x for x in known if len(x) >= 3)))
CompanionOrchestrator._known_tokens = _known_tokens_o18

def _normalize_linked_words_o18(self, words: Sequence[str]) -> Tuple[str, ...]:
    known = self._known_tokens()
    out: List[str] = []
    for raw in words:
        tok = raw.lower().strip()
        if not tok:
            continue
        canon = _normalize_token_alias(tok, known)
        if canon != tok:
            self.autocorrect_aliases[tok] = canon
        out.append(canon)
    return tuple(out)
CompanionOrchestrator._normalize_linked_words = _normalize_linked_words_o18

_old_ingest_event_o18 = CompanionOrchestrator.ingest_event
def _ingest_event_o18(
    self,
    *,
    kind: str,
    payload: str,
    tags: Sequence[str] = (),
    linked_words: Optional[Sequence[str]] = None,
    payload_ref: Optional[str] = None,
    emotion: Optional[EmotionVector] = None,
    intimacy: Optional[float] = None,
) -> MemoryAtom:
    raw_words = tuple(linked_words or _simple_tokenize(payload))
    norm_words = self._normalize_linked_words(raw_words)
    norm_tags = tuple(_normalize_token_alias(str(t).lower(), self._known_tokens()) for t in tags)
    return _old_ingest_event_o18(
        self,
        kind=kind,
        payload=payload,
        tags=norm_tags,
        linked_words=norm_words,
        payload_ref=payload_ref,
        emotion=emotion,
        intimacy=intimacy,
    )
CompanionOrchestrator.ingest_event = _ingest_event_o18

_old_snapshot_o18 = CompanionOrchestrator.snapshot_for_ios
def _snapshot_for_ios_o18(self) -> Dict[str, Any]:
    snap = _old_snapshot_o18(self)
    snap["warmth_profile"] = self.warmth_profile.to_dict()
    snap["autocorrect_aliases"] = dict(sorted(self.autocorrect_aliases.items()))
    return snap
CompanionOrchestrator.snapshot_for_ios = _snapshot_for_ios_o18

def build_lexicon_from_companion(companion: CompanionOrchestrator) -> Dict[str, LexiconEntry]:
    memory_map: Dict[str, List[MemoryAtom]] = defaultdict(list)
    symbol_names = {s.name for s in companion.state.shared_symbols}
    for memory in companion.state.memories:
        for token in memory.linked_words:
            canon = _normalize_token_alias(token, companion._known_tokens())
            memory_map[canon].append(memory)

    lexicon: Dict[str, LexiconEntry] = {}
    for token, memories in memory_map.items():
        emo = _emotion_average([m.emotion for m in memories])
        base_tags = tuple(sorted({tag for m in memories for tag in m.tags if len(tag) >= 3}))[:6]
        personal_tags = []
        if token in symbol_names:
            personal_tags.append("shared_symbol")
        if emo.warmth > 0.58:
            personal_tags.append("warm")
        if emo.ache > 0.30:
            personal_tags.append("aching")
        if emo.endlessness > 0.28:
            personal_tags.append("lingering")
        confidence = min(1.0, 0.30 + 0.16 * len(memories))
        warmth_bias = max(0.0, min(1.0, 0.55 * emo.warmth + 0.25 * emo.intimacy + (0.15 if token in symbol_names else 0.0)))
        ambiguity = max(0.0, min(1.0, 0.20 + 0.18 * len({tuple(m.tags) for m in memories}) + 0.15 * emo.drift))
        lexicon[token] = LexiconEntry(
            token=token,
            base_tags=base_tags,
            personal_tags=tuple(personal_tags),
            linked_memory_ids=tuple(m.id for m in memories[-8:]),
            confidence=confidence,
            warmth_bias=warmth_bias,
            ambiguity=ambiguity,
        )
    return lexicon
globals()['build_lexicon_from_companion'] = build_lexicon_from_companion

_old_melody_init_o18 = MelodyDomainPack.__init__
def _melody_init_o18(self, descriptor: SongDescriptor, anchor_symbol: Optional[str] = None, warmth_profile: Optional[WarmthProfile] = None):
    _old_melody_init_o18(self, descriptor, anchor_symbol=anchor_symbol)
    self.warmth_profile = warmth_profile or warmth_profile_for(getattr(self, "constitution", "balanced"))
MelodyDomainPack.__init__ = _melody_init_o18

_old_melody_eval_o18 = MelodyDomainPack.evaluate
def _melody_evaluate_o18(self, candidate: MelodyCandidate, arena: Arena[SongDescriptor], mode: str, root: Optional[MelodyCandidate] = None) -> EvalMetrics:
    metrics = _old_melody_eval_o18(self, candidate, arena, mode, root)
    wp = getattr(self, "warmth_profile", warmth_profile_for("balanced"))
    warm_delta = abs(candidate.sustain - (0.45 + 0.35 * wp.memory_recall_bias))
    restraint_delta = abs(candidate.rest_ratio - (0.08 + 0.18 * wp.restraint))
    symbolic_penalty = 0.0
    if getattr(self, "anchor_symbol", ""):
        symbolic_penalty = 0.25 * (0.0 if self.anchor_symbol in candidate.atmosphere else 1.0)
    scalar = metrics.scalar + 18.0 * warm_delta + 16.0 * restraint_delta + 22.0 * symbolic_penalty
    return EvalMetrics(metrics.correct, metrics.total, metrics.avg_cost, metrics.worst_cost, metrics.complexity, scalar, dict(metrics.failures))
MelodyDomainPack.evaluate = _melody_evaluate_o18

_old_language_init_o18 = LanguageDomainPack.__init__
def _language_init_o18(self, companion: CompanionOrchestrator, prompt_case: LanguagePromptCase):
    _old_language_init_o18(self, companion, prompt_case)
    self.warmth_profile = companion.warmth_profile
LanguageDomainPack.__init__ = _language_init_o18

_old_language_eval_o18 = LanguageDomainPack.evaluate
def _language_evaluate_o18(self, candidate: ResponseCandidate, arena: Arena[LanguagePromptCase], mode: str, root: Optional[ResponseCandidate] = None) -> EvalMetrics:
    metrics = _old_language_eval_o18(self, candidate, arena, mode, root)
    wp = getattr(self, "warmth_profile", warmth_profile_for("balanced"))
    text = candidate.text.lower()
    uncertainty_pen = 0.0
    if wp.uncertainty_honesty > 0.65 and candidate.certainty < 0.45:
        if "not sure" not in text and "maybe" not in text:
            uncertainty_pen += 0.22
    restraint_pen = 0.0
    if wp.restraint > 0.65 and len(_simple_tokenize(candidate.text)) > 26:
        restraint_pen += 0.20
    symbol_pen = 0.0
    anchor = self.companion.anchor()
    if anchor["kind"] == "symbol" and wp.symbolic_reuse > 0.60:
        if not candidate.anchor_tokens or anchor["name"] not in candidate.anchor_tokens:
            symbol_pen += 0.28
    tenderness_pen = abs(candidate.warmth - max(0.35, wp.tenderness_bias))
    scalar = metrics.scalar + 30.0 * tenderness_pen + 60.0 * uncertainty_pen + 40.0 * restraint_pen + 55.0 * symbol_pen
    return EvalMetrics(metrics.correct, metrics.total, metrics.avg_cost, metrics.worst_cost, metrics.complexity, scalar, dict(metrics.failures))
LanguageDomainPack.evaluate = _language_evaluate_o18

def warmth_unification_demo() -> Dict[str, Any]:
    companion = CompanionOrchestrator("tender")
    companion.ingest_text("blu again at the windo, soft and late, I remember us", tags=("late", "blu", "memory"))
    companion.ingest_song("Blue Window Demo", tags=("blue", "night", "song"))
    companion.ingest_text("still blue, still warm, still close", tags=("blue", "warm", "recurrence"))
    prompt_case = _language_prompt_descriptor("I feel strange tonight and I dont really know why", "reflect")
    language_domain = LanguageDomainPack(companion, prompt_case)
    l_engine, l_results, l_memory, l_forensics, l_scores = run_domain(language_domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=709), constitution="tender")
    language_result = l_results[0]

    descriptor = melody_descriptor_from_texts([(m.payload_ref + " " + " ".join(m.tags), m.tags) for m in companion.state.memories[-4:]])
    melody_domain = MelodyDomainPack(descriptor, anchor_symbol=companion.anchor().get("name"), warmth_profile=companion.warmth_profile)
    m_engine, m_results, m_memory, m_forensics, m_scores = run_domain(melody_domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=711), constitution="endless")
    melody_result = m_results[0]

    snapshot = companion.snapshot_for_ios()
    snapshot["response"] = {
        "text": language_result.champion.text,
        "path": list(language_result.path),
        "proof": _metrics_to_dict(language_result.proof),
        "hidden": _metrics_to_dict(language_result.hidden),
    }
    snapshot["melody"] = {
        "champion_summary": melody_domain.summarize(melody_result.champion),
        "path": list(melody_result.path),
        "proof": _metrics_to_dict(melody_result.proof),
        "hidden": _metrics_to_dict(melody_result.hidden),
    }
    return {
        "snapshot": snapshot,
        "language_report": l_engine.build_report(l_results, l_memory, l_forensics, l_scores),
        "melody_report": m_engine.build_report(m_results, m_memory, m_forensics, m_scores),
    }

_old_build_release_report_o18 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o18(bundles, auto)
    warm = warmth_unification_demo()
    extra = []
    extra.append("## Warmth unification + autocorrect (o1.8)")
    extra.append("- Added WarmthProfile so melody and language can be biased by the same companion temperament.")
    extra.append("- Added token autocorrect / canonicalization to avoid duplicate meanings from small misspellings.")
    extra.append("- Shared symbols, lexicon entries, melody mood, and language anchoring now lean on the same warm-state profile.")
    extra.append(f"- Demo autocorrect aliases: {json.dumps(warm['snapshot'].get('autocorrect_aliases', {}), sort_keys=True)}")
    extra.append(f"- Demo response: {warm['snapshot']['response']['text']}")
    extra.append(f"- Demo melody: {warm['snapshot']['melody']['champion_summary']}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report

_old_package_info_o18 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o18())
    info["version"] = __version__
    info["script_name"] = "forge_o1_8.py"
    cmds = list(info.get("commands", []))
    if "warmth-demo" not in cmds:
        cmds.append("warmth-demo")
    info["commands"] = cmds
    info["warmth_layer"] = "unified"
    info["autocorrect"] = "canonical_token_aliasing"
    return info
globals()['package_info'] = package_info

def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    companion_snapshot = companion_demo()
    mel = melody_demo()
    lang = language_demo()
    warm = warmth_unification_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        mel["report"],
        lang["report"],
        warm["melody_report"],
        warm["language_report"],
        build_release_report(bundles, auto),
        "## Companion snapshot demo\n- " + json.dumps(companion_snapshot, sort_keys=True),
        "## Melody snapshot demo\n- " + json.dumps(mel["snapshot"], sort_keys=True),
        "## Language snapshot demo\n- " + json.dumps(lang["snapshot"], sort_keys=True),
        "## Warmth snapshot demo\n- " + json.dumps(warm["snapshot"], sort_keys=True),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_8_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.8")
    print("warmth unification and autocorrect active")
    for name in ("sorting", "pathfinding", "regex"):
        print(f"\n[{name}]")
        for r in sorted(bundles[name]["results"], key=lambda x: (x.mode, x.root_name)):
            print(f"  [{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
    print("\n[language]")
    print("  reply:", warm["snapshot"]["response"]["text"])
    print("\n[melody]")
    print("  champion:", warm["snapshot"]["melody"]["champion_summary"])
    print("\n[autocorrect]")
    print("  aliases:", json.dumps(warm["snapshot"].get("autocorrect_aliases", {}), sort_keys=True))
    return {
        "bundles": bundles,
        "counter": counter_reports,
        "autolift": auto,
        "companion_snapshot": companion_snapshot,
        "melody_snapshot": mel["snapshot"],
        "language_snapshot": lang["snapshot"],
        "warmth_snapshot": warm["snapshot"],
    }
globals()['run_release_demo'] = run_release_demo

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_8", description="Forge o1.8 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("companion-demo")
    sub.add_parser("melody-demo")
    sub.add_parser("language-demo")
    sub.add_parser("warmth-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli

def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    cmd = args.command or "demo"
    if cmd == "demo":
        run_release_demo(write_report=True)
        return 0
    if cmd == "autolift-demo":
        res = autolift_demo()
        print("FORGE o1.8 / autolift-demo")
        print("constitution:", res.constitution.summary() if hasattr(res, "constitution") and res.constitution else "none")
        print("path:", " -> ".join(res.path) if res.path else "(root kept)")
        print("proof:", f"{res.proof.correct}/{res.proof.total}", "hidden:", f"{res.hidden.correct}/{res.hidden.total}")
        print(res.improved_source)
        return 0
    if cmd == "regex-domain-demo":
        engine, results, memory, forensics, scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
        print("FORGE o1.8 / regex-domain-demo")
        for r in sorted(results, key=lambda x: (x.mode, x.root_name)):
            print(f"[{r.mode}] {r.root_name}: proof {r.proof.correct}/{r.proof.total} hidden {r.hidden.correct}/{r.hidden.total} avg {r.proof.avg_cost:.4f}")
        Path("/mnt/data/forge_o1_8_regex_report.md").write_text(engine.build_report(results, memory, forensics, scores), encoding="utf-8")
        return 0
    if cmd == "deep-ir-demo":
        demo = deep_ir_demo()
        print("FORGE o1.8 / deep-ir-demo")
        print("ir_summary:", demo.ir_summary)
        print("opportunities:", ", ".join(demo.opportunities) if demo.opportunities else "none")
        print("quantifiers:", "; ".join(demo.loops) if demo.loops else "none")
        print("guards:", "; ".join(demo.guards[:4]) if demo.guards else "none")
        print("reductions:", "; ".join(demo.reductions) if demo.reductions else "none")
        return 0
    if cmd == "constitution-demo":
        print(constitution_demo())
        print()
        for name, const in companion_constitutions().items():
            print(f"- {const.summary()}")
        print()
        for name, wp in warmth_profile_presets().items():
            print(f"- {name}: {json.dumps(wp.to_dict(), sort_keys=True)}")
        return 0
    if cmd == "companion-demo":
        print(json.dumps(companion_demo(), indent=2, sort_keys=True))
        return 0
    if cmd == "melody-demo":
        print(json.dumps(melody_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "language-demo":
        print(json.dumps(language_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "warmth-demo":
        print(json.dumps(warmth_unification_demo()["snapshot"], indent=2, sort_keys=True))
        return 0
    if cmd == "list-domains":
        print(json.dumps(package_info(), indent=2))
        return 0
    parser.print_help()
    return 1
globals()['run_cli'] = run_cli



# ===== Forge o1.8.1 Canonical Spelling Preference =====

__version__ = "o1.8.1"

def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _token_quality(token: str, frequency: int = 0) -> float:
    vowels = set("aeiouy")
    vowel_count = sum(1 for ch in token if ch in vowels)
    repeat_penalty = 0.0
    run = 1
    for i in range(1, len(token)):
        if token[i] == token[i - 1]:
            run += 1
            if run >= 2:
                repeat_penalty += 0.25
        else:
            run = 1
    alpha_bonus = 1.6 if token.isalpha() else 0.6
    vowel_bonus = 0.8 if vowel_count >= 1 else 0.0
    length_bonus = min(len(token), 10) * 0.24
    freq_bonus = min(frequency, 6) * 0.18
    truncation_penalty = 0.0
    if len(token) >= 3 and token[-1] not in vowels and vowel_count == 1:
        truncation_penalty += 0.18
    return alpha_bonus + vowel_bonus + length_bonus + freq_bonus - repeat_penalty - truncation_penalty

def _tokens_should_merge(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 3 or len(b) < 3:
        return False
    if a[0] != b[0]:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    if _levenshtein(a, b, limit=1) > 1:
        return False
    prefix = _common_prefix_len(a, b)
    return prefix >= max(2, min(len(a), len(b)) - 1)

def _choose_cluster_canonical(tokens: Sequence[str], counts: Dict[str, int]) -> str:
    ranked = sorted(
        set(tokens),
        key=lambda t: (_token_quality(t, counts.get(t, 0)), counts.get(t, 0), len(t), t),
        reverse=True,
    )
    return ranked[0] if ranked else ""

def _cluster_tokens(tokens: Sequence[str], counts: Dict[str, int]) -> Dict[str, str]:
    unique = sorted(set(t for t in tokens if t))
    parent = {t: t for t in unique}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            if _tokens_should_merge(a, b):
                union(a, b)

    groups: Dict[str, List[str]] = {}
    for tok in unique:
        groups.setdefault(find(tok), []).append(tok)

    mapping: Dict[str, str] = {}
    for members in groups.values():
        canon = _choose_cluster_canonical(members, counts)
        for m in members:
            mapping[m] = canon
    return mapping

_old_companion_init_o181 = CompanionOrchestrator.__init__
def _companion_init_o181(self, constitution_name: str = "tender"):
    _old_companion_init_o181(self, constitution_name)
    self.token_frequency: Dict[str, int] = {}
CompanionOrchestrator.__init__ = _companion_init_o181

def _collect_state_tokens_o181(self) -> Tuple[str, ...]:
    toks = []
    for memory in self.state.memories:
        toks.extend(memory.linked_words)
        toks.extend(tag.lower() for tag in memory.tags if isinstance(tag, str))
    for sym in self.state.shared_symbols:
        toks.append(sym.name)
        toks.extend(sym.linked_words)
    toks.extend(self.autocorrect_aliases.keys())
    toks.extend(self.autocorrect_aliases.values())
    return tuple(sorted(set(t for t in toks if len(t) >= 3)))
CompanionOrchestrator._collect_state_tokens = _collect_state_tokens_o181

def _canonical_map_o181(self, extra_tokens: Sequence[str] = ()) -> Dict[str, str]:
    tokens = list(self._collect_state_tokens()) + [t for t in extra_tokens if len(t) >= 3]
    counts = dict(getattr(self, "token_frequency", {}))
    return _cluster_tokens(tokens, counts)
CompanionOrchestrator._canonical_map = _canonical_map_o181

def _normalize_linked_words_o181(self, words: Sequence[str]) -> Tuple[str, ...]:
    raw = [w.lower().strip() for w in words if w and w.strip()]
    canon_map = self._canonical_map(raw)
    out = []
    for tok in raw:
        canon = canon_map.get(tok, tok)
        if canon != tok:
            self.autocorrect_aliases[tok] = canon
        out.append(canon)
    return tuple(out)
CompanionOrchestrator._normalize_linked_words = _normalize_linked_words_o181

def _recanonicalize_state_o181(self) -> None:
    canon_map = self._canonical_map()
    if not canon_map:
        return
    new_memories = []
    for memory in self.state.memories:
        linked_words = tuple(canon_map.get(w, w) for w in memory.linked_words)
        tags = tuple(canon_map.get(str(t).lower(), str(t).lower()) for t in memory.tags)
        new_memories.append(
            MemoryAtom(
                id=memory.id,
                kind=memory.kind,
                timestamp=memory.timestamp,
                tags=tags,
                emotion=memory.emotion,
                intimacy=memory.intimacy,
                recurrence=memory.recurrence,
                payload_ref=memory.payload_ref,
                linked_words=linked_words,
            )
        )
    alias_map = {k: v for k, v in canon_map.items() if k != v}
    self.autocorrect_aliases = dict(sorted(alias_map.items()))
    self.state = CompanionState(
        emotion=self.state.emotion,
        bond=self.state.bond,
        memories=tuple(new_memories),
        shared_symbols=self.state.shared_symbols,
        constitution_name=self.state.constitution_name,
    )
    self.update_symbols()
CompanionOrchestrator._recanonicalize_state = _recanonicalize_state_o181

_old_ingest_event_o181 = CompanionOrchestrator.ingest_event
def _ingest_event_o181(
    self,
    *,
    kind: str,
    payload: str,
    tags: Sequence[str] = (),
    linked_words: Optional[Sequence[str]] = None,
    payload_ref: Optional[str] = None,
    emotion: Optional[EmotionVector] = None,
    intimacy: Optional[float] = None,
) -> MemoryAtom:
    raw_words = tuple((linked_words or _simple_tokenize(payload)))
    raw_tags = tuple(str(t).lower() for t in tags)
    for tok in list(raw_words) + list(raw_tags):
        tok = tok.lower().strip()
        if tok:
            self.token_frequency[tok] = self.token_frequency.get(tok, 0) + 1
    atom = _old_ingest_event_o181(
        self,
        kind=kind,
        payload=payload,
        tags=raw_tags,
        linked_words=raw_words,
        payload_ref=payload_ref,
        emotion=emotion,
        intimacy=intimacy,
    )
    self._recanonicalize_state()
    latest = next((m for m in reversed(self.state.memories) if m.id == atom.id), atom)
    return latest
CompanionOrchestrator.ingest_event = _ingest_event_o181

def warmth_unification_demo() -> Dict[str, Any]:
    companion = CompanionOrchestrator("tender")
    companion.ingest_text("blu again at the windo, soft and late, I remember us", tags=("late", "blu", "memory"))
    companion.ingest_song("Blue Window Demo", tags=("blue", "night", "song"))
    companion.ingest_text("still blue, still warm, still close", tags=("blue", "warm", "recurrence"))
    prompt_case = _language_prompt_descriptor("I feel strange tonight and I dont really know why", "reflect")
    language_domain = LanguageDomainPack(companion, prompt_case)
    l_engine, l_results, l_memory, l_forensics, l_scores = run_domain(language_domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=709), constitution="tender")
    language_result = l_results[0]

    descriptor = melody_descriptor_from_texts([(m.payload_ref + " " + " ".join(m.tags), m.tags) for m in companion.state.memories[-4:]])
    melody_domain = MelodyDomainPack(descriptor, anchor_symbol=companion.anchor().get("name"), warmth_profile=companion.warmth_profile)
    m_engine, m_results, m_memory, m_forensics, m_scores = run_domain(melody_domain, EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=711), constitution="endless")
    melody_result = m_results[0]

    snapshot = companion.snapshot_for_ios()
    snapshot["response"] = {
        "text": language_result.champion.text,
        "path": list(language_result.path),
        "proof": _metrics_to_dict(language_result.proof),
        "hidden": _metrics_to_dict(language_result.hidden),
    }
    snapshot["melody"] = {
        "champion_summary": melody_domain.summarize(melody_result.champion),
        "path": list(melody_result.path),
        "proof": _metrics_to_dict(melody_result.proof),
        "hidden": _metrics_to_dict(melody_result.hidden),
    }
    return {
        "snapshot": snapshot,
        "language_report": l_engine.build_report(l_results, l_memory, l_forensics, l_scores),
        "melody_report": m_engine.build_report(m_results, m_memory, m_forensics, m_scores),
    }
globals()['warmth_unification_demo'] = warmth_unification_demo

_old_build_release_report_o181 = build_release_report
def build_release_report(bundles: Dict[str, Any], auto: ImproveResult) -> str:
    base = _old_build_release_report_o181(bundles, auto)
    warm = warmth_unification_demo()
    extra = []
    extra.append("## Canonical spelling preference (o1.8.1)")
    extra.append("- Misspelled near-duplicates are now clustered and rewritten toward the cleaner canonical form, not the first form seen.")
    extra.append("- Companion memories, symbols, and lexicon rebuild after canon selection so meanings do not split into fake parallel branches.")
    extra.append(f"- Demo autocorrect aliases: {json.dumps(warm['snapshot'].get('autocorrect_aliases', {}), sort_keys=True)}")
    extra.append(f"- Demo anchor: {json.dumps(warm['snapshot'].get('anchor', {}), sort_keys=True)}")
    return base + "\n\n" + "\n".join(extra)
globals()['build_release_report'] = build_release_report

_old_package_info_o181 = package_info
def package_info() -> Dict[str, Any]:
    info = dict(_old_package_info_o181())
    info["version"] = __version__
    info["script_name"] = "forge_o1_8_1.py"
    info["autocorrect"] = "canonical_preference_rewrite"
    return info
globals()['package_info'] = package_info

def run_release_demo(write_report: bool = True) -> Dict[str, Any]:
    bundles: Dict[str, Any] = {}
    sorting_engine, sorting_results, sorting_memory, sorting_forensics, sorting_scores = run_domain(SortingDomainPack(), EngineConfig(max_depth=3, base_beam_width=6, meta_rounds=1, base_seed=19), constitution="mode_default")
    path_engine, path_results, path_memory, path_forensics, path_scores = run_domain(PathfindingDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=41), constitution="mode_default")
    regex_engine, regex_results, regex_memory, regex_forensics, regex_scores = run_domain(RegexDomainPack(), EngineConfig(max_depth=2, base_beam_width=4, meta_rounds=1, base_seed=211), constitution="mode_default")
    bundles["sorting"] = {"engine": sorting_engine, "results": sorting_results, "memory": sorting_memory, "forensics": sorting_forensics, "scores": sorting_scores}
    bundles["pathfinding"] = {"engine": path_engine, "results": path_results, "memory": path_memory, "forensics": path_forensics, "scores": path_scores}
    bundles["regex"] = {"engine": regex_engine, "results": regex_results, "memory": regex_memory, "forensics": regex_forensics, "scores": regex_scores}
    counter = CounterForgePlus(bundles)
    counter_reports, counter_md = counter.run()
    apply_counter_feedback(bundles, counter_reports)
    auto = autolift_demo()
    companion_snapshot = companion_demo()
    mel = melody_demo()
    lang = language_demo()
    warm = warmth_unification_demo()
    report_parts = [
        sorting_engine.build_report(sorting_results, sorting_memory, sorting_forensics, sorting_scores),
        path_engine.build_report(path_results, path_memory, path_forensics, path_scores),
        regex_engine.build_report(regex_results, regex_memory, regex_forensics, regex_scores),
        auto.report_markdown,
        counter_md,
        mel["report"],
        lang["report"],
        warm["melody_report"],
        warm["language_report"],
        build_release_report(bundles, auto),
        "## Companion snapshot demo\n- " + json.dumps(companion_snapshot, sort_keys=True),
        "## Melody snapshot demo\n- " + json.dumps(mel["snapshot"], sort_keys=True),
        "## Language snapshot demo\n- " + json.dumps(lang["snapshot"], sort_keys=True),
        "## Warmth snapshot demo\n- " + json.dumps(warm["snapshot"], sort_keys=True),
    ]
    combined = "\n\n---\n\n".join(report_parts)
    if write_report:
        Path("/mnt/data/forge_o1_8_1_report.md").write_text(combined, encoding="utf-8")
    print("FORGE o1.8.1")
    print("canonical spelling preference active")
    print("\n[language]")
    print("  reply:", warm["snapshot"]["response"]["text"])
    print("\n[melody]")
    print("  champion:", warm["snapshot"]["melody"]["champion_summary"])
    print("\n[autocorrect]")
    print("  aliases:", json.dumps(warm["snapshot"].get("autocorrect_aliases", {}), sort_keys=True))
    return {
        "bundles": bundles,
        "counter": counter_reports,
        "autolift": auto,
        "companion_snapshot": companion_snapshot,
        "melody_snapshot": mel["snapshot"],
        "language_snapshot": lang["snapshot"],
        "warmth_snapshot": warm["snapshot"],
    }
globals()['run_release_demo'] = run_release_demo

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge_o1_8_1", description="Forge o1.8.1 standalone engine")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    sub.add_parser("autolift-demo")
    sub.add_parser("regex-domain-demo")
    sub.add_parser("deep-ir-demo")
    sub.add_parser("constitution-demo")
    sub.add_parser("companion-demo")
    sub.add_parser("melody-demo")
    sub.add_parser("language-demo")
    sub.add_parser("warmth-demo")
    sub.add_parser("list-domains")
    return parser
globals()['build_cli'] = build_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
