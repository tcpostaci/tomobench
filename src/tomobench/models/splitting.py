"""Leakage-safe split helpers for target-bearing ML samples.

The paper-facing observation corpus contains multiple acquisition geometries
for some of the same velocity targets.  A case-level random split therefore
does not necessarily create independent evaluation units.  This module keeps
all samples with the same canonical target identity in one split and chooses
the assignment with family balance and the configured case fractions as
competing objectives.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import Any, Sequence

from tomobench.datasets.target_identity import target_vector_sha256


SPLIT_NAMES = ("train", "validation", "test")


def target_grouped_family_split(
    samples: Sequence[Any],
    *,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[str, str]:
    """Assign case IDs to target-atomic, approximately family-balanced splits.

    The grouping key is the canonical SHA-256 identity of each target vector.
    Thus acquisition geometry, station seeds, and earthquake seeds cannot
    separate otherwise identical targets.  For the small paper corpus an
    exhaustive assignment is used within each family, minimizing deviation
    from the requested case fractions while requiring each feasible split to
    be represented.  A deterministic greedy fallback handles larger corpora.

    ``samples`` only needs ``case_id``, ``scenario``, and ``target_vector``
    attributes, which keeps the helper reusable for training and unit tests.
    """
    _validate_fractions(train_fraction, validation_fraction)
    if not samples:
        raise ValueError("Target-grouped splitting requires at least one sample.")

    groups_by_hash: dict[str, list[Any]] = defaultdict(list)
    case_ids_seen: set[str] = set()
    for sample in samples:
        case_id = str(getattr(sample, "case_id", ""))
        if not case_id:
            raise ValueError("Every sample must define a non-empty case_id.")
        if case_id in case_ids_seen:
            raise ValueError(f"Duplicate case_id encountered: {case_id}")
        case_ids_seen.add(case_id)
        target_hash = target_vector_sha256(getattr(sample, "target_vector"))
        groups_by_hash[target_hash].append(sample)

    group_families = {
        target_hash: {str(getattr(sample, "scenario", "")) for sample in group_samples}
        for target_hash, group_samples in groups_by_hash.items()
    }
    if any(not families for families in group_families.values()):
        raise ValueError("Every target group must have a non-empty scenario/family.")

    rng = random.Random(random_seed)
    if all(len(families) == 1 for families in group_families.values()):
        split_by_hash: dict[str, str] = {}
        families_to_hashes: dict[str, list[str]] = defaultdict(list)
        for target_hash, families in group_families.items():
            families_to_hashes[next(iter(families))].append(target_hash)
        for family in sorted(families_to_hashes):
            family_hashes = sorted(families_to_hashes[family])
            family_groups = [
                {"target_hash": target_hash, "samples": groups_by_hash[target_hash]}
                for target_hash in family_hashes
            ]
            assignment = _assign_groups(
                family_groups,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                random_generator=rng,
                preserve_family_balance=False,
            )
            split_by_hash.update(assignment)
    else:
        # A target identity can theoretically be shared by records labelled
        # with different families.  Keep it atomic and use a global objective
        # rather than silently allowing family-local assignments to conflict.
        groups = [
            {"target_hash": target_hash, "samples": group_samples}
            for target_hash, group_samples in sorted(groups_by_hash.items())
        ]
        split_by_hash = _assign_groups(
            groups,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            random_generator=rng,
            preserve_family_balance=True,
        )

    split_by_case_id: dict[str, str] = {}
    for target_hash, group_samples in groups_by_hash.items():
        split_name = split_by_hash[target_hash]
        for sample in group_samples:
            split_by_case_id[str(sample.case_id)] = split_name

    if set(split_by_case_id) != case_ids_seen:
        raise RuntimeError("Target-grouped split did not assign every input case exactly once.")
    _validate_target_group_atomicity(samples, split_by_case_id)
    return split_by_case_id


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    if train_fraction <= 0.0 or validation_fraction < 0.0:
        raise ValueError("Train fraction must be positive and validation fraction non-negative.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave room for a test split.")


def _assign_groups(
    groups: list[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    random_generator: random.Random,
    preserve_family_balance: bool,
) -> dict[str, str]:
    split_names = _feasible_split_names(
        len(groups),
        validation_fraction=validation_fraction,
        test_fraction=1.0 - train_fraction - validation_fraction,
    )
    if len(groups) <= 8:
        candidates = list(product(split_names, repeat=len(groups)))
        candidates = [
            candidate
            for candidate in candidates
            if set(split_names).issubset(set(candidate))
        ]
        random_generator.shuffle(candidates)
        best_assignment = min(
            candidates,
            key=lambda candidate: _assignment_score(
                groups,
                candidate,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                preserve_family_balance=preserve_family_balance,
            ),
        )
    else:
        best_assignment = _greedy_assignment(
            groups,
            split_names=split_names,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            random_generator=random_generator,
            preserve_family_balance=preserve_family_balance,
        )
    return {
        str(group["target_hash"]): split_name
        for group, split_name in zip(groups, best_assignment, strict=True)
    }


def _feasible_split_names(
    group_count: int,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[str, ...]:
    if group_count <= 0:
        raise ValueError("At least one target group is required.")
    if group_count == 1:
        return ("train",)
    if group_count == 2:
        if test_fraction > 0.0:
            return ("train", "test")
        if validation_fraction > 0.0:
            return ("train", "validation")
        return ("train",)
    names = ["train"]
    if validation_fraction > 0.0:
        names.append("validation")
    if test_fraction > 0.0:
        names.append("test")
    return tuple(names)


def _assignment_score(
    groups: list[dict[str, Any]],
    assignment: tuple[str, ...],
    *,
    train_fraction: float,
    validation_fraction: float,
    preserve_family_balance: bool,
) -> float:
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    total_cases = sum(len(group["samples"]) for group in groups)
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    family_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {split_name: 0 for split_name in SPLIT_NAMES}
    )
    family_totals: dict[str, int] = defaultdict(int)
    for group, split_name in zip(groups, assignment, strict=True):
        group_size = len(group["samples"])
        counts[split_name] += group_size
        for sample in group["samples"]:
            family = str(getattr(sample, "scenario", ""))
            family_counts[family][split_name] += 1
            family_totals[family] += 1

    score = sum(
        ((counts[split_name] / total_cases) - fractions[split_name]) ** 2
        for split_name in SPLIT_NAMES
        if fractions[split_name] > 0.0
    )
    if preserve_family_balance:
        for family, family_split_counts in family_counts.items():
            family_total = family_totals[family]
            score += sum(
                (
                    (family_split_counts[split_name] / family_total)
                    - fractions[split_name]
                )
                ** 2
                for split_name in SPLIT_NAMES
                if fractions[split_name] > 0.0
            )
    return score


def _greedy_assignment(
    groups: list[dict[str, Any]],
    *,
    split_names: tuple[str, ...],
    train_fraction: float,
    validation_fraction: float,
    random_generator: random.Random,
    preserve_family_balance: bool,
) -> tuple[str, ...]:
    ordered_groups = sorted(
        groups,
        key=lambda group: (-len(group["samples"]), str(group["target_hash"])),
    )
    assignment_by_hash: dict[str, str] = {}
    for index, group in enumerate(ordered_groups):
        if index < len(split_names):
            assignment_by_hash[str(group["target_hash"])] = split_names[index]
            continue
        candidates = list(split_names)
        random_generator.shuffle(candidates)
        candidate_scores: list[tuple[float, str]] = []
        for split_name in candidates:
            trial = dict(assignment_by_hash)
            trial[str(group["target_hash"])] = split_name
            trial_groups = [
                next(item for item in groups if str(item["target_hash"]) == target_hash)
                for target_hash in trial
            ]
            trial_assignment = tuple(trial[str(item["target_hash"])] for item in trial_groups)
            candidate_scores.append(
                (
                    _assignment_score(
                        trial_groups,
                        trial_assignment,
                        train_fraction=train_fraction,
                        validation_fraction=validation_fraction,
                        preserve_family_balance=preserve_family_balance,
                    ),
                    split_name,
                )
            )
        assignment_by_hash[str(group["target_hash"])] = min(candidate_scores)[1]
    return tuple(assignment_by_hash[str(group["target_hash"])] for group in groups)


def _validate_target_group_atomicity(
    samples: Sequence[Any],
    split_by_case_id: dict[str, str],
) -> None:
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        target_hash = target_vector_sha256(getattr(sample, "target_vector"))
        splits_by_hash[target_hash].add(split_by_case_id[str(sample.case_id)])
    crossing = sorted(target_hash for target_hash, splits in splits_by_hash.items() if len(splits) > 1)
    if crossing:
        raise RuntimeError(
            "Target-grouped split assigned a target identity to multiple splits: "
            + ", ".join(crossing)
        )


__all__ = ["SPLIT_NAMES", "target_grouped_family_split"]
