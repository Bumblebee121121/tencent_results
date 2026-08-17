"""Evaluation candidate union and coverage helpers."""

from __future__ import annotations

from typing import Iterable, Mapping


def candidate_history_rid(oid: int, oid_to_rid: Mapping[int, int]) -> int | None:
    """Resolve a candidate OID without fabricating a historical RID."""

    value = oid_to_rid.get(int(oid))
    return None if value is None else int(value)


def ordered_candidate_additions(
    official_oids: Iterable[int],
    validation_target_oids: Iterable[int],
    test_target_oids: Iterable[int],
) -> tuple[list[int], list[int]]:
    """Return deterministic val additions and then new test additions."""

    official = {int(value) for value in official_oids}
    validation = {int(value) for value in validation_target_oids}
    test = {int(value) for value in test_target_oids}
    validation_additions = sorted(validation - official)
    test_additions = sorted(test - official - set(validation_additions))
    return validation_additions, test_additions


def coverage_count(target_oids: Iterable[int], candidate_oids: set[int]) -> tuple[int, int]:
    targets = [int(value) for value in target_oids]
    covered = sum(value in candidate_oids for value in targets)
    return covered, len(targets)
