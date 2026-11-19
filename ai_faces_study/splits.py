"""Pure, label-independent participant-and-sequence split construction."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Fold:
    name: str
    held_subjects: tuple[str, ...]
    held_sequence: int
    train: tuple[int, ...]
    test: tuple[int, ...]


def crossed_holdout(
    subjects: Sequence[str],
    sequences: Sequence[int],
    participant_groups: Sequence[Sequence[str]],
    canonical_sequences: Sequence[int],
) -> tuple[Fold, ...]:
    """Hold out one participant group and one complete presentation sequence."""
    if len(subjects) != len(sequences) or len(subjects) == 0:
        raise ValueError("Nonempty, equally sized trial metadata are required")
    groups = tuple(tuple(g) for g in participant_groups)
    flat = tuple(s for group in groups for s in group)
    canonical = tuple(canonical_sequences)
    if len(groups) < 2 or any(not g for g in groups):
        raise ValueError("At least two nonempty participant groups are required")
    if len(flat) != len(set(flat)) or set(flat) != set(subjects):
        raise ValueError("Participant groups must partition all observed subjects")
    if len(canonical) < 2 or len(canonical) != len(set(canonical)):
        raise ValueError("At least two unique canonical sequences are required")
    if set(sequences) != set(canonical):
        raise ValueError("Observed and specified canonical sequences differ")
    trials = tuple(zip(subjects, sequences))
    observed_cells = set(trials)
    if any((s, q) not in observed_cells for s in flat for q in canonical):
        raise ValueError("Missing participant-by-sequence cell; review eligibility")

    folds = []
    coverage = [0] * len(subjects)
    for group_index, group in enumerate(groups, start=1):
        held = set(group)
        for sequence in canonical:
            train = tuple(i for i, (s, q) in enumerate(trials)
                          if s not in held and q != sequence)
            test = tuple(i for i, (s, q) in enumerate(trials)
                         if s in held and q == sequence)
            if not train or not test:
                raise ValueError("An outer fold has an empty training or test set")
            for i in test:
                coverage[i] += 1
            folds.append(Fold(f"f{group_index}_s{sequence}", group, sequence, train, test))
    if any(n != 1 for n in coverage):
        raise ValueError("Every trial must receive exactly one test prediction")
    return tuple(folds)
