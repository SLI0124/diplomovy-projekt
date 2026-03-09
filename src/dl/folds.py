from __future__ import annotations

from dataclasses import dataclass

from config import RuntimeConfig


@dataclass(frozen=True)
class FoldSpec:
    test_year: int

    @property
    def train_end_year(self) -> int:
        return self.test_year - 1

    @property
    def train_years_label(self) -> str:
        return f"2013-{self.train_end_year}"


def folds_for_action(config: RuntimeConfig) -> list[FoldSpec]:
    if config.action == "test":
        return [FoldSpec(test_year=config.test_year)]

    start = 2014
    if config.test_year < start:
        raise ValueError("test-year must be >= 2014")

    return [FoldSpec(test_year=year) for year in range(start, config.test_year + 1)]
