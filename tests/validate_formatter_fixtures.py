"""Run deterministic formatter quality checks across transcript fixtures."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_mom.experimental.integration import generate_mom
from ml_mom.experimental.mom_formatter import decision_outcome_key, semantic_topic_key

FIXTURES = sorted((ROOT / "datasets" / "raw_transcripts").glob("*.txt")) + sorted(
    (ROOT / "tests" / "fixtures").glob("*.txt")
)


def normalized_task_key(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - {
        "the",
        "a",
        "an",
        "was",
        "were",
        "is",
        "will",
        "be",
        "team",
    }


def validate_fixture(path: Path) -> tuple[str, int, int, int]:
    result = generate_mom(path.read_text(encoding="utf-8"), meeting_title=path.stem)
    if not result.is_valid or result.experimental_mom is None:
        raise AssertionError(f"{path.name}: generation failed: {result.error_message}")
    mom = result.experimental_mom

    decision_keys = [decision_outcome_key(item) for item in mom.decisions]
    assert len(decision_keys) == len(set(decision_keys)), f"{path.name}: duplicate decisions"

    action_tasks = [item.task for item in mom.action_items]
    assert not any(
        re.search(
            r"(?i)\b(good morning|good afternoon|welcome|let'?s begin|"
            r"today'?s meeting|we'?ll review)\b",
            task,
        )
        for task in action_tasks
    ), f"{path.name}: greeting/introduction leaked into actions"

    for decision in mom.decisions:
        decision_tokens = normalized_task_key(decision)
        for task in action_tasks:
            task_tokens = normalized_task_key(task)
            overlap = len(decision_tokens & task_tokens) / max(1, len(task_tokens))
            assert overlap < 0.8, f"{path.name}: decision duplicated as action: {task}"

    headings = [point.split(":", 1)[0] for point in mom.discussion_points]
    heading_keys = [semantic_topic_key(heading) for heading in headings]
    assert len(heading_keys) == len(set(heading_keys)), f"{path.name}: duplicate topics"
    assert all(len(heading.split()) >= 2 for heading in headings), (
        f"{path.name}: generic one-word discussion title"
    )

    repeat = generate_mom(path.read_text(encoding="utf-8"), meeting_title=path.stem)
    assert repeat.deterministic_markdown == result.deterministic_markdown, (
        f"{path.name}: formatter output is not deterministic"
    )
    return path.name, len(mom.decisions), len(action_tasks), len(headings)


if __name__ == "__main__":
    for fixture in FIXTURES:
        name, decisions, actions, topics = validate_fixture(fixture)
        print(f"PASS {name}: decisions={decisions}, actions={actions}, topics={topics}")
