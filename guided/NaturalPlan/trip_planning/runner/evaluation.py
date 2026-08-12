from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from architect.state_schema import Segment, TripProblem
from builder.nl2state_agent import parse_trip_example


@dataclass(frozen=True)
class EvaluationResult:
    example_id: str
    success: bool
    exact_match: bool
    predicted_plan: str
    golden_plan: str | None = None
    error: str | None = None


def evaluate_prediction(
    example_id: str,
    raw_example: Dict[str, Any],
    predicted_plan: str,
    include_golden: bool = False,
) -> EvaluationResult:
    problem = parse_trip_example(example_id, raw_example)
    success = satisfies_constraints(problem, predicted_plan)
    golden = raw_example.get("golden_plan")
    exact = _normalize_plan(predicted_plan) == _normalize_plan(golden or "")
    return EvaluationResult(
        example_id=example_id,
        success=success,
        exact_match=exact,
        predicted_plan=predicted_plan,
        golden_plan=golden if include_golden else None,
    )


def summarize(results: Iterable[EvaluationResult]) -> Dict[str, float]:
    result_list = list(results)
    if not result_list:
        return {"count": 0, "SR": 0.0, "EM": 0.0}
    return {
        "count": len(result_list),
        "SR": sum(1 for result in result_list if result.success) / len(result_list),
        "EM": sum(1 for result in result_list if result.exact_match) / len(result_list),
    }


def satisfies_constraints(problem: TripProblem, plan_text: str) -> bool:
    segments = _parse_plan_segments(plan_text)
    if len(segments) != problem.num_cities:
        return False
    if tuple(segment.city for segment in segments) != tuple(dict.fromkeys(segment.city for segment in segments)):
        return False
    if set(segment.city for segment in segments) != set(problem.cities):
        return False
    if segments[0].start_day != 1 or segments[-1].end_day != problem.total_days:
        return False

    for index, segment in enumerate(segments):
        if segment.duration != problem.durations.get(segment.city):
            return False
        if segment.end_day - segment.start_day + 1 != segment.duration:
            return False
        if index > 0:
            previous = segments[index - 1]
            if segment.start_day != previous.end_day:
                return False
            if (previous.city, segment.city) not in problem.flights:
                return False

    for constraint in problem.constraints:
        if not any(
            segment.city == constraint.city
            and segment.start_day <= constraint.start_day
            and segment.end_day >= constraint.end_day
            for segment in segments
        ):
            return False
    return True


def _parse_plan_segments(plan_text: str) -> List[Segment]:
    import re

    segment_re = re.compile(
        r"\*\*Day\s+(\d+)-(\d+):\*\*\s+(?:Arriving in\s+)?(.+?)\s+and visit\s+\3\s+for\s+(\d+)\s+days\.|"
        r"\*\*Day\s+(\d+)-(\d+):\*\*\s+Visit\s+(.+?)\s+for\s+(\d+)\s+days\.",
        re.IGNORECASE,
    )
    segments: List[Segment] = []
    for match in segment_re.finditer(plan_text):
        if match.group(1):
            start, end, city, duration = match.group(1), match.group(2), match.group(3), match.group(4)
        else:
            start, end, city, duration = match.group(5), match.group(6), match.group(7), match.group(8)
        segments.append(Segment(city=city.strip(), start_day=int(start), end_day=int(end), duration=int(duration)))
    return segments


def _normalize_plan(plan: str) -> str:
    return "\n".join(line.rstrip() for line in plan.strip().splitlines())

