import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent import run as run_agent
from orchestrator import run as run_orchestrator


class EvalError(Exception):
    pass


@dataclass(frozen=True)
class EvalCase:
    name: str
    task: str
    runner: str
    expected_contains: List[str]
    max_steps: int = 8
    candidates: int = 2

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EvalCase":
        name = data.get("name")
        task = data.get("task")
        runner = data.get("runner", "agent")
        expected_contains = data.get("expected_contains", [])
        max_steps = data.get("max_steps", 8)
        candidates = data.get("candidates", 2)

        if not isinstance(name, str) or not name.strip():
            raise EvalError("Each eval case needs a non-empty 'name'.")
        if not isinstance(task, str) or not task.strip():
            raise EvalError(f"Eval case {name!r} needs a non-empty 'task'.")
        if runner not in {"agent", "orchestrator"}:
            raise EvalError(f"Eval case {name!r} has invalid runner: {runner}")
        if not isinstance(expected_contains, list) or not all(
            isinstance(item, str) and item for item in expected_contains
        ):
            raise EvalError(f"Eval case {name!r} needs string 'expected_contains' items.")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise EvalError(f"Eval case {name!r} needs max_steps > 0.")
        if not isinstance(candidates, int) or candidates <= 0:
            raise EvalError(f"Eval case {name!r} needs candidates > 0.")

        return EvalCase(
            name=name,
            task=task,
            runner=runner,
            expected_contains=expected_contains,
            max_steps=max_steps,
            candidates=candidates,
        )


def load_cases(path: str) -> List[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise EvalError("Eval file must contain a non-empty list of cases.")
    return [EvalCase.from_dict(item) for item in cases]


def _run_case(case: EvalCase, model: Optional[str], mock_answer: Optional[str]) -> str:
    if mock_answer is not None:
        return mock_answer

    if case.runner == "agent":
        return run_agent(case.task, max_steps=case.max_steps, model=model)

    return run_orchestrator(
        case.task,
        candidates=case.candidates,
        max_steps=case.max_steps,
        model=model,
        vote=True,
        verbose=False,
    )


def evaluate_text(answer: str, expected_contains: List[str]) -> Dict[str, Any]:
    lower = answer.lower()
    missing = [item for item in expected_contains if item.lower() not in lower]
    return {
        "passed": not missing,
        "missing": missing,
    }


def run_evals(
    cases: List[EvalCase],
    *,
    model: Optional[str] = None,
    mock_answer: Optional[str] = None,
    clock: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    started = clock()

    for case in cases:
        case_started = clock()
        try:
            answer = _run_case(case, model, mock_answer)
            check = evaluate_text(answer, case.expected_contains)
            error = None
        except Exception as e:
            answer = ""
            check = {"passed": False, "missing": case.expected_contains}
            error = str(e)

        results.append(
            {
                "name": case.name,
                "runner": case.runner,
                "passed": check["passed"],
                "missing": check["missing"],
                "error": error,
                "duration_sec": round(clock() - case_started, 3),
                "answer": answer,
            }
        )

    passed = sum(1 for item in results if item["passed"])
    return {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "duration_sec": round(clock() - started, 3),
        "results": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run simple keyword evals.")
    parser.add_argument("--cases", default="evals/smoke.json", help="Path to eval cases JSON.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument(
        "--mock-answer",
        default=None,
        help="Use a fixed answer instead of calling the LLM; useful for harness checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args(argv)

    summary = run_evals(load_cases(args.cases), model=args.model, mock_answer=args.mock_answer)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"passed={summary['passed']} failed={summary['failed']} total={summary['total']}")
        for item in summary["results"]:
            status = "PASS" if item["passed"] else "FAIL"
            detail = ""
            if item["missing"]:
                detail = " missing=" + ", ".join(item["missing"])
            if item["error"]:
                detail += " error=" + item["error"]
            print(f"{status} {item['name']} ({item['runner']}){detail}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
