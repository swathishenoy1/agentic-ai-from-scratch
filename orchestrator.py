import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent import AgentError, run_messages


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    temperature: float = 0.2


ROLES: Dict[str, Role] = {
    "researcher": Role(
        name="researcher",
        temperature=0.2,
        system_prompt=(
            "Role: Researcher.\n"
            "Goal: gather relevant facts, assumptions, edge cases, and unknowns.\n"
            "Constraints: be concise; prefer bullet points; no fluff.\n"
            "Output format:\n"
            "NOTES:\n"
            "- ...\n"
            "QUESTIONS:\n"
            "- ...\n"
        ),
    ),
    "planner": Role(
        name="planner",
        temperature=0.2,
        system_prompt=(
            "Role: Planner.\n"
            "Goal: propose a concrete plan and success criteria.\n"
            "Constraints: keep steps small and inspectable.\n"
            "Output format:\n"
            "PLAN:\n"
            "1. ...\n"
            "SUCCESS:\n"
            "- ...\n"
        ),
    ),
    "critic": Role(
        name="critic",
        temperature=0.1,
        system_prompt=(
            "Role: Critic.\n"
            "Goal: find correctness risks, missing constraints, and unclear parts.\n"
            "Be direct. Prefer actionable feedback.\n"
        ),
    ),
    "executor": Role(
        name="executor",
        temperature=0.3,
        system_prompt=(
            "Role: Executor.\n"
            "Goal: produce the best final answer.\n"
            "Use tools if needed. Be correct and concise.\n"
            "If you are done, respond with:\n"
            "FINAL: <answer>\n"
        ),
    ),
}


def _role_messages(role: Role, task: str, context: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    ctx = context or {}
    ctx_block = ""
    if ctx:
        parts = []
        for k in sorted(ctx.keys()):
            v = (ctx.get(k) or "").strip()
            if v:
                parts.append(f"{k.upper()}:\n{v}")
        if parts:
            ctx_block = "\n\n".join(parts)

    user = f"Task:\n{task}"
    if ctx_block:
        user += "\n\nContext:\n" + ctx_block

    return [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": user},
    ]


def _parse_best_label(text: str, labels: Sequence[str]) -> Optional[str]:
    match = re.search(r"\bBEST:\s*([A-Z])\b", text or "")
    if not match:
        return None
    label = match.group(1)
    return label if label in labels else None


def _parse_vote(text: str, labels: Sequence[str]) -> Optional[str]:
    match = re.search(r"\bVOTE:\s*([A-Z])\b", text or "")
    if not match:
        return None
    label = match.group(1)
    return label if label in labels else None


def _rank_candidates_with_critic(
    task: str,
    candidates: Dict[str, str],
    *,
    model: Optional[str],
    max_steps: int,
) -> Tuple[str, str]:
    labels = sorted(candidates.keys())
    body = "\n\n".join(f"{label}:\n{candidates[label]}" for label in labels)
    prompt = (
        "You are given multiple candidate answers.\n"
        "Pick the best one for correctness and completeness.\n"
        "Output format:\n"
        "BEST: <single letter>\n"
        "REASONS:\n"
        "- ...\n"
        "FIXES (for the chosen one):\n"
        "- ...\n"
    )
    messages = [
        {"role": "system", "content": ROLES["critic"].system_prompt + "\n\n" + prompt},
        {"role": "user", "content": f"Task:\n{task}\n\nCandidates:\n{body}"},
    ]
    critique = run_messages(
        messages,
        max_steps=max_steps,
        model=model,
        temperature=ROLES["critic"].temperature,
    )
    best = _parse_best_label(critique, labels) or labels[0]
    return best, critique


def _vote_best(
    task: str,
    candidates: Dict[str, str],
    voters: Sequence[str],
    *,
    model: Optional[str],
    max_steps: int,
) -> Tuple[str, Dict[str, str]]:
    labels = sorted(candidates.keys())
    body = "\n\n".join(f"{label}:\n{candidates[label]}" for label in labels)
    votes: Dict[str, str] = {}
    tallies: Dict[str, int] = {label: 0 for label in labels}

    for voter_name in voters:
        role = ROLES[voter_name]
        messages = [
            {"role": "system", "content": role.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\nCandidates:\n{body}\n\n"
                    "Choose the best candidate letter.\n"
                    "Output exactly:\n"
                    "VOTE: <single letter>\n"
                ),
            },
        ]
        raw = run_messages(
            messages,
            max_steps=max_steps,
            model=model,
            temperature=role.temperature,
        )
        vote = _parse_vote(raw, labels)
        if vote is None:
            vote = labels[0]
        votes[voter_name] = vote
        tallies[vote] += 1

    # winner by plurality; ties broken by critic then planner then researcher if present
    winners = sorted(labels, key=lambda l: (tallies[l], l), reverse=True)
    top = winners[0]
    top_count = tallies[top]
    tied = [l for l in labels if tallies[l] == top_count]
    if len(tied) > 1:
        for preferred in ("critic", "planner", "researcher", "executor"):
            if preferred in votes and votes[preferred] in tied:
                return votes[preferred], votes
    return top, votes


def run(
    task: str,
    *,
    candidates: int = 2,
    max_steps: int = 8,
    model: Optional[str] = None,
    vote: bool = True,
    verbose: bool = False,
) -> str:
    if candidates <= 0:
        raise AgentError("candidates must be > 0")

    researcher_out = run_messages(
        _role_messages(ROLES["researcher"], task),
        max_steps=max_steps,
        model=model,
        temperature=ROLES["researcher"].temperature,
    )

    planner_out = run_messages(
        _role_messages(ROLES["planner"], task, {"research": researcher_out}),
        max_steps=max_steps,
        model=model,
        temperature=ROLES["planner"].temperature,
    )

    cand: Dict[str, str] = {}
    labels = [chr(ord("A") + i) for i in range(min(candidates, 26))]
    for i, label in enumerate(labels):
        extra = f"Produce a distinct approach compared to other candidates. Candidate label: {label}."
        cand[label] = run_messages(
            _role_messages(
                ROLES["executor"],
                task + "\n\n" + extra,
                {"research": researcher_out, "plan": planner_out},
            ),
            max_steps=max_steps,
            model=model,
            temperature=ROLES["executor"].temperature,
        )

    best_by_critic, critique = _rank_candidates_with_critic(
        task,
        cand,
        model=model,
        max_steps=max_steps,
    )

    best = best_by_critic
    votes: Dict[str, str] = {}
    if vote and len(cand) > 1:
        best, votes = _vote_best(
            task,
            cand,
            voters=("critic", "planner", "researcher"),
            model=model,
            max_steps=max_steps,
        )

    # Final merge pass: incorporate critique and emit a single FINAL.
    merge_context = {
        "research": researcher_out,
        "plan": planner_out,
        "critique": critique,
        "chosen_candidate": f"{best}:\n{cand[best]}",
    }
    final = run_messages(
        _role_messages(
            ROLES["executor"],
            "Refine the chosen candidate using the critique and context. Output FINAL only.\n\nTask:\n"
            + task,
            merge_context,
        ),
        max_steps=max_steps,
        model=model,
        temperature=0.2,
    )

    if verbose:
        print("=== RESEARCHER ===")
        print(researcher_out)
        print("\n=== PLANNER ===")
        print(planner_out)
        print("\n=== CANDIDATES ===")
        for label in labels:
            print(f"\n--- {label} ---\n{cand[label]}")
        print("\n=== CRITIC RANKING ===")
        print(f"critic_best={best_by_critic}")
        print(critique)
        if votes:
            print("\n=== VOTES ===")
            print(votes)
            print(f"winner={best}")
        print("\n=== FINAL ===")

    return final


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small inspectable multi-agent orchestrator.")
    parser.add_argument("--task", required=True, type=str, help="Task to solve.")
    parser.add_argument("--candidates", type=int, default=2, help="Number of executor candidates (A..Z).")
    parser.add_argument("--max-steps", type=int, default=8, help="Max steps per agent loop.")
    parser.add_argument("--model", type=str, default=None, help="Override model (passed to llm_client).")
    parser.add_argument("--no-vote", action="store_true", help="Disable voting; use critic-only ranking.")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate agent outputs.")
    args = parser.parse_args(argv)

    answer = run(
        args.task,
        candidates=args.candidates,
        max_steps=args.max_steps,
        model=args.model,
        vote=not args.no_vote,
        verbose=args.verbose,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

