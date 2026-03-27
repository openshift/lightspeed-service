---
name: review-pr
description: Review PR with structured approach covering architecture, naming, patterns, and critical questions
disable-model-invocation: true
---

# Review PR

When asked to review a PR, follow this structured approach.

## 1. Fetch Latest Changes

**Always** fetch the latest PR state before reviewing. The cached PR data may be stale.

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git log pr-<PR_NUMBER> --oneline -10
git diff upstream/main...pr-<PR_NUMBER> --stat
```

For follow-up reviews, re-fetch to get new commits:

```bash
git fetch upstream pull/<PR_NUMBER>/head:pr-<PR_NUMBER> --force
```

Read diffs per area (endpoints, models, utils, tests) rather than one massive diff.

## 2. Understand What It Implements

- Summarize the feature/fix in 2-3 sentences
- Identify the flow: entry point -> processing -> output
- Map which files serve which role (endpoint, model, utility, config)

## 3. Evaluate How It's Implemented

Only raise issues if you have a concrete concern — not as a checklist to fill:

- **Architecture**: Is logic clearly in the wrong layer? (e.g. business logic leaking into an endpoint)
- **Error handling**:
  - Are errors silently swallowed, or is an exception missing where failure is plausible?
  - **Pattern consistency**: If a function has N-1 error paths that degrade gracefully (return None, log warning, fallback), verify the Nth path does too. A single unguarded call among guarded ones is the most common miss.
  - **Cross-boundary exceptions**: For every new method call added in the diff, check what the **caller** does if that method throws. Read the callee's exception paths, then verify the caller handles them. Don't treat methods as self-contained.
- **Duplication**: Run the [find-duplication](../find-duplication/SKILL.md) skill in branch mode, scoped to the PR's changed files.
- **Complexity**: Run the [find-complexity](../find-complexity/SKILL.md) skill in branch mode, scoped to the PR's changed files. Flag any function the PR adds or worsens to grade C or higher.
- **Dead code / docs**: Is there obviously unused code or a doc update that's clearly missing?

Skip this section if nothing stands out.

## 4. Assess Naming

Only flag naming if it is genuinely misleading or inconsistent with established patterns in the codebase. Do not flag stylistic preferences or minor wording variations.

## 5. Check Pythonic Patterns

Only flag if the pattern causes a real problem (correctness, type safety, maintainability), not just because an alternative exists:

- Hand-rolled parsing where Pydantic would enforce correctness
- Missing type hints on public functions, or `Any` used where a concrete type is knowable
- Improper use of framework features (e.g. bypassing FastAPI dependency injection, skipping Pydantic validators)
- JSON-inside-JSON when native types work

Skip this section if nothing meaningful to flag.

## 6. Ask Critical Questions

Only ask questions where the answer is genuinely unclear from the code and matters for correctness or design. Do not manufacture questions for completeness.

- What happens on invalid/missing/malformed input — if error paths are not visible in the diff?
- Are there security implications not addressed? (token logging, size limits, injection)
- Do tests cover behavior (specific assertions) or just confirm the code runs?
- Is the PR clearly bundling unrelated changes that should be in a separate PR?

Skip this section if the design is clear and the concerns above don't apply.

## 7. Verify Each Issue with a Subagent

Before writing the final output, launch one subagent per candidate issue to confirm it is real. Do this in parallel for all issues found in sections 3–6.

Each subagent task should:
- Receive the specific concern as its prompt (e.g. "Is error X silently swallowed? Check `foo.py` lines 40-55 and any callers.")
- Read the relevant file(s) and any related context (callers, tests, existing patterns in the codebase)
- Return a verdict: **confirmed** / **not an issue** / **unsure**, with a one-sentence rationale

After all subagents complete:
- Drop any issue whose verdict is **not an issue**
- Downgrade confidence to **unsure** for any issue whose verdict is **unsure**
- Only **confirmed** issues appear in the issues table at full confidence

This step exists to filter out false positives before they reach the output. Do not skip it when there are candidate issues.

## 8. Output Format

Structure the review as:

1. **Summary**: What the PR does (2-3 sentences)
2. **File-by-file analysis**: Role of each changed file (one line each; skip files with trivial changes)
3. **Issues table**: Only issues confirmed by subagent verification (section 7) — Priority (must-fix / should-fix / nice-to-have), issue, location, confidence. If there are no confirmed issues, say so explicitly.
4. **What's good**: Acknowledge well-done aspects — keep this genuine, not filler
5. **Critical questions**: Only if section 6 above produced anything; omit otherwise
6. **Closing reminder**: Always end the review with: *"You, as a human, need to evaluate these AI findings instead of just copying them as review comments. That would just shift the responsibility of validation to the PR creator."*
