---
name: find-complexity
description: Find functions and methods with high cyclomatic complexity, excessive length, or too many parameters. Use when the user asks to find complex code, complexity hotspots, refactoring candidates, or wants to improve code maintainability.
disable-model-invocation: true
---

# Find Complexity Hotspots

Identify functions that are hard to review, test, and maintain.

## Rules

- Report findings, do not refactor. Refactoring is a separate task.
- Focus on production code (`ols/`). Skip tests unless explicitly asked.
- Rank by severity: highest complexity first.

## Step 1: Determine Scope

Ask the user:
- **Branch mode**: only files changed in the current branch vs main.
- **Full mode**: scan the entire `ols/` directory.

For branch mode:

```bash
git diff --name-only origin/main -- 'ols/' | grep '\.py$'
```

## Step 2: Cyclomatic Complexity

```bash
uvx radon cc <target> -s -n C -a
```

This shows functions with complexity grade C or worse (threshold: 11+).

Grades: A (1-5), B (6-10), C (11-15), D (16-20), E (21-25), F (26+).

## Step 3: Maintainability Index

```bash
uvx radon mi <target> -s -n B
```

This shows files with maintainability grade B or worse.

Grades: A (20+, good), B (10-19, medium), C (0-9, poor).

## Step 4: Cognitive Complexity

Cognitive complexity weights nesting depth — a 5-deep `if` scores much higher than 5 sequential `if`s.

```bash
uvx --with flake8-cognitive-complexity flake8 --select=CCR001 --max-cognitive-complexity=10 <target>
```

## Step 5: Function Length

Find long functions (30+ lines of logic, excluding docstrings and blank lines):

```bash
uvx radon raw <target> -s
```

Also use pylint for method length:

```bash
uv run pylint --disable=all --enable=too-many-statements,too-many-branches,too-many-return-statements,too-many-arguments,too-many-locals <target>
```

## Step 6: File Size

Find large files (500+ lines):

```bash
wc -l $(find <target> -name '*.py') | sort -rn | head -20
```

Files over 500 lines are candidates for splitting into focused modules.

## Step 7: Classify Findings

For each function found, classify:

| Category | Criteria | Action |
|----------|----------|--------|
| **Split** | High complexity + long body | Break into smaller functions |
| **Simplify** | High complexity + short body | Reduce branching (early returns, lookup tables) |
| **Parameterize** | Too many arguments (6+) | Group into config/dataclass |
| **Monitor** | Grade C, not growing | Note it, revisit if it gets worse |
| **Split file** | File over 500 lines | Break into focused modules |

## Step 8: Report

For each finding:

1. File, function name, line number
2. Complexity grade and score
3. Number of statements / arguments
4. Classification (split / simplify / parameterize / monitor)
5. Brief suggestion

Summary: total hotspots by grade, top 5 worst offenders.
