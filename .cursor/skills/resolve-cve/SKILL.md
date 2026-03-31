---
name: resolve-cve
description: >-
  Resolve a CVE vulnerability issue from Jira. Reads the
  CVE details, assesses impact, and either marks "not
  affected" with a Jira comment and transition, bumps the
  affected dependency, or implements a code fix. Use when
  the user says "cve", "resolve CVE", or provides a CVE
  Jira issue.
---

# resolve-cve

The user provides a Jira key (e.g., `OLS-789`) or a Jira
URL. If no specific issue is given, find CVEs to triage by
searching the current sprint in the **OpenShift Lightspeed
Service** (OLS) project:

```
project = OLS AND type = Vulnerability AND sprint in openSprints()
  AND summary ~ "openshift-lightspeed/lightspeed-service-api-rhel9"
  AND statusCategory = "To Do"
  ORDER BY priority DESC
```

Only process issues whose summary contains
`openshift-lightspeed/lightspeed-service-api-rhel9` — these
are the service CVEs. Skip issues targeting other components
(e.g., operator, console plugin).

The summary format is:
`CVE-YYYY-NNNNN openshift-lightspeed/lightspeed-service-api-rhel9: {Package}: {Title} [ols-N]`

If using Jira MCP and the `cloudId` is unknown, call
`getAccessibleAtlassianResources` to discover it, or ask
the user.

## Step 1: Read the CVE Issue

Fetch the issue via `getJiraIssue` with
`responseContentFormat: "markdown"`. The issue type is
`Vulnerability`, not a regular story.

Parse the data from these locations:

- **CVE ID** — embedded in the `summary` field, e.g.,
  `CVE-2026-33231 openshift-lightspeed/...: NLTK: ...`
- **Affected package** — mentioned in the description's
  `Flaw:` section (the description starts with boilerplate
  — "Security Tracking Issue", "Do not make this issue
  public" — skip to the flaw text after the `---` separator)
- **Vulnerable version range** — in the flaw prose
- **Fix reference** — upstream commit or PR link, if
  mentioned in the flaw text

Then look up severity externally:

- **CVSS score** — use `WebSearch` for the CVE ID on NVD
  (e.g., `CVE-2026-33231 NVD`) to get the severity rating

If the issue is missing a CVE ID or the affected package
is unclear from the flaw text, ask the user to clarify.

## Step 2: Assess Impact

Determine whether this project is affected:

1. **Check if the package is a dependency** — search
   `pyproject.toml` and `uv.lock` for the package name.
   Match case-insensitively (Jira may say `NLTK`, lock
   file has `nltk`). If not present at all, the project
   is **not affected**.
2. **Check the installed version** — find the exact version
   in `uv.lock`. Compare against the vulnerable version
   range from the advisory.
3. **Check if the vulnerable code path is reachable** — if
   the CVE targets a specific feature or module of the
   package, search the codebase for imports and usage of
   that feature. If the project never calls the affected
   API, it may be **not affected** even if the version
   is in range.
4. **Check transitive dependencies** — if the package isn't
   a direct dependency, check whether it appears as a
   transitive dependency in `uv.lock`. Trace which direct
   dependency pulls it in.

## Step 3: Present Assessment

Present the finding to the user clearly:

```
CVE Assessment: {CVE-ID}

Package: {package name}
Vulnerable versions: {range}
Installed version: {version from uv.lock}
Direct dependency: {yes/no — if no, pulled in by {parent}}

Verdict: {NOT AFFECTED / AFFECTED — bump needed / AFFECTED — code change needed}

Reasoning:
- {why this verdict — e.g., "package not in dependency tree",
  "installed version is outside vulnerable range",
  "vulnerable API is not used by this project",
  "project uses the affected code path in module X"}
```

**GATE — do not proceed without user acknowledgment.**
The user may have context that changes the verdict
(e.g., the package is used indirectly, or the feature
is enabled in production but not in tests). Present the
assessment and stop. Only continue after explicit "go".

## Step 4: Resolve

Based on the verdict and user acknowledgment:

### Path A: Not Affected

1. Add a comment to the Jira issue via
   `addCommentToJiraIssue` with `contentFormat: "markdown"`:

   ```
   **Assessment: Not Affected**

   {CVE-ID} targets {package} versions {range}.

   {Reason — one of:}
   - Package is not in the dependency tree.
   - Installed version ({version}) is outside the
     vulnerable range.
   - The vulnerable code path ({specific API/module}) is
     not used by this project.

   No action required.
   ```

2. Transition the issue to **Done / Closed** with
   resolution **"Won't Do"**. Call
   `getTransitionsForJiraIssue` to find the transition ID
   for "Done" or "Closed", then `transitionJiraIssue` with
   that ID and `resolution: { name: "Won't Do" }` in the
   fields.

### Path B: Dependency Bump

Follow the `deps-update` skill with the specific package
name. It will bump only that package, run all verification
gates, and raise a PR.

After the bump, verify the new version in `uv.lock` is
outside the vulnerable range. If the latest release is
still vulnerable, stop and tell the user — no fix is
available upstream yet.

Then add a Jira comment:

```
**Resolution: Dependency bumped**

{CVE-ID} targets {package} versions {range}.
Bumped {package} from {old version} to {new version}.

Lint/types/tests: passing.
```

Ask user about Jira transition (same as Path A step 2).

### Path C: Code Change (Rare)

1. Explain to the user what code change is needed and why.
   This is unusual — confirm the approach before
   implementing.
2. Make the targeted fix, write or update tests, and run
   `make verify && make check-types && make test-unit`.
3. Add a Jira comment summarizing the code change.
4. Ask user about Jira transition.

## Step 5: Report

```
CVE {CVE-ID} resolved for {story_id}.

Verdict: {Not Affected / Bumped {package} to {version} / Code fix applied}
Jira: {commented / commented + transitioned to {status}}

{If files changed:}
Files changed:
  - {list files}

Ready to commit.
{End if}
```

If the user wants a commit (Path B or C), use message:

```
fix: resolve {CVE-ID} — bump {package} to {version}
```

or for code changes:

```
fix: resolve {CVE-ID} — {brief description}
```

## Constraints

- **User acknowledgment required** — never act on the
  verdict without the user confirming the assessment.
  They may know things the codebase analysis cannot reveal.
- **Jira transitions** — Path A (Not Affected) transitions
  automatically to Done/Closed with resolution "Won't Do".
  For Paths B and C, ask the user which transition to use.
- **Minimal changes** — bump only the affected package,
  not all dependencies. Use `--upgrade-package`, not
  `--upgrade`.
- **Verify after every change** — lint, types, and unit
  tests must pass before declaring done.
- **Do not downplay severity** — if the project is
  affected, say so clearly. Do not stretch "not affected"
  reasoning to avoid work.
