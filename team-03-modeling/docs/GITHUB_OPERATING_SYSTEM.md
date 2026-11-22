# Team 03 — GitHub Operating System
**Status Nov 10, 2026:** Issues and root README live; board not yet configured; CI/branch rules not yet verified.

## One source of truth per surface
| Surface | Purpose |
| --- | --- |
| `docs/` on private `main` | Reviewed technical contracts, privacy rules, policies and decisions |
| Issues | Atomic work, one owner, dependencies, acceptance, exact path scope and evidence |
| GitHub Projects | A visual view of these same issues; do **not** create duplicate tasks |
| PRs | Actual implementation diff, test evidence, human/agent handoff and independent review |
| GitHub Actions | Fast synthetic/unit tests, lint and security checks; **no real AA data** |
| Google Drive | Existing handbook/technical plan (unchanged) and only explicitly authorized artifacts |
| Official ADSC GitHub | PM-governed final submissions; not the day-to-day work queue |
| Wiki | Optional read-only human mirror, never a second authoritative contract |

## Canonical issue hierarchy
- **Tier A core** epic #1; tasks #4–#11.
- **Tier B experiment factory** epic #2; tasks #12–#17.
- **Tier C optional research** epic #3; tasks #18–#20.
- All issues were **created as proposed backlog**; no member assignment or DONE state implied.

## Project board setup (requires GitHub UI/admin)
Create a GitHub **Projects v2** project in owner `AsvM45` called `AA Team 03 | Forecasting & Experiment Factory`. Add **existing issues #1–#20** from private repo; do not create clones. Recommended fields:
- Status: Backlog, Ready, Claimed, In Progress, Blocked, Ready for Review, Done.
- Tier: A Core, B Ambitious, C Research.
- Priority: P0, P1, P2.
- Workstream: Data & Privacy, Validation & Statistics, ML & Experiments, Software & QA, Report/PM, Learning.
- Owner, Reviewer, Dependencies, Estimate (person-hours), Deadline (ONLY verified), Evidence URL.
Views: **Critical path** (Tier A + gates), **Experiment factory** (Tier B), **Research** (Tier C), **By owner**, **Ready for review**, and a lean overall Kanban.

Permissions: inspect who can view this private Project before adding NDA-sensitive information. Prefer the same minimal public-safe issue metadata that is already in the repo.

## Agent/human issue claims
One issue = one active implementing owner; concurrent editors need separate physical worktrees/clones and unique branches. Post a claim containing owner/agent ID, branch, physical workspace, exact writable paths, dependencies, tests, and reviewer. A GitHub assignee/claim is only a record, **not an agent process**.

Branch examples: `human/asvath-issue-7`, `agent/codex-issue-13`. Do not reuse team-03 branches from the crowded official repo. No direct main pushes after explicit bootstrap exception, no force pushes, no automatic merge.

## PR standard
Link exactly one primary issue; report intended vs changed paths, actual test commands and observed outcomes, source/data versions without leaking data, failures, privacy checks, and reviewer. PR approval requires genuine tests, no leaked artifacts, and an independent reviewer with enough context to assess methods. Local green checks are not official acceptance.

## Protected output
Data and outputs are never uploaded to GitHub or hosted CI without express permission. GitHub can hold general/abstract schema, fake fixtures and control code. Model score tables, weights, routes/markets, source field names and real-data-derived charts can be sensitive.

## Initial owner actions
1. Confirm sponsor and PM permissions, D3 status, cutoff and approved private GitHub use: issue #4.
2. Review/accept team policy, initialize synthetic CI and configure branch ruleset: issue #5.
3. Create the Project v2 in GitHub UI, add all existing issues, configure statuses/tier and view; report actual URL in #5.
4. Ask each member for **sustainable new hours**, current lesson and GitHub ID. Assign only after explicit acceptance.
5. Integrate actual code through issue PRs; record real evidence. Keep a separate PM-owned official submission gate (#11).

Until the board exists, **the issues themselves are the operational backlog**.
