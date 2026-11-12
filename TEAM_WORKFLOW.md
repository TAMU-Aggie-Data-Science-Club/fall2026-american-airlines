# Team Folder and Pull Request Guide

This guide explains how each team's work should be organized in the American
Airlines project and what to do before opening a pull request into `main`.

The rule is simple:

> **Each team owns one top-level folder. Every team-specific file belongs in
> that folder.**

## 1. Team folders in `main`

Use this naming pattern:

```text
team-XX/
```

Replace `XX` with the team's number. For example:

```text
team-04/
team-09/
team-11/
team-12/
```

Do not create a folder using a person's name, a vague name such as `final`,
or a shared folder such as `deliverable1/`.

Each team's folder should contain all of that team's work:

```text
team-XX/
├── README.md
├── deliverable1/
│   ├── README.md
│   ├── notebooks/
│   ├── reports/
│   └── figures/
├── deliverable2/
│   ├── README.md
│   ├── notebooks/
│   ├── reports/
│   └── figures/
├── deliverable3/
│   └── ...
├── data/
│   └── .gitkeep
└── src/
    └── ...
```

Create only the subfolders that the team needs.

### What goes where

- `team-XX/README.md`: team members, project goal, progress, how to run the
  work, and important decisions.
- `deliverableN/notebooks/`: notebooks for that deliverable.
- `deliverableN/reports/`: written analysis, methodology, and conclusions.
- `deliverableN/figures/`: charts and images used by the deliverable.
- `src/`: reusable code used by the team's notebooks.
- `data/`: local data work only. Do not commit datasets; `.gitkeep` is allowed.

Use descriptive names such as
`team-09/deliverable3/notebooks/passenger-features.ipynb`, not
`final.ipynb`, `new.ipynb`, or `copy2.ipynb`.

## 2. What belongs in the repository root

The root is for project-wide files only. Team-specific work must not be added:

- directly at the root;
- inside another team's folder;
- inside shared `deliverable1/`, `deliverable2/`, or similar folders;
- inside `.github/`, workflows, or secret-scanning configuration without PM
  approval; or
- as committed datasets, archives, credentials, or API keys.

Project-wide files such as `README.md`, `DATA.md`, and `DELIVERABLES.md` should
only be changed when the change applies to the entire project. Explain that
reason in the PR and avoid unrelated edits.

## 3. Starting a team's work

Before creating a branch:

1. Read the issue and its acceptance criteria.
2. Confirm the team's folder name with a PM if it does not exist.
3. Start from the latest `main`.
4. Make sure the working tree is clean.

```sh
git switch main
git pull origin main
git status
git switch -c team-XX/short-description
```

Good branch names include:

```text
team-09/deliverable3-passenger-features
team-11/deliverable2-model-comparison
```

Never work directly on `main`.

## 4. Requirements for a PR into `main`

Before requesting approval, confirm all of the following:

- The PR targets `main`.
- The branch is based on the latest `main`.
- Every team-specific changed file is below the correct `team-XX/` folder.
- The PR has one clear purpose and does not combine unrelated work.
- Notebooks run from top to bottom without errors.
- Data files, credentials, tokens, and secrets are not committed.
- Existing project documentation and governance files were not deleted to
  resolve a conflict.
- Required checks, including the secret scan, pass.
- The PR description explains what changed and how it was verified.
- Review comments have been answered or fixed.

Always inspect the PR's complete **Files changed** list. If a team notebook
appears at the root, under a shared deliverable folder, or under another
team's folder, move it before requesting review.

## 5. Recommended PR description

```markdown
## Team
Team XX

## Deliverable
Deliverable N: short name

## What changed
- Added `team-XX/deliverableN/notebooks/example.ipynb`
- Added `team-XX/deliverableN/README.md`

## Verification
- Restarted the notebook kernel and ran all cells successfully
- Confirmed no data or secrets were committed
- Confirmed every changed path is under `team-XX/`

## Follow-up work
- Known limitations:
- Future improvements:
```

## 6. Approval process

Follow this sequence:

1. Open or use an issue describing the goal and definition of done.
2. Create a branch from the latest `main`.
3. Keep all work inside the team's folder.
4. Make small, focused commits.
5. Run and inspect the notebooks or code.
6. Push the branch and open a PR into `main`.
7. Complete the PR description and checklist.
8. Fix failed checks before requesting final review.
9. Respond to review comments on the same PR.
10. Keep the branch current if `main` changes.
11. A PM or authorized officer performs the final review and merge.

Members may create branches, push changes, open PRs, and review teammates'
work. Members do not merge directly into `main`.

## 7. Common reasons a PR is returned

A PM will return a PR for changes when it:

- modifies a shared notebook that belongs to another team;
- adds a notebook or report at the repository root;
- mixes work from multiple teams without a project-wide reason;
- commits downloaded data or generated archives;
- includes credentials, tokens, connection strings, or personal information;
- deletes project documentation or workflows to make a conflict disappear;
- is based on an old `main` and has unresolved conflicts;
- does not explain how the work was tested; or
- includes unrelated cleanup that makes review difficult.

If a PR has conflicts, update it from the latest `main`, resolve the conflicts
carefully, run the checks again, and push the result. Do not force-push over a
teammate's branch without agreement.

## 8. Final team checklist

- [ ] My branch started from the latest `main`.
- [ ] My branch name identifies my team and change.
- [ ] All team work is under `team-XX/`.
- [ ] The team README explains what was added and how to run it.
- [ ] Notebooks run successfully from a clean/restarted kernel.
- [ ] No data or secrets are committed.
- [ ] The PR targets `main`.
- [ ] The PR description explains the change and verification.
- [ ] I checked the complete changed-files list.
