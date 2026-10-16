# Issue tracker: GitHub

Track issues and specifications in GitHub Issues:
CorbinMilne/cuda-telemetry-engine.

Use the `gh` CLI from the repository root.

## Common operations

- Read an issue: `gh issue view <number> --comments`
- List open issues: `gh issue list --state open`
- Create an issue:
  `gh issue create --title "..." --body-file <file>`
- Add a label:
  `gh issue edit <number> --add-label "<label>"`
- Remove a label:
  `gh issue edit <number> --remove-label "<label>"`
- Close an issue: `gh issue close <number>`

For multiline issue bodies or comments, use a temporary text file
and `--body-file`.

When a skill says to publish to the issue tracker, create a GitHub
issue. When it says to fetch a ticket, read the corresponding issue.

Creating or updating issues does not authorize implementation,
commits, or pushes. Follow the user's approval instructions.

## Pull requests as a triage surface

PRs as a request surface: no.
