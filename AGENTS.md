# Project instructions

## What we are building

CUDA Telemetry Engine is an undergraduate portfolio and learning project for
systems and GPU software engineering roles. It compares measurements with
reference values and reports sample count, threshold violations, maximum
absolute error, and mean squared error.

The current C++20 CPU implementation runs on an Apple Silicon Mac. It includes
synthetic telemetry generation, correctness tests, a command-line interface,
and repeated CPU timing. CUDA acceleration is the next major phase and needs
an NVIDIA GPU for compilation and execution. Keep CPU-only builds working.

Read `README.md` and the relevant source before proposing changes. Check the
actual implementation rather than assuming this overview is always current.
Use the CPU implementation as the correctness reference for future CUDA work.
Report GPU kernel time separately from time including transfers. Do not claim
GPU correctness or speedups before verifying them on NVIDIA hardware.

## Keep the code understandable

- Do not use em dashes.
- Do not overengineer. Prefer the smallest correct solution.
- Favor clear names, straightforward control flow, and the standard library.
- Do not add dependencies, generic frameworks, or layers for hypothetical needs.
- Readability matters more than minimizing line count or using clever syntax.
- When duplication or oversized functions make code harder to follow, propose
  a small refactor that removes the specific problem. Avoid unrelated rewrites.
- Preserve behavior during refactoring and verify it with relevant tests.
- Explain any extra complexity required for measured performance improvements.
- Add comments explaining important decisions, not obvious syntax.

## Learning and approval workflow

- Explain changes in plain language so the user can learn and take notes.
- Work in small steps. Explain what changes, why, and how to verify it.
- Show proposed code or documentation changes before editing.
- Wait for approval, then implement the approved changes.
- Refactors follow the same review and approval process as new features.
- Run relevant checks after changing code. See `README.md` for build commands.
- Commit and push only when requested.

## Agent skills

### Issue tracker

Use GitHub Issues in CorbinMilne/cuda-telemetry-engine.
See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels.
See `docs/agents/triage-labels.md`.

### Domain docs

Use one root glossary and one shared directory for design decisions.
See `docs/agents/domain.md`.
