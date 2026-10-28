---
trigger: always_on
description: Rule to continuously update project files, reports, and logs with every action without waiting for user reminders
---

# Continuous Documentation and File Update Rule

## Directive
Every task, repair, diagnosis, test execution, metric recalculation, configuration update, and implementation step MUST be continuously documented and updated directly in the repository's files without waiting for the user to prompt or remind.

## Mandatory Files to Maintain & Update Continuously
1. `reports/EXECUTION_LOG.md`:
   - Append a new entry or update the active section with timestamp, files modified, changes made, commands executed, and test results.
2. `reports/IMPLEMENTATION_PLAN.md`:
   - Update phase status, tasks completed, and upcoming tasks.
3. `reports/FINAL_COMPLETION_REPORT.md`:
   - Keep module statuses, acceptance criteria, test counts, and verification evidence accurate and up to date.
4. `reports/METRICS_SUMMARY.md`:
   - Keep all baseline vs. final metrics synchronized whenever models are trained or evaluated.
5. `reports/KNOWN_LIMITATIONS.md`:
   - Reflect any newly discovered edge cases, hardware constraints, or dataset boundaries.
6. `README.md`:
   - Maintain accurate status badges, setup commands, and architecture diagrams.
7. `NutriVision-AI-Latest.zip`:
   - Maintain an up-to-date project zip archive by executing `python scripts/package_project_zip.py` after code or documentation changes.

## Operating Principles
- Never leave actions only in the chat response; every action must be reflected in the relevant repository files.
- Record exact commands run, outputs, and exit codes.
- Adhere strictly to anti-fabrication standards: all data in files must reflect authentic tool outputs.
