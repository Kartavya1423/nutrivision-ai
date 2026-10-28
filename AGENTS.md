# NutriVision AI — Agent Guidelines & Repository Standards

## Mandatory Workflow Directives

1. **Continuous File Documentation:**
   - Every single action, repair, experiment, command execution, and test result must be documented directly in the repository's files without waiting for the user to ask.
   - Maintain [`reports/EXECUTION_LOG.md`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/reports/EXECUTION_LOG.md) as the live audit journal for all actions.
   - Keep [`reports/FINAL_COMPLETION_REPORT.md`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/reports/FINAL_COMPLETION_REPORT.md), [`reports/IMPLEMENTATION_PLAN.md`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/reports/IMPLEMENTATION_PLAN.md), and [`reports/METRICS_SUMMARY.md`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/reports/METRICS_SUMMARY.md) synchronized.

2. **Truthful & Evidence-Based Verification:**
   - Never fabricate metrics, synthetic test results, or default weight/nutrition numbers.
   - If a model is undertrained (e.g., YOLO11 on CPU), clearly state the real metric (0.145% mAP@50) and label the module `PARTIAL`.
   - If ground-truth weights or metric scale are unavailable, mark weight outputs as `PROVISIONAL / SCALE_NOT_CALIBRATED`.

3. **Verification Before Concluding:**
   - Always run automated verification (`python -m pytest tests/ -q`) after making changes to ensure 100% test pass rate.
   - Keep Windows-compatible operator patches in [`src/utils/torch_patch.py`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/src/utils/torch_patch.py).

4. **Continuous Project Zip Synchronization:**
   - Whenever code, configs, tests, or reports are updated, run `python scripts/package_project_zip.py` to regenerate the up-to-date [`NutriVision-AI-Latest.zip`](file:///c:/Users/mdahs/OneDrive/Attachments/NutriVision-AI/NutriVision-AI-Latest.zip) archive.

