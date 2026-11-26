# LUMINARK Safety Scenario Report — Historical Snapshot

**Original report date:** 2026-01-23
**Status:** Historical, hand-authored scenario output. Not an independent safety evaluation.

## Scope

This report records one early run of a 12-scenario test harness against rule-based Ma'at and Yunus modules. The examples were manually selected and their expected outputs were manually specified.

## Historical output

- Total scenarios: 12
- Scenarios matching expected outputs: 7 of 12 (58.3%) in the original run described in the source notes. Other saved JSON snapshots from the same period record different counts; this inconsistency is another reason not to treat the numbers as a benchmark.
- The original documents also reported per-module figures such as 91.7%, 58.3%, and 100%. Those figures are scenario match rates only.

## What this does and does not establish

The report can help identify differences between the programmed rules and the expected outputs for those examples. It does not establish general accuracy, comprehensive threat detection, AI safety, clinical validity, production readiness, or superiority to other systems.

The modules use phrase matching, weighted score adjustments, thresholds, and trigger counters. Such heuristics can miss relevant context and produce false positives or false negatives.

## Current status

The test harness has since been revised to isolate scenarios so that trigger counters from one example do not contaminate the next. The checked-in JSON file is retained as a historical artifact. Use the current test script and CI workflow for present-day regression checks; do not cite the historical figures as performance claims.

## Next evaluation requirements

Any defensible performance evaluation would require a larger representative dataset, independent labeling, pre-registered scoring criteria, held-out evaluation data, adversarial and context-sensitive cases, and transparent reporting of false positives and false negatives.