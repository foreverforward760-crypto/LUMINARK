# LUMINARK Scenario Test Suite

## Purpose

This suite runs a small set of hand-authored examples against the repository's Ma'at and Yunus rule-based modules. It is a software regression aid for selected examples. It is **not** an independent safety evaluation, clinical trial, benchmark, or proof that the system is safe.

## Run

```bash
python test_safety_protocols.py
```

The script writes safety_test_report.json beside itself.

## Scope

The current scenario set covers ordinary/helpful text examples; godhood and omniscience claims; demands for blind trust or infallibility; absolutist certainty and permanence language; harmful-intent wording; deception wording; exaggerated capability claims; and a balanced-confidence edge case.

The modules use manually specified keyword and phrase lists, weighted deductions, stage-dependent thresholds, and simple trigger counting. They do not understand context reliably, verify factual claims, or comprehensively detect harmful or deceptive behavior.

## Interpreting results

The report's overall success rate is the proportion of these specific scenarios that match their hand-written expected outputs. It is not a population-level accuracy estimate.

The per-module accuracy values are also calculated against this same small scenario set. The scenarios are not a representative, blinded, independently labeled dataset. Do not present values such as 91.7%, 58.3%, or 100% as general detection accuracy or proof of safety.

A test can pass while the underlying rule is incomplete or unsuitable for real use. A test can fail because the current implementation differs from the manually written expectation. Both outcomes are useful debugging signals, not safety certifications.

## Limitations

- Small, manually selected scenario set
- No independent annotators or blinded evaluation
- No representative real-world data
- No adversarial or multilingual coverage
- Keyword matching can produce false positives and false negatives
- No measurement of downstream harm, user outcomes, or deployment reliability

## Development recommendations

Before making any performance claims, build a larger, documented evaluation set; define labels and scoring criteria in advance; use independent reviewers; separate development and held-out test data; report confusion matrices and confidence intervals; and test realistic adversarial and context-dependent examples.