# LUMINARK Personal and Corporate Data Components — Audit Notes

**Status:** This document replaces earlier claims that the repository contained validated assessment questions, evidence-based interventions, or validated corporate datasets.

## Personal assessment content

The file luminark_personal_datasets.py contains manually assembled research references, interpretive mappings of selected historical figures to SAP stages, stage-specific text, and heuristic crisis-related rules.

These mappings are interpretations made for the SAP framework. The cited authors did not validate SAP stage assignments, and the presence of a published psychological theory does not validate the mapping of that theory to this framework. The historical case-study scores are not independently measured assessments.

The current content should be described as an exploratory reflection library, not a clinically validated assessment, diagnostic instrument, or evidence-based intervention protocol.

## Corporate data components

The repository includes connector code that attempts to request information from Yahoo Finance, Alpha Vantage, NewsAPI, and SEC EDGAR. The presence of connector classes does not establish that every endpoint currently works, that the returned data is complete, or that the information has been reconciled against authoritative financial statements.

The automated analyzer derives or estimates multiple fields when data is missing, including cash-reserve months, R&D spending, employee turnover, management layers, employee satisfaction, market share, competitive threats, leadership tenure, and market growth. Those estimates are hard-coded assumptions, not observed company facts.

The analyzer's source score is based on the presence of source responses, not on data accuracy, completeness, freshness, or reliability. Its stage-match score and investment signal are rule-based outputs and have not been validated against market outcomes.

## Permitted description

These components are prototypes for exploring how external company data and manually authored SAP rules might be combined. They are not a validated investment research service, predictive model, clinical assessment, or commercial data product.

## Required validation

- Verify each connector against its provider's current API and terms.
- Preserve raw source values, timestamps, and provenance.
- Separate observed data from estimates and defaults.
- Remove or label all estimated values in reports.
- Test the stage classifier against a pre-defined, independently reviewed dataset.
- Evaluate any investment signal against historical out-of-sample data and a baseline.
- Review privacy, licensing, and regulatory obligations before external use.