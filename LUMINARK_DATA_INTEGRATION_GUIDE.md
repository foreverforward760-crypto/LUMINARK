# LUMINARK Corporate Data Integration — Prototype Guide

## Purpose

The repository contains Python connector classes intended to retrieve selected company data and pass it into a rule-based SAP corporate analyzer. This is an experimental integration, not a verified live-data service.

## Implemented connector attempts

- Yahoo Finance: requests quote-summary data and extracts selected financial/profile fields.
- Alpha Vantage: optional API-key connector for company overview data.
- NewsAPI: optional API-key connector used to calculate a basic news sentiment score.
- SEC EDGAR: optional email-configured connector intended to retrieve filing information.

These integrations have not been comprehensively tested against current provider APIs. Provider access, rate limits, authentication, response schemas, and terms can change. A successful HTTP response is not by itself proof that the extracted data is correct.

## Important SEC connector limitation

The current ticker-to-CIK lookup should be verified and corrected before relying on SEC filing retrieval. The implementation attempts to construct a submissions URL from the ticker symbol rather than first resolving the ticker to a valid CIK. Do not treat the SEC connector as operational until this is fixed and tested.

## How the analyzer handles missing data

The automated analyzer fills many missing corporate fields with fixed assumptions based on sector, employee count, market capitalization, or other available values. Examples include cash-reserve months, R&D percentage, turnover, management layers, employee satisfaction, market share, leadership tenure, and innovation metrics.

Those values are estimates, not retrieved company facts. The resulting SAP classification and investment signal are therefore exploratory rule-based outputs, not validated investment recommendations.

## Data score limitation

The current data-quality score awards points when source responses are present. It does not evaluate source accuracy, completeness, freshness, or provenance. It should be interpreted as a source-presence indicator only.

## Safe development sequence

1. Test each connector separately and save redacted sample responses.
2. Record source, timestamp, units, and missing-value handling for each field.
3. Distinguish observed values from estimates in the data model and user interface.
4. Fix and test ticker-to-CIK resolution for SEC data.
5. Add retries, timeout handling, and provider-specific rate limiting.
6. Test normalization with known companies and manually checked financial statements.
7. Keep all investment signals disabled or explicitly marked experimental until independently validated.

## No revenue or production claims

Do not use earlier pricing, throughput, accuracy, subscriber, or revenue projections in this repository as factual commercial estimates. They were planning scenarios, not measured results.