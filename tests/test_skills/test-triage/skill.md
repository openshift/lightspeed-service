---
name: triage
description: Triage production incidents involving data corruption, data loss, slow performance, or outages. Classify severity and recommend immediate actions.
---

# Incident Report Triage

When a user provides an incident report, follow these steps in order.

## Step 1: Classify Severity

Read the incident text and classify:
- Contains "data loss" or "corruption" → Critical → go to Step 2
- Contains "slow" or "timeout" → Degraded → go to Step 3
- Contains "cosmetic" or "typo" → Low → go to Step 4

State the severity classification and transition reason.

## Step 2: Critical Incident Response

1. State: "This is a Critical incident requiring immediate action."
2. Identify the affected component from the report text.
3. Recommend: rollback to last known good state.
4. Recommend: notify the on-call team.
5. Proceed to Step 5.

## Step 3: Degraded Incident Response

1. State: "This is a Degraded incident requiring investigation."
2. Identify the affected component from the report text.
3. Recommend: capture diagnostics.
4. Recommend: scale up if load-related.
5. Proceed to Step 5.

## Step 4: Low Incident Response

1. State: "This is a Low priority incident."
2. Recommend: file a backlog ticket.
3. Proceed to Step 5.

## Step 5: Summary

Provide a structured summary:
- Severity: (Critical/Degraded/Low)
- Component: (identified from text)
- Actions: (list from the corresponding step)
