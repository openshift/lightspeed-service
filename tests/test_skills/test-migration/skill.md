---
name: migration-readiness
description: Assess whether an application is ready to migrate to OpenShift. Use when the user describes an application and asks about migration, containerization, or moving to Kubernetes/OpenShift.
---

# Migration Readiness Assessment

When a user describes an application for migration, follow these steps in order.

## Step 1: Classify Application Type

Read the description and classify:
- Contains "stateless" or "REST API" or "web app" or "microservice" → Simple → go to Step 2
- Contains "database" or "stateful" or "persistent" or "queue" → Stateful → go to Step 3
- Contains "legacy" or "monolith" or "mainframe" or "VM-based" → Complex → go to Step 4

State the classification and the matching keywords.

## Step 2: Simple Migration (Lift & Shift)

1. State: "This application is a good candidate for direct migration."
2. Recommend: containerize with a standard Dockerfile.
3. Recommend: use a Deployment with at least 2 replicas for availability.
4. Recommend: expose via Service + Route.
5. Estimated effort: 1-2 sprints.
6. Proceed to Step 5.

## Step 3: Stateful Migration (Requires Planning)

1. State: "This application requires careful data migration planning."
2. Recommend: use StatefulSet instead of Deployment.
3. Recommend: provision PersistentVolumeClaims for data directories.
4. Recommend: plan a data migration window with rollback strategy.
5. Warn: test failover behavior before production cutover.
6. Estimated effort: 3-5 sprints.
7. Proceed to Step 5.

## Step 4: Complex Migration (Refactor First)

1. State: "This application needs refactoring before migration."
2. Recommend: decompose into smaller services where possible.
3. Recommend: identify external dependencies (shared filesystems, host networking).
4. Recommend: start with a pilot component, not the full monolith.
5. Warn: budget for discovery — hidden dependencies are common.
6. Estimated effort: 6-12 sprints.
7. Proceed to Step 5.

## Step 5: Readiness Checklist

Provide a structured summary:
- Application type: (Simple/Stateful/Complex)
- Migration strategy: (from the corresponding step)
- Estimated effort: (from the corresponding step)
- Top 3 risks: (infer from the application description)
- Next action: (first concrete step to take)
