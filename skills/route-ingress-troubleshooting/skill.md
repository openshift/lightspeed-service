---
name: route-ingress-troubleshooting
description: Troubleshoot Route or Ingress connectivity failures. Use when traffic returns 502, 503, connection refused, or the endpoint is not reachable externally.
---

# Route and Ingress Troubleshooting

When a user reports that their application URL returns errors (503, connection refused, TLS errors) or is completely unreachable, follow this structured approach to trace the request path and identify the broken layer.

## 1. Verify the Route Exists and Is Admitted

Check the Route object first:

1. Confirm the Route exists in the expected namespace.
2. Check the Route's status — is it `Admitted` by the ingress controller? If not admitted, the ingress controller has rejected it. Check the rejection reason (conflicting hostname, invalid TLS config, etc.).
3. Verify the hostname — does it match what the user is trying to reach? Typos in the hostname are common.
4. If using a custom domain: confirm DNS resolves to the cluster's ingress VIP or load balancer.

Do not proceed to service/endpoint debugging until the Route itself is confirmed as Admitted.

## 2. Verify the Target Service

Check that the Route points to a valid Service:

1. Confirm the Service named in the Route's `spec.to` exists in the same namespace.
2. Check the Service's `selector` — does it match the labels on the running pods?
3. Check the Service's `targetPort` — does it match the port the application is actually listening on inside the container?

If the Route references a Service that does not exist, that is the root cause. If the Service exists but has the wrong selector or port, report the specific mismatch.

## 3. Check Endpoints

Verify that the Service has backend pod IPs:

1. Check the Endpoints object for the Service — are there IP addresses listed?
2. If **endpoints are empty**: the Service selector does not match any **running and ready** pods. Report the exact selector and the labels on available pods so the user can see the mismatch.
3. If **endpoints exist but the application returns 503**: the pods are registered but may be failing readiness probes. Check the pod readiness probe configuration and recent probe failure events.

Empty endpoints are the most common cause of 503 errors on Routes. Always check this before investigating the ingress controller.

## 4. Verify Pod Readiness

If endpoints exist but the application is still unreachable:

1. Check if the pods are `Ready` (all readiness probes passing).
2. If pods are not ready: check which readiness probe is failing and why (wrong path, wrong port, application not fully started).
3. Verify the container is listening on the expected port — a mismatch between the declared containerPort and the actual listening port causes silent failures.
4. If pods are ready and the port is correct: the issue may be application-level (the app returns errors for the specific request path).

## 5. Diagnose TLS Issues

If the error is TLS-related (certificate errors, HTTPS not working):

1. Identify the Route's TLS termination type: **edge**, **reencrypt**, or **passthrough**.
2. For **edge** termination:
   - The ingress controller terminates TLS. Check if the Route has a custom certificate or uses the default wildcard certificate.
   - If custom certificate: verify the certificate matches the hostname and is not expired.
3. For **reencrypt** termination:
   - The ingress controller terminates and re-encrypts to the backend. Check the `destinationCACertificate` — it must trust the backend pod's certificate.
   - Verify the backend pod is serving TLS on the target port.
4. For **passthrough** termination:
   - TLS is terminated by the application pod. The ingress controller does not inspect the certificate.
   - Verify the pod is serving valid TLS on the expected port.

Report which TLS layer has the issue — do not suggest regenerating all certificates when only one is wrong.

## 6. Check the Ingress Controller

If the Route is Admitted, the Service has endpoints, pods are ready, and TLS is correct, the issue may be at the ingress controller level:

1. Check the IngressController/router pods in `openshift-ingress` — are they running and ready?
2. Check the IngressController's status conditions for errors.
3. If using a non-default IngressController: verify the Route is exposed by the correct one (check `routeSelector` and `namespaceSelector`).

Ingress controller issues are rare compared to Service/Endpoint issues. Only investigate here after ruling out the common causes.

## Quality Standards

- Always trace the full chain: Route → Service → Endpoints → Pod. Do not skip layers — report which specific layer is broken.
- Report the exact selector mismatch if endpoints are empty — show both the Service selector and the pod labels side by side.
- For TLS issues: specify which termination type is in use and which side has the problem. Generic "check your certificates" is not helpful.
- Do not suggest creating a new Route if the existing one has a fixable misconfiguration. Fix what exists first.
- If the issue is application-level (the app itself returns errors), say so clearly rather than continuing to troubleshoot infrastructure.
