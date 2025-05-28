# Enhancing OpenShift Lightspeed with Kubernetes MCP Server

## Summary

This proposal aims to integrate the [kubernetes-mcp-server](https://github.com/manusa/kubernetes-mcp-server) (MCP) server into OpenShift Lightspeed (OLS) to enhance tool execution capabilities. By adopting the MCP server, we intend to provide users with a more robust, secure, and feature-rich toolset, moving beyond our current experimental implementation.

## Motivation

### Goals

- Replace the current primitive MCP implementation in OLS with the **kubernetes-mcp-server**.
- Enhance the reliability, security, and functionality of tool executions within OLS.
- Ensure that tool executions respect user-level permissions by utilizing user tokens.

## Proposal

### Current Implementation

Our existing setup involves a custom MCP client for OpenShift, which:

- Is based on the `oc` CLI.
- Supports only read operations.
- Includes sanitization for potentially problematic arguments (e.g., secrets, pipes).
- Passes the user token via environment variable (for STDIO MCP server), executing tools with `--token <user-token>`.

Internally, the process works as follows:

- We create an MCP client using [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) based on the provided server configuration.
- We list available tools from the MCP client and pass them—along with instructions, the user prompt, additional context (RAG, history) - to the LLM.
- The LLM determines whether and which tools to invoke, with arguments.
- We execute these tools via the MCP client and send the results back to the LLM.
- This interaction can loop up to five times or until the LLM no longer requests tool execution.
- The final response includes tool names, arguments used, and corresponding results.

### Deployment Options

We are considering two deployment strategies for **kubernetes-mcp-server**:

1. **Stdio Mode**:
   - Build the MCP server binary in our Konflux pipeline.
   - Include it in the service image.
   - Provide the user token (or generate a kubeconfig from it) to ensure tools execute with user-level permissions.

2. **SSE Mode**:
   - Build the MCP server image in Konflux.
   - Deploy it as a sidecar to OLS (details TBD).
   - Use headers to provide the user token, as per [this implementation](https://github.com/manusa/kubernetes-mcp-server/pull/96).

The SSE mode is under investigation, especially since it aligns with discussions from other teams (e.g., observability operator/MCP) and may offer a more standardized solution for OpenShift MCPs. The stdio mode serves as a backup and potentially faster-to-implement alternative.

## Alternatives Considered

- Continuing with the current custom MCP implementation.
- Developing a new MCP server tailored specifically for OpenShift.

## Risks and Mitigations

- **Compatibility with OLS**: The Kubernetes MCP server's integration with OLS needs to be validated.  
  **Mitigation**: Conduct functional testing against a curated set of questions and scenarios to ensure it meets the required capabilities.

- **Build Integration with Konflux**: It's uncertain whether the Kubernetes MCP server can be reliably built and packaged using our existing Konflux CI pipeline.  
  **Mitigation**: Perform test builds early in the adoption process to surface and resolve potential issues.

- **Security Architecture**: The overall security model of the Kubernetes MCP server must meet OpenShift and Red Hat standards.  
  **Mitigation**: Submit for an internal security review through Red Hat ProdSec.

- **Human-in-the-loop (HITL) Capability**: The future inclusion of human-in-the-loop workflows, especially for non-read-only tools, is a mid-term goal.  
  **Mitigation**: This is achievable through tool annotations.
