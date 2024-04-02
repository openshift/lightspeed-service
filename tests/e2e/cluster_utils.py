"""Utilities for interacting with the OpenShift cluster."""

import subprocess


def run_oc(args: list[str]) -> subprocess.CompletedProcess:
    """Run a command in the OpenShift cluster."""
    try:
        result = subprocess.run(
            ["oc", *args],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        raise Exception("Error running command in OpenShift cluster") from e


def create_user(name: str) -> None:
    """Create a service account user for testing."""
    try:
        run_oc(["create", "sa", name])
    except subprocess.CalledProcessError as e:
        raise Exception("Error creating service account") from e


def delete_user(name: str) -> None:
    """Delete a service account user."""
    try:
        run_oc(["delete", "sa", name])
    except subprocess.CalledProcessError as e:
        raise Exception("Error deleting service account") from e


def get_user_token(name: str) -> str:
    """Get the token for the service account user."""
    try:
        result = run_oc(["create", "token", name])
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting token for service account") from e


def grant_sa_user_access(name: str, role: str) -> None:
    """Grant the service account user access to OLS."""
    try:
        run_oc(
            [
                "adm",
                "policy",
                "add-cluster-role-to-user",
                role,
                "-z",
                name,
            ]
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error granting user access") from e


def get_ols_url(route_name: str) -> str:
    """Get the URL for the OLS route."""
    try:
        result = run_oc(
            [
                "get",
                "route",
                route_name,
                "-o",
                "jsonpath={.spec.host}",
            ]
        )
        hostname = result.stdout.strip()
        return "https://" + hostname
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting route hostname") from e
