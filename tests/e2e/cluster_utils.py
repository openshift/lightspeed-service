"""Utilities for interacting with the OpenShift cluster."""

import subprocess


def create_user(name: str):
    """Create a service account user for testing."""
    try:
        subprocess.run(
            ["oc", "create", "sa", name],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error creating service account") from e


def delete_user(name: str):
    """Delete a service account user."""
    try:
        subprocess.run(
            ["oc", "delete", "sa", name],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error deleting service account") from e


def get_user_token(name: str):
    """Get the token for the service account user."""
    try:
        result = subprocess.run(
            ["oc", "create", "token", name],  # noqa: S603,S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting token for service account") from e


def grant_sa_user_access(name: str, role: str):
    """Grant the service account user access to OLS."""
    try:
        subprocess.run(
            [  # noqa: S603,S607
                "oc",
                "adm",
                "policy",
                "add-cluster-role-to-user",
                role,
                "-z",
                name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Error granting user access") from e


def get_ols_url(route_name):
    """Get the URL for the OLS route."""
    try:
        # Run 'oc get route <route_name> -o jsonpath={.spec.host}' command
        result = subprocess.run(
            [  # noqa: S603,S607
                "oc",
                "get",
                "route",
                route_name,
                "-o",
                "jsonpath={.spec.host}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        hostname = result.stdout.strip()
        return "https://" + hostname
    except subprocess.CalledProcessError as e:
        raise Exception("Error getting route hostname") from e
