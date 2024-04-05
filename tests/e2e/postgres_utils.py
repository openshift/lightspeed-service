"""Postgres-related utilities."""

import os

import psycopg2


def _get_env(env_name):
    if env_name not in os.environ:
        raise ValueError(f"Environment variable {env_name} not set.")
    return os.getenv(env_name)


def retrieve_connection():
    """Perform setup e2e tests for conversation cache based on PostgreSQL."""
    global connection
    try:
        pg_host = _get_env("PG_HOST")
        pg_port = _get_env("PG_PORT")
        pg_user = _get_env("PG_USER")
        pg_password = _get_env("PG_PASSWORD")
        pg_dbname = _get_env("PG_DBNAME")
        connection = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            dbname=pg_dbname,
        )
        assert connection is not None
        return connection
    except Exception as e:
        print("Skipping PostgreSQL tests because of", e)
        return None


def read_conversation_history_count(postgres_connection):
    """Read number of items in conversation history."""
    query = "SELECT count(*) FROM cache;"
    with postgres_connection.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchone()


def read_conversation_history(postgres_connection, conversation_id):
    """Read number of items in conversation history."""
    query = "SELECT value, updated_at FROM cache WHERE conversation_id = %s"
    with postgres_connection.cursor() as cursor:
        cursor.execute(query, (conversation_id,))
        return cursor.fetchone()
