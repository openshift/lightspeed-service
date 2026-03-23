"""Unit tests for PostgresBase and the connection decorator."""

from unittest.mock import MagicMock, call, patch

import psycopg2
import pytest

from ols.utils.postgres import PostgresBase, connection


class FakeComponent(PostgresBase):
    """Concrete subclass for testing PostgresBase."""

    @property
    def _ddl_statements(self) -> list[str]:
        return [
            "CREATE TABLE IF NOT EXISTS t1 (id int)",
            "CREATE INDEX IF NOT EXISTS i1 ON t1 (id)",
        ]


class TestConnectionDecorator:
    """Tests for the @connection decorator."""

    class Connectable:
        """Minimal connectable for decorator tests."""

        def __init__(self, raise_on_call: bool = False):
            """Initialize connectable."""
            self._connected = False
            self._raise_on_call = raise_on_call

        def connected(self) -> bool:
            """Check connection status."""
            return self._connected

        def connect(self) -> None:
            """Establish connection."""
            self._connected = True

        def disconnect(self) -> None:
            """Drop connection."""
            self._connected = False

        @connection
        def do_work(self) -> str:
            """Perform work requiring a connection."""
            if self._raise_on_call:
                raise RuntimeError("work failed")
            return "done"

    def test_auto_reconnects_when_disconnected(self):
        """Decorator calls connect() when not connected."""
        c = self.Connectable()
        c.disconnect()
        assert c.connected() is False

        result = c.do_work()
        assert c.connected() is True
        assert result == "done"

    def test_does_not_reconnect_when_connected(self):
        """Decorator skips connect() when already connected."""
        c = self.Connectable()
        c.connect()
        with patch.object(c, "connect") as mock_connect:
            c.do_work()
            mock_connect.assert_not_called()

    def test_propagates_exception_after_reconnect(self):
        """Decorator reconnects then lets the wrapped exception propagate."""
        c = self.Connectable(raise_on_call=True)
        c.disconnect()

        with pytest.raises(RuntimeError, match="work failed"):
            c.do_work()
        assert c.connected() is True


class TestPostgresBaseConnect:
    """Tests for PostgresBase.connect()."""

    def test_connect_executes_ddl_in_order_and_commits(self):
        """DDL statements are executed in declared order, then committed."""
        with patch("psycopg2.connect") as mock_connect:
            cursor = mock_connect.return_value.cursor.return_value
            FakeComponent(config=MagicMock())

        cursor.execute.assert_has_calls(
            [
                call("CREATE TABLE IF NOT EXISTS t1 (id int)"),
                call("CREATE INDEX IF NOT EXISTS i1 ON t1 (id)"),
            ]
        )
        cursor.close.assert_called_once()
        mock_connect.return_value.commit.assert_called_once()

    def test_connect_sets_autocommit_after_init(self):
        """Autocommit is set to True after successful initialization."""
        with patch("psycopg2.connect") as mock_connect:
            FakeComponent(config=MagicMock())

        assert mock_connect.return_value.autocommit is True

    def test_connect_closes_connection_on_ddl_failure(self):
        """Connection is closed and exception propagates when DDL fails."""
        with patch("psycopg2.connect") as mock_connect:
            cursor = mock_connect.return_value.cursor.return_value
            cursor.execute.side_effect = psycopg2.DatabaseError("CREATE failed")

            with pytest.raises(psycopg2.DatabaseError, match="CREATE failed"):
                FakeComponent(config=MagicMock())

        mock_connect.return_value.close.assert_called_once()

    def test_connect_does_not_set_autocommit_on_failure(self):
        """Autocommit stays at default when initialization fails."""
        with patch("psycopg2.connect") as mock_connect:
            cursor = mock_connect.return_value.cursor.return_value
            cursor.execute.side_effect = psycopg2.DatabaseError("fail")
            mock_connect.return_value.autocommit = False

            with pytest.raises(psycopg2.DatabaseError):
                FakeComponent(config=MagicMock())

        assert mock_connect.return_value.autocommit is False


class TestPostgresBaseConnected:
    """Tests for PostgresBase.connected()."""

    def test_connected_returns_true_on_healthy_connection(self):
        """connected() returns True when SELECT 1 succeeds."""
        with patch("psycopg2.connect"):
            component = FakeComponent(config=MagicMock())

        assert component.connected() is True

    def test_connected_returns_false_when_no_connection(self):
        """connected() returns False when connection is None."""
        with patch("psycopg2.connect"):
            component = FakeComponent(config=MagicMock())

        component.connection = None
        assert component.connected() is False

    def test_connected_returns_false_on_operational_error(self):
        """connected() returns False on OperationalError."""
        with patch("psycopg2.connect") as mock_connect:
            component = FakeComponent(config=MagicMock())

        cursor_mock = mock_connect.return_value.cursor.return_value
        cursor_mock.__enter__.return_value.execute.side_effect = (
            psycopg2.OperationalError("connection lost")
        )
        assert component.connected() is False

    def test_connected_returns_false_on_interface_error(self):
        """connected() returns False on InterfaceError."""
        with patch("psycopg2.connect") as mock_connect:
            component = FakeComponent(config=MagicMock())

        cursor_mock = mock_connect.return_value.cursor.return_value
        cursor_mock.__enter__.return_value.execute.side_effect = (
            psycopg2.InterfaceError("cannot reach server")
        )
        assert component.connected() is False
