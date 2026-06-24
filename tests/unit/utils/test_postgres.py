"""Unit tests for PostgresBase and the connection decorator."""

from unittest.mock import MagicMock, call, patch

import psycopg2
import pytest

from ols.app.models.config import TLSSecurityProfile
from ols.src.cache.cache_error import CacheError
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
            self._call_count = 0
            self._unhealthy_marked = False

        def connected(self) -> bool:
            """Check connection status."""
            return self._connected

        def connect(self) -> None:
            """Establish connection."""
            self._connected = True

        def disconnect(self) -> None:
            """Drop connection."""
            self._connected = False

        def _mark_unhealthy(self) -> None:
            """Mark as unhealthy."""
            self._unhealthy_marked = True

        @connection
        def do_work(self) -> str:
            """Perform work requiring a connection."""
            if self._raise_on_call:
                raise RuntimeError("work failed")
            return "done"

        @connection
        def do_work_operational_error(self) -> str:
            """Raise OperationalError on first call, succeed on retry."""
            self._call_count += 1
            if self._call_count == 1:
                raise psycopg2.OperationalError("connection lost")
            return "recovered"

        @connection
        def do_work_interface_error(self) -> str:
            """Raise InterfaceError on first call, succeed on retry."""
            self._call_count += 1
            if self._call_count == 1:
                raise psycopg2.InterfaceError("cannot reach server")
            return "recovered"

        @connection
        def do_work_database_error(self) -> str:
            """Raise DatabaseError (SQL error, no retry)."""
            raise psycopg2.DatabaseError("SQL syntax error")

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

    def test_operational_error_triggers_reconnect_and_retry(self):
        """OperationalError causes reconnect + retry, succeeding on second attempt."""
        c = self.Connectable()
        c.connect()
        result = c.do_work_operational_error()
        assert result == "recovered"
        assert c._unhealthy_marked is True

    def test_interface_error_triggers_reconnect_and_retry(self):
        """InterfaceError causes reconnect + retry, succeeding on second attempt."""
        c = self.Connectable()
        c.connect()
        result = c.do_work_interface_error()
        assert result == "recovered"
        assert c._unhealthy_marked is True

    def test_database_error_wraps_in_cache_error_no_retry(self):
        """DatabaseError is wrapped in CacheError without reconnect attempt."""
        c = self.Connectable()
        c.connect()
        with pytest.raises(CacheError, match="SQL syntax error"):
            c.do_work_database_error()

    def test_connection_error_reconnect_failure_raises_cache_error(self):
        """When reconnect fails after OperationalError, CacheError is raised."""
        c = self.Connectable()
        c.connect()
        # Make connect() fail on reconnect attempt
        original_connect = c.connect
        call_count = [0]

        def failing_connect():
            call_count[0] += 1
            if call_count[0] > 0:
                raise psycopg2.OperationalError("cannot connect")
            original_connect()

        c.connect = failing_connect
        with pytest.raises(CacheError, match="reconnect failed"):
            c.do_work_operational_error()


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
        # close is called twice: once for DDL cursor, once for statement_timeout cursor
        assert cursor.close.call_count == 2
        mock_connect.return_value.commit.assert_called_once()

    def test_connect_sets_autocommit_after_init(self):
        """Autocommit is set to True after successful initialization."""
        with patch("psycopg2.connect") as mock_connect:
            FakeComponent(config=MagicMock())

        assert mock_connect.return_value.autocommit is True

    def test_connect_sets_statement_timeout(self):
        """Statement timeout is set after autocommit is enabled."""
        mock_config = MagicMock()
        mock_config.statement_timeout = 5000
        with patch("psycopg2.connect") as mock_connect:
            cursor = mock_connect.return_value.cursor.return_value
            FakeComponent(config=mock_config)

        cursor.execute.assert_any_call("SET statement_timeout = %s", ("5000",))

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


class TestPostgresBaseTlsProfile:
    """Tests for TLS security profile integration in PostgresBase."""

    def _mock_config(self, profile: TLSSecurityProfile | None = None) -> MagicMock:
        """Return a MagicMock PostgresConfig with the given TLS profile."""
        cfg = MagicMock()
        cfg.ca_cert_path = None
        cfg.tls_security_profile = profile
        return cfg

    def test_ssl_min_protocol_version_passed_when_profile_set(self):
        """Verify psycopg2.connect receives ssl_min_protocol_version."""
        profile = TLSSecurityProfile()
        profile.profile_type = "IntermediateType"

        with patch("psycopg2.connect") as mock_connect:
            FakeComponent(config=self._mock_config(profile))

        kwargs = mock_connect.call_args.kwargs
        assert kwargs.get("ssl_min_protocol_version") == "TLSv1.2"

    def test_ssl_min_protocol_version_not_passed_when_no_profile(self):
        """Verify psycopg2.connect has no ssl_min_protocol_version without profile."""
        with patch("psycopg2.connect") as mock_connect:
            FakeComponent(config=self._mock_config())

        kwargs = mock_connect.call_args.kwargs
        assert "ssl_min_protocol_version" not in kwargs

    def test_ssl_min_protocol_version_not_passed_when_profile_type_is_none(self):
        """Verify psycopg2.connect skips enforcement when profile_type is None."""
        profile = TLSSecurityProfile()
        profile.profile_type = None

        with patch("psycopg2.connect") as mock_connect:
            FakeComponent(config=self._mock_config(profile))

        kwargs = mock_connect.call_args.kwargs
        assert "ssl_min_protocol_version" not in kwargs
