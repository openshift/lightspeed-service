"""Mock for Redis pipeline."""


class MockRedisPipeline:
    """Mocked Redis pipeline."""

    def __init__(self, mock_redis):
        """Initialize the pipeline, store instance of Redis client to use."""
        self.mock_redis = mock_redis
        self.transaction_started = False
        self.watch_started = False

    def __enter__(self):
        """Simulate ability of Redis pipeline to be used as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Simulate ability of Redis pipeline to be used as context manager."""

    def set(self, key, value):
        """Set item into the cache (implementation of SET command)."""
        self.mock_redis.set(key, value)

    def multi(self):
        """Begin the transaction."""
        assert not self.transaction_started, "Transaction started already"
        self.transaction_started = True

    def execute(self):
        """Commit the transaction."""
        assert self.transaction_started, "Transaction must be started"
        self.transaction_started = False

    def watch(self, key):
        """Start watching changes in database."""
        self.watch_started = True

    def unwatch(self):
        """Stop watching changes in database."""
        assert self.watch_started, "Can not UNWATCH when WATCH is not started"
        self.watch_started = False
