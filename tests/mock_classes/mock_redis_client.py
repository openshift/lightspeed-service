"""Mock for StrictRedis client."""


class MockRedisClient:
    """Mock for StrictRedis client.

    Example usage in a test:

        @patch("redis.StrictRedis", new=MockRedis)
        def test_xyz():

        or within test function or test method:
        with patch("redis.StrictRedis", new=MockRedis):
            some test steps
    """

    def __init__(self, **kwargs):
        """Initialize simple dict used instead of Redis storage."""
        self.kwargs = kwargs
        print(kwargs)
        self.cache = {}

    def config_set(self, parameter, value):
        """Allow passing any parameter."""

    def get(self, key):
        """Return item from cache (implementation of GET command)."""
        # real Redis accepts keys as strings only
        assert isinstance(key, str)

        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, key, value, *args, **kwargs):
        """Set item into the cache (implementation of SET command)."""
        # real Redis accepts keys as strings only
        assert isinstance(key, str)
        # real Redis accepts values of types str, bytes, int, or float
        # (no other types are allowed!)
        assert isinstance(value, (str, bytes, int, float))

        self.cache[key] = value
