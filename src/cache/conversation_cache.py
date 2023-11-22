from collections import OrderedDict


class LRUCache:
    """
    A simple Least Recently Used (LRU) cache implementation using OrderedDict.

    Note:
    - This implementation is not thread-safe.
    - It does not scale well for large datasets or in a distributed system.
    - It's a basic placeholder and may need to be replaced with a more robust solution (e.g., memcached) in production.
    """

    def __init__(self, max_size):
        """
        Initialize the LRUCache.

        Args:
        - max_size (int): The maximum size of the cache.
        """
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """
        Get the value associated with the given key from the cache.

        Args:
        - key: The key to retrieve the value for.

        Returns:
        - str: The value associated with the key if it exists, an empty string otherwise.
        """
        if key in self.cache:
            # Move the accessed key to the end to mark it as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return ""

    def upsert(self, key, value):
        """
        Add or update a key-value pair in the cache.

        Args:
        - key: The key to add or update.
        - value: The value associated with the key.

        Returns:
        - None
        """
        old_val = self.get(key)
        if old_val:
            # TODO: Limit the length of cached value if needed
            self.cache[key] = new_value = "\n".join([old_val, value])
        else:
            # Make room for a new entry if needed
            if len(self.cache) >= self.max_size:
                # Remove the oldest entry if the cache is full
                self.cache.popitem(last=False)
            self.cache[key] = value
        # Mark the entry as recently used
        self.cache.move_to_end(key)
