from collections import OrderedDict


# TODO
# this is not thread safe
# this does not scale
# this does not work in a distributed system
# this is a placeholder to be replaced with something like memcached
class LRUCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size


    # Get the value from the cache.
    # Returns the value if the key exists, empty string if not
    def get(self, key):
        if key in self.cache:
            # Move the accessed key to the end to mark it as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return ""

    # Adds a new key-value pair to the cache.  If the key already exists the new value will be appended to the existing value
    def upsert(self, key, value):
        oldVal=self.get(key)
        if oldVal:
            # TODO limit length of cached value
            self.cache[key]=oldVal+"\n"+value
        else:
            # Make room for a new entry if needed
            if len(self.cache) >= self.max_size:
                # Remove the oldest entry if the cache is full
                self.cache.popitem(last=False)
            self.cache[key] = value
        # mark the entry as recently used
        self.cache.move_to_end(key)
