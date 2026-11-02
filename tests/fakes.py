import fnmatch


class FakeRedis:
    def __init__(self):
        self.storage = {}
        self.ttl_store = {}
        self._clock = 0
        self._deadlines = {}

    async def rpush(self, key, value):
        # Real RPUSH returns the post-push list length. The fake used to
        # return None, which hid that add_message's created_bucket must come
        # from this atomic return rather than a racy follow-up LLEN.
        items = self.storage.setdefault(key, [])
        items.append(value)
        return len(items)

    async def expire(self, key, ttl):
        self.ttl_store[key] = int(ttl)
        self._deadlines[key] = self._clock + int(ttl)

    def advance(self, seconds):
        """Move the fake clock on and drop every key whose TTL has elapsed.

        What a real Redis does by itself while a session sits idle: the only
        way a test can simulate idle-TTL expiry of a buffer without sleeping.
        ``ttl_store`` keeps reporting the last TTL that was *set* (existing
        tests read it that way); the clock is separate bookkeeping.
        """
        self._clock += seconds
        for key, deadline in list(self._deadlines.items()):
            if deadline <= self._clock:
                self.storage.pop(key, None)
                self.ttl_store.pop(key, None)
                del self._deadlines[key]

    async def llen(self, key):
        return len(self.storage.get(key, []))

    async def lrange(self, key, start, end):
        items = list(self.storage.get(key, []))
        if not items:
            return []
        start = self._normalize_index(start, len(items))
        end = self._normalize_index(end, len(items))
        if end < start:
            return []
        return items[start : end + 1]

    async def ttl(self, key):
        return self.ttl_store.get(key, -1)

    async def delete(self, key):
        self.storage.pop(key, None)
        self.ttl_store.pop(key, None)
        self._deadlines.pop(key, None)

    async def scan(self, cursor, match, count=100):
        keys = [k for k in self.storage.keys() if fnmatch.fnmatch(k, match)]
        return 0, keys

    async def incr(self, key):
        current = int(self.storage.get(key, 0))
        current += 1
        self.storage[key] = current
        return current

    async def hset(self, key, mapping):
        self.storage.setdefault(key, {}).update(mapping)

    async def hgetall(self, key):
        return dict(self.storage.get(key, {}))

    async def close(self):
        return None

    @staticmethod
    def _normalize_index(index, length):
        if index < 0:
            index = length + index
        if index < 0:
            return 0
        if index >= length:
            return length - 1
        return index
