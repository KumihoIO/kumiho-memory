import fnmatch


class FakeRedis:
    def __init__(self):
        self.storage = {}
        self.ttl_store = {}

    async def rpush(self, key, value):
        self.storage.setdefault(key, []).append(value)

    async def expire(self, key, ttl):
        self.ttl_store[key] = int(ttl)

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

    async def scan(self, cursor, match, count=100):
        keys = [k for k in self.storage.keys() if fnmatch.fnmatch(k, match)]
        return 0, keys

    async def incr(self, key):
        current = int(self.storage.get(key, 0))
        current += 1
        self.storage[key] = current
        return current

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
