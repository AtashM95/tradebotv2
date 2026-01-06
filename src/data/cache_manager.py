
class CacheManager:
    def __init__(self) -> None:
        self.cache = {}

    def set(self, key: str, value) -> None:
        self.cache[key] = value

    def get(self, key: str, default=None):
        return self.cache.get(key, default)
