import redis

class AliasService:
    def __init__(self, host='localhost', port=6379, db=0, logger=None):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.logger = logger

    def alias(self, key, value):
        if self.redis.set(key, value):
            return
        self.redis.delete(key)
        if not self.redis.set(key, value) and logger is not None:
            self.logger.warning('Failed to alias %s to value %s', key, value)

    def dealias(self, key):
        return self.redis.get(key)
