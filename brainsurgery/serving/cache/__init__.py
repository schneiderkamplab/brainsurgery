from .base import KVCache, CacheEntry
from .paged import TorchPagedKVCache
from .mlx_paged import MLXPagedKVCache
from .tinygrad_paged import TinygradPagedKVCache

PagedKVCache = TorchPagedKVCache

__all__ = [
    "KVCache", "CacheEntry",
    "PagedKVCache", "TorchPagedKVCache", "MLXPagedKVCache", "TinygradPagedKVCache",
]
