#Only needs to be done once
from gutenberg.acquire import get_metadata_cache
cache = get_metadata_cache()
cache.populate()
