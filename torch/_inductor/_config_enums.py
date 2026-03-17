import enum


class PreGradPassTiming(str, enum.Enum):
    # Run late if the custom pass supports a UUID (cacheable), early otherwise.
    DEFAULT = "default"
    # Run after cache lookup, only on cache miss.
    LATE = "late"
    # Run before cache lookup so the cache key reflects the
    # already-transformed graph and passes always execute.
    EARLY = "early"
