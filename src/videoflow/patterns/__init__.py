"""Pattern catalog — the unifying modulation vocabulary across haptics,
stim, multi-axis, and edit consumers.

This is the Python source of truth for `pattern_id` per
`forge-ui-design/iterations/09-fixes-and-forge-streamer/.../implementation_handoff.md` §1
and `architecture_feel.md` §3 / §4.5. The same catalog shape ships to
React UIs via the `videoflow patterns-list` CLI command (JSON to stdout).

v1 scope: data only (id, label, color, category, consumers, params, etc.).
The closed-form sampler (`patternSamples` in `patterns-data.js`) is not
yet ported — add a `sample(t, params) -> float` method when downstream
needs live preview generation.
"""

from videoflow.patterns.catalog import (
    CATALOG,
    CONSUMERS,
    NONE_ID,
    VALID_PATTERN_IDS,
    Pattern,
    PatternConsumer,
    PatternParam,
    get,
    is_valid,
    to_json,
)

__all__ = [
    "CATALOG",
    "CONSUMERS",
    "NONE_ID",
    "VALID_PATTERN_IDS",
    "Pattern",
    "PatternConsumer",
    "PatternParam",
    "get",
    "is_valid",
    "to_json",
]
