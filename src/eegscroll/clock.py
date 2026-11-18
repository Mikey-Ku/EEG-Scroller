"""One shared clock for every stream."""

import time

# LSL clock, same one muselsl stamps samples with
from pylsl import local_clock as now


def unix_to_clock_offset() -> float:
    """Add this to Unix seconds to get clock seconds."""
    # Browser events arrive in Unix time
    return now() - time.time()

