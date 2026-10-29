"""Exact parameters of the float-grid fit and of the membership likelihoods."""

MIN_GAP_ULPS = 4.0
TOL_ULPS = 1.5
REL_BUDGET = 2e-6
REL_FLOOR = 1e-9
REL_K = 6.0
MIN_STEP_ULPS = 8.0
MIN_FIT_FRACTION = 0.90
MAX_CHANCE_RATE = 0.30
MIN_FIT_GAPS = 30
MAX_SUBMULTIPLE = 6
LADDER = (4, 40, 400, 4000, 40000)
LOOSE = 0.2
MIN_CONSISTENT = 20
MIN_LADDER_SELECTED = 5
OFF_RATE_FLOOR = 2e-3
LAM_FLOOR = 1e-12
R_CAP = 1e12
R_FLOOR = 1e-300
MAD_NORMAL = 1.4826
MEDIAN_SE = 1.2533
POW2 = tuple(2.0**k for k in range(-126, 128))
FLOAT32_MAX = 3.4028234663852886e38
FLOAT32_TINY = 2.0**-126
FLOAT32_SUBNORMAL = 2.0**-149
TWO52 = 4503599627370496.0
BAND_LIMITS = (float("inf"), 1.0, 0.5, 0.25)
BAND_MIN_VALUES = 64
BAND_MAX_SUBMULTIPLE = 40
BAND_MIN_SELECTED = 3
BAND_MAX_RUNG = 4_000_000
BAND_OFF_RATE_FLOOR = 0.025

CONFIG_NAMES = (
    "BAND_LIMITS",
    "BAND_MIN_VALUES",
    "BAND_MAX_SUBMULTIPLE",
    "BAND_MIN_SELECTED",
    "BAND_MAX_RUNG",
    "BAND_OFF_RATE_FLOOR",
    "MIN_GAP_ULPS",
    "TOL_ULPS",
    "REL_BUDGET",
    "REL_FLOOR",
    "REL_K",
    "MIN_STEP_ULPS",
    "MIN_FIT_FRACTION",
    "MAX_CHANCE_RATE",
    "MIN_FIT_GAPS",
    "MAX_SUBMULTIPLE",
    "LADDER",
    "LOOSE",
    "MIN_CONSISTENT",
    "MIN_LADDER_SELECTED",
    "OFF_RATE_FLOOR",
    "LAM_FLOOR",
    "R_CAP",
    "R_FLOOR",
    "MAD_NORMAL",
    "MEDIAN_SE",
)


def constants_hex():
    """Return a fresh configuration, including integer parameters as float hex."""
    values = globals()
    return {
        name: [float(x).hex() for x in values[name]]
        if name in ("LADDER", "BAND_LIMITS") else float(values[name]).hex()
        for name in CONFIG_NAMES
    }
