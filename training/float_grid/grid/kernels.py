"""Scalar arithmetic shared by fitting, streaming, batch, and diagnostics."""

import math
from time import perf_counter

import numpy as np
from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from .constants import (
    BAND_LIMITS,
    BAND_MAX_RUNG,
    BAND_MAX_SUBMULTIPLE,
    BAND_MIN_SELECTED,
    BAND_MIN_VALUES,
    BAND_OFF_RATE_FLOOR,
    FLOAT32_MAX,
    FLOAT32_SUBNORMAL,
    FLOAT32_TINY,
    LADDER,
    LAM_FLOOR,
    LOOSE,
    MAD_NORMAL,
    MAX_CHANCE_RATE,
    MAX_SUBMULTIPLE,
    MEDIAN_SE,
    MIN_CONSISTENT,
    MIN_FIT_FRACTION,
    MIN_FIT_GAPS,
    MIN_GAP_ULPS,
    MIN_LADDER_SELECTED,
    MIN_STEP_ULPS,
    OFF_RATE_FLOOR,
    POW2,
    R_CAP,
    R_FLOOR,
    REL_BUDGET,
    REL_FLOOR,
    REL_K,
    TOL_ULPS,
    TWO52,
)
from .runtime import compiled

_START = perf_counter()
F = types.float64
INT = types.int64
A = types.Array(F, 1, "C")
B = types.Array(F, 2, "C")

# Parameters are owned by one series. Counts are exactly representable here.
DETECTED, STEP, REL, FRACTION, CHANCE, OFF_RATE, COUNT, GAP_COUNT, ACCUMULATOR, REASON, UMED, OK_COUNT = range(12)
MODE, BAND_LIMIT, BAND_K = range(12, 15)
PARAM_COUNT = 15
REASONS = ("detected", "nonfinite_history", "few_unique_values", "few_fit_gaps", "no_valid_candidate", "not_detected")
BAND_VALUE_COUNT, BAND_STATUS = PARAM_COUNT, PARAM_COUNT + 1
BAND_COLUMNS = PARAM_COUNT + 2


@intrinsic
def ieee_sqrt(typing_context, value):
    """Emit the IEEE instruction directly, including on targets with an external sqrt lowering."""
    if value != types.float64:
        return None
    signature = types.float64(types.float64)

    def emit(context, builder, sig, arguments):
        function = builder.module.declare_intrinsic("llvm.sqrt", [ir.DoubleType()])
        return builder.call(function, arguments)

    return signature, emit


@compiled(F(F))
def rint(x):
    ax = abs(x)
    if ax < TWO52:
        return math.copysign((ax + TWO52) - TWO52, x)
    return x


@compiled(F(F))
def ulp32(x):
    a = abs(x)
    if a < FLOAT32_TINY:
        return FLOAT32_SUBNORMAL
    if a == FLOAT32_MAX or a == math.inf:
        return math.inf
    if a != a:
        return a
    lo, hi = 0, len(POW2)
    while lo < hi:
        mid = (lo + hi) // 2
        if POW2[mid] <= a:
            lo = mid + 1
        else:
            hi = mid
    return POW2[lo - 1] * (1.0 / 8388608.0)


@compiled(F(A))
def fsum(values):
    """Partial expansion with the final halfway correction used by math.fsum."""
    partials = np.empty(len(values), np.float64)
    n = 0
    special = 0.0
    infinity_sum = 0.0
    for original in values:
        x = original
        i = 0
        for j in range(n):
            y = partials[j]
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                partials[i] = lo
                i += 1
            x = hi
        n = i
        if x != 0.0:
            if not math.isfinite(x):
                if math.isfinite(original):
                    raise OverflowError("intermediate overflow in fsum")
                if math.isinf(original):
                    infinity_sum += original
                special += original
                n = 0
            else:
                partials[n] = x
                n += 1
    if special != 0.0:
        if infinity_sum != infinity_sum:
            raise ValueError("opposite infinities in fsum")
        return special
    hi = 0.0
    lo = 0.0
    if n > 0:
        n -= 1
        hi = partials[n]
        while n > 0:
            x = hi
            n -= 1
            y = partials[n]
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            if y == yr:
                hi = x
    return hi


@compiled(F(A))
def median(values):
    n = len(values)
    if n == 0:
        return math.nan
    ordered = np.sort(values)
    k = n // 2
    if n % 2:
        return ordered[k]
    return (ordered[k - 1] + ordered[k]) / 2.0


@compiled(INT(INT))
def quantile_index(n):
    return ((n - 1) * 3) // 4


@compiled(F(A, F))
def ladder(gaps, d):
    selected = np.empty(len(gaps), np.float64)
    for limit in LADDER:
        n = 0
        for gap in gaps:
            ratio = gap / d
            m = rint(ratio)
            if m >= 1 and m <= limit and abs(ratio - m) <= LOOSE:
                selected[n] = gap / m
                n += 1
        if n < MIN_LADDER_SELECTED:
            break
        d = median(selected[:n])
    return d


@compiled(types.UniTuple(F, 3)(A, A, F))
def score_fit(gaps, spacing, d):
    chances = np.empty(len(gaps), np.float64)
    count = 0
    for j in range(len(gaps)):
        m = rint(gaps[j] / d)
        tol = (TOL_ULPS * spacing[j]) + ((m * d) * REL_BUDGET)
        if m >= 1 and abs(gaps[j] - m * d) <= tol:
            count += 1
        chances[j] = min(1.0, (2.0 * tol) / d)
    n = len(gaps)
    return count / n, fsum(chances) / n, float(count)


@compiled(types.UniTuple(F, 2)(A, F))
def refine_span(support, d):
    center = support[len(support) // 2]
    spans = np.empty(len(support), np.float64)
    multiples = np.empty(len(support), np.float64)
    magnitudes = np.empty(len(support), np.float64)
    count = 0
    for value in support:
        delta = value - center
        m = rint(delta / d)
        if abs(m) >= 1 and abs(delta - m * d) <= LOOSE * d:
            spans[count] = delta
            multiples[count] = m
            magnitudes[count] = abs(m)
            count += 1
    if count < MIN_CONSISTENT:
        return d, REL_BUDGET
    ordered = np.sort(magnitudes[:count])
    threshold = ordered[quantile_index(count)]
    estimates = np.empty(count, np.float64)
    n = 0
    for j in range(count):
        if magnitudes[j] >= threshold:
            estimates[n] = spans[j] / multiples[j]
            n += 1
    d2 = median(estimates[:n])
    for j in range(n):
        estimates[j] = abs(estimates[j] - d2)
    mad = median(estimates[:n])
    rel = ((((REL_K * MEDIAN_SE) * MAD_NORMAL) * mad) / ieee_sqrt(float(n))) / d2
    return d2, rel if rel > REL_FLOOR else REL_FLOOR


@compiled(types.void(A, A, A, A))
def fit(history, support, support_ulp, params):
    for j in range(PARAM_COUNT):
        params[j] = math.nan
    params[DETECTED] = 0.0
    params[COUNT] = 0.0
    params[GAP_COUNT] = 0.0
    params[OK_COUNT] = 0.0
    params[ACCUMULATOR] = 0.0
    params[OFF_RATE] = OFF_RATE_FLOOR
    for value in history:
        if not math.isfinite(value):
            params[REASON] = 1.0
            return
    ordered = np.sort(history)
    n = 0
    for value in ordered:
        if n == 0 or value != support[n - 1]:
            support[n] = 0.0 if value == 0.0 else value
            support_ulp[n] = ulp32(value)
            n += 1
    params[COUNT] = float(n)
    if n < MIN_FIT_GAPS + 1:
        params[REASON] = 2.0
        return
    gaps = np.empty(n - 1, np.float64)
    spacing = np.empty(n - 1, np.float64)
    count = 0
    smallest = math.inf
    for j in range(n - 1):
        gap = support[j + 1] - support[j]
        ul = max(support_ulp[j], support_ulp[j + 1])
        if gap > MIN_GAP_ULPS * ul:
            gaps[count] = gap
            spacing[count] = ul
            count += 1
            if gap < smallest:
                smallest = gap
    params[GAP_COUNT] = float(count)
    if count < MIN_FIT_GAPS:
        params[REASON] = 3.0
        return
    ge, ue = gaps[:count], spacing[:count]
    umed = median(ue)
    params[UMED] = umed
    best_d, best_fraction = math.nan, -1.0
    for k in range(1, MAX_SUBMULTIPLE + 1):
        d = ladder(ge, smallest / k)
        if not (d > 0.0 and math.isfinite(d)):
            continue
        fraction, chance, _ = score_fit(ge, ue, d)
        if fraction > best_fraction:
            best_d, best_fraction = d, fraction
        if fraction >= MIN_FIT_FRACTION and chance <= MAX_CHANCE_RATE and d >= MIN_STEP_ULPS * umed:
            best_d = d
            break
    if best_fraction < 0:
        params[REASON] = 4.0
        return
    d, rel = refine_span(support[:n], best_d)
    d, rel = refine_span(support[:n], d)
    fraction, chance, ok_count = score_fit(ge, ue, d)
    detected = fraction >= MIN_FIT_FRACTION and chance <= MAX_CHANCE_RATE and d >= MIN_STEP_ULPS * umed
    params[DETECTED] = 1.0 if detected else 0.0
    params[STEP], params[REL] = d, rel
    params[FRACTION], params[CHANCE] = fraction, chance
    params[OFF_RATE] = max(OFF_RATE_FLOOR, (1.0 - fraction) / 2.0)
    params[REASON] = 0.0 if detected else 5.0
    params[OK_COUNT] = ok_count


@compiled(F(A, INT, INT))
def select_nth(values, n, rank):
    """Exact three-way selection; only the first n scratch entries are changed."""
    left, right = 0, n - 1
    while left < right:
        a, b, c = values[left], values[(left + right) // 2], values[right]
        if a > b:
            a, b = b, a
        if b > c:
            b = c
        pivot = a if a > b else b
        lower, cursor, upper = left, left, right
        while cursor <= upper:
            value = values[cursor]
            if value < pivot:
                values[lower], values[cursor] = value, values[lower]
                lower += 1
                cursor += 1
            elif value > pivot:
                values[upper], values[cursor] = value, values[upper]
                upper -= 1
            else:
                cursor += 1
        if rank < lower:
            right = lower - 1
        elif rank > upper:
            left = upper + 1
        else:
            return pivot
    return values[left]


@compiled(F(A, INT))
def median_inplace(values, n):
    """The sorted central value or the exact (lower + upper) / 2.0."""
    if n == 0:
        return math.nan
    k = n // 2
    upper = select_nth(values, n, k)
    if n % 2:
        return upper
    lower = values[0]
    for j in range(1, k):
        if values[j] > lower:
            lower = values[j]
    return (lower + upper) / 2.0


@compiled(types.UniTuple(F, 2)(A, F, B))
def refine_span_buffered(support, d, work):
    """Same span arithmetic as refine_span, with caller-owned scratch space."""
    center = support[len(support) // 2]
    spans, multiples, magnitudes, estimates = work[3], work[4], work[5], work[6]
    scratch = work[2]
    count = 0
    for value in support:
        delta = value - center
        m = rint(delta / d)
        if abs(m) >= 1 and abs(delta - m * d) <= LOOSE * d:
            spans[count] = delta
            multiples[count] = m
            magnitudes[count] = abs(m)
            scratch[count] = abs(m)
            count += 1
    if count < MIN_CONSISTENT:
        return d, REL_BUDGET
    threshold = select_nth(scratch, count, quantile_index(count))
    n = 0
    for j in range(count):
        if magnitudes[j] >= threshold:
            estimates[n] = spans[j] / multiples[j]
            n += 1
    d2 = median_inplace(estimates, n)
    for j in range(n):
        estimates[j] = abs(estimates[j] - d2)
    mad = median_inplace(estimates, n)
    rel = ((((REL_K * MEDIAN_SE) * MAD_NORMAL) * mad) / ieee_sqrt(float(n))) / d2
    return d2, rel if rel > REL_FLOOR else REL_FLOOR


@compiled(F(A, INT, A))
def fsum_buffered(values, count, partials):
    """Exact partial expansion for the finite nonnegative fit probabilities."""
    n = 0
    for j in range(count):
        x = values[j]
        i = 0
        for p in range(n):
            y = partials[p]
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                partials[i] = lo
                i += 1
            x = hi
        n = i
        if x != 0.0:
            partials[n] = x
            n += 1
    hi, lo = 0.0, 0.0
    if n > 0:
        n -= 1
        hi = partials[n]
        while n > 0:
            x = hi
            n -= 1
            y = partials[n]
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            if y == yr:
                hi = x
    return hi


@compiled(types.UniTuple(F, 3)(A, A, INT, F, B))
def score_fit_buffered(gaps, spacing, count, d, work):
    chances, partials = work[2], work[7]
    ok_count = 0
    for j in range(count):
        m = rint(gaps[j] / d)
        tol = (TOL_ULPS * spacing[j]) + ((m * d) * REL_BUDGET)
        if m >= 1 and abs(gaps[j] - m * d) <= tol:
            ok_count += 1
        chances[j] = min(1.0, (2.0 * tol) / d)
    return ok_count / count, fsum_buffered(chances, count, partials) / count, float(ok_count)


@compiled(types.void(A, A, A, A, B))
def fit_band(history, support, support_ulp, params, work):
    """Doubling-rung fit with exact selection and no allocations in the candidate loop."""
    for j in range(PARAM_COUNT):
        params[j] = math.nan
    params[DETECTED], params[COUNT], params[GAP_COUNT], params[OK_COUNT] = 0.0, 0.0, 0.0, 0.0
    params[ACCUMULATOR], params[MODE], params[BAND_K] = 0.0, 0.0, 0.0
    params[OFF_RATE] = BAND_OFF_RATE_FLOOR
    for value in history:
        if not math.isfinite(value):
            params[REASON] = 1.0
            return
    ordered = np.sort(history)
    n = 0
    for value in ordered:
        if n == 0 or value != support[n - 1]:
            support[n] = 0.0 if value == 0.0 else value
            support_ulp[n] = ulp32(value)
            n += 1
    params[COUNT] = float(n)
    if n < MIN_FIT_GAPS + 1:
        params[REASON] = 2.0
        return
    gaps, spacing, selected = work[0], work[1], work[2]
    count = 0
    smallest = math.inf
    for j in range(n - 1):
        gap = support[j + 1] - support[j]
        ul = max(support_ulp[j], support_ulp[j + 1])
        if gap > MIN_GAP_ULPS * ul:
            gaps[count], spacing[count], selected[count] = gap, ul, ul
            count += 1
            if gap < smallest:
                smallest = gap
    params[GAP_COUNT] = float(count)
    if count < MIN_FIT_GAPS:
        params[REASON] = 3.0
        return
    umed = median_inplace(selected, count)
    params[UMED] = umed
    best_fraction, best_chance = -1.0, 0.0
    for k in range(1, BAND_MAX_SUBMULTIPLE + 1):
        d = smallest / k
        limit = max(4, 2 * k)
        while limit <= BAND_MAX_RUNG:
            chosen = 0
            for j in range(count):
                ratio = gaps[j] / d
                m = rint(ratio)
                if m >= 1 and m <= limit and abs(ratio - m) <= LOOSE:
                    selected[chosen] = gaps[j] / m
                    chosen += 1
            if chosen < BAND_MIN_SELECTED:
                break
            d = median_inplace(selected, chosen)
            limit *= 2
        d, rel = refine_span_buffered(support[:n], d, work)
        d, rel = refine_span_buffered(support[:n], d, work)
        fraction, chance, ok_count = score_fit_buffered(gaps, spacing, count, d, work)
        good = fraction >= MIN_FIT_FRACTION and chance <= MAX_CHANCE_RATE and d >= MIN_STEP_ULPS * umed
        if k == 1 or fraction - chance > best_fraction - best_chance or good:
            best_fraction, best_chance = fraction, chance
            params[STEP], params[REL] = d, rel
            params[FRACTION], params[CHANCE], params[OK_COUNT] = fraction, chance, ok_count
            params[DETECTED], params[BAND_K] = 1.0 if good else 0.0, float(k)
        if good:
            break
    params[REASON] = 0.0 if params[DETECTED] else 5.0
    params[OFF_RATE] = max(BAND_OFF_RATE_FLOOR, (1.0 - params[FRACTION]) / 2.0)


@compiled(types.void(A, A, A, A, B))
def fit_cascade(history, support, support_ulp, params, bands):
    """Preserve the original fit, then retain the first successful absolute band."""
    fit(history, support, support_ulp, params)
    params[MODE] = params[DETECTED]
    params[BAND_LIMIT], params[BAND_K] = math.nan, 0.0
    for i in range(len(BAND_LIMITS)):
        for j in range(BAND_COLUMNS):
            bands[i, j] = math.nan
    if params[DETECTED] or params[REASON] == 1.0:
        return
    n = len(history)
    values = np.empty(n, np.float64)
    band_support = np.empty(n, np.float64)
    band_ulp = np.empty(n, np.float64)
    candidate = np.empty(PARAM_COUNT, np.float64)
    work = np.empty((8, n), np.float64)
    for i in range(len(BAND_LIMITS)):
        limit = BAND_LIMITS[i]
        count = 0
        for value in history:
            if abs(value) < limit:
                values[count] = value
                count += 1
        bands[i, BAND_LIMIT], bands[i, BAND_VALUE_COUNT] = limit, float(count)
        bands[i, BAND_STATUS] = 0.0
        if count < BAND_MIN_VALUES:
            continue
        fit_band(values[:count], band_support, band_ulp, candidate, work)
        candidate[BAND_LIMIT] = limit
        for j in range(PARAM_COUNT):
            bands[i, j] = candidate[j]
        bands[i, BAND_STATUS] = 1.0
        if candidate[DETECTED]:
            for j in range(PARAM_COUNT):
                params[j] = candidate[j]
            params[MODE] = 2.0
            for j in range(int(params[COUNT])):
                support[j], support_ulp[j] = band_support[j], band_ulp[j]
            return


@compiled(types.UniTuple(F, 5)(F, A, A, A))
def membership_point(x, support, support_ulp, params):
    if not math.isfinite(x):
        return 0.0, 1.0, math.nan, math.nan, math.nan
    n = int(params[COUNT])
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if support[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    left, right = min(max(lo - 1, 0), n - 1), min(lo, n - 1)
    j = left if abs(x - support[left]) <= abs(support[right] - x) else right
    delta = x - support[j]
    d = params[STEP]
    ratio = delta / d
    m = rint(ratio)
    tol = (TOL_ULPS * max(ulp32(x), support_ulp[j])) + ((abs(m) * d) * params[REL])
    residual = abs(delta - m * d)
    on = residual <= tol
    q = min(1.0, (2.0 * tol) / d)
    return (1.0 if on else 0.0), q, tol - residual, tol, 0.5 - abs(ratio - m)


@compiled(types.UniTuple(F, 9)(F, INT, A, A, A))
def step(x, t, support, support_ulp, params):
    if params[DETECTED] == 0.0:
        return 0.0, 0.0, 1.0, 1.0, 0.0, math.nan, math.nan, math.nan, math.nan
    in_band = params[MODE] != 2.0 or abs(x) < params[BAND_LIMIT]
    if in_band:
        on, q, margin, tol, midpoint = membership_point(x, support, support_ulp, params)
    else:
        on, q, margin, tol, midpoint = 0.0, 1.0, math.nan, math.nan, math.nan
    if not math.isfinite(x) or not in_band:
        lam = 1.0
    elif on != 0.0:
        lam = q / (1.0 - params[OFF_RATE])
    else:
        w = 1.0 - q
        lam = (w if w > LAM_FLOOR else LAM_FLOOR) / params[OFF_RATE]
    r = (1.0 + params[ACCUMULATOR]) * lam
    if r > R_CAP:
        r = R_CAP
    if r < R_FLOOR:
        r = R_FLOOR
    params[ACCUMULATOR] = r
    argument = r / (float(t) + 1.0)
    return math.log(argument), on, q, lam, r, argument, margin, tol, midpoint


@compiled(types.void(A, A, A, A, A))
def batch_steps(online, support, support_ulp, params, output):
    for t in range(len(online)):
        output[t] = step(online[t], t, support, support_ulp, params)[0]


@compiled(types.void(A, A, A, A, B))
def diagnostic_steps(online, support, support_ulp, params, output):
    for t in range(len(online)):
        result = step(online[t], t, support, support_ulp, params)
        for j in range(9):
            output[t, j] = result[j]


COMPILATION_SECONDS = perf_counter() - _START
