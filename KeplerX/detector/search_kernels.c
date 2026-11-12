// Exact kernels for the Detector search, bitwise as upstream; no fast-math or contraction (fma() only in fold).

#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// np.mod(t, P) for t >= 0, P > 0: one exact fma once k = floor(t/P) is right; k only steps up.
static inline double fold(double t, double period, double *cycle) {
    double k = *cycle;
    double r = fma(-k, period, t);
    while (r >= period) {
        k += 1.0;
        r = fma(-k, period, t);
    }
    while (r < 0.0) {
        k -= 1.0;
        r = fma(-k, period, t);
    }
    *cycle = k;
    return r;
}

// fast_histogram 0.14's bin index; an index of nx moves to the last bin.
static inline int32_t bin_of(double x, double scale, int32_t nbins) {
    int32_t ix = (int32_t)(x * scale);
    return ix - (ix == nbins);
}

// Search_for_TCEs_in_all_Tdur_models per period: ses.calc_mes histograms per duration, NaN-propagating first argmax.
#define MAX_BLOCK 64
#define CHUNK 512

// plain: finite inputs, so only bins passing an exact squared screen get num / sqrt(den), in bin order.
static void evaluate(int32_t m, int32_t width, int32_t nb, const double *sums, int32_t lo,
                     double n_trans, double threshold, int plain, double *out_mes,
                     int32_t *out_dur, int32_t *out_bin) {
    double max_mes = threshold;
    int32_t best_dur = -1, best_bin = -1;
    for (int32_t j = 0; j < m; j++) {
        if (plain) {
            double bar = max_mes, limit = bar * bar * (1.0 - 1e-12), cell_max = 0.0;
            int32_t arg = -1;
            for (int32_t b = 0; b < nb; b++) {
                const double *cell = sums + (int64_t)b * width;
                double numerator = cell[1 + 2 * j];
                if (cell[0] < n_trans || !(numerator > 0.0)) continue;
                double denominator = cell[2 + 2 * j];
                if (!(numerator * numerator > limit * denominator)) continue;
                double mes = numerator / sqrt(denominator);
                if (mes > bar) {
                    bar = cell_max = mes;
                    limit = bar * bar * (1.0 - 1e-12);
                    arg = b;
                }
            }
            if (arg >= 0) {
                max_mes = cell_max;
                best_dur = lo + j;
                best_bin = arg;
            }
            continue;
        }
        double cell_max = 0.0;
        int32_t arg = 0, nan = 0;
        for (int32_t b = 0; b < nb; b++) {
            const double *cell = sums + (int64_t)b * width;
            double numerator = 0.0, denominator = 1.0;
            if (!(cell[0] < n_trans)) {
                numerator = cell[1 + 2 * j];
                denominator = cell[2 + 2 * j];
            }
            double mes = numerator / sqrt(denominator);
            if (isnan(mes)) {
                nan = 1;
                break;
            }
            if (b == 0 || mes > cell_max) {
                cell_max = mes;
                arg = b;
            }
        }
        if (!nan && cell_max > max_mes) {
            max_mes = cell_max;
            best_dur = lo + j;
            best_bin = arg;
        }
    }
    *out_mes = max_mes;
    *out_dur = best_dur;
    *out_bin = best_bin;
}

// One period over cadences [first, last), each bin accumulating in time order as fast_histogram does.
#define ACCUMULATE_BODY(WIDTH)                                                         \
    double k = *cycle;                                                                 \
    for (int64_t i = first; i < last; i++) {                                           \
        double ti = t[i];                                                              \
        double r = fma(-k, period, ti);                                                \
        while (r >= period) {                                                          \
            k += 1.0;                                                                  \
            r = fma(-k, period, ti);                                                   \
        }                                                                              \
        while (r < 0.0) {                                                              \
            k -= 1.0;                                                                  \
            r = fma(-k, period, ti);                                                   \
        }                                                                              \
        int32_t ix = bin_of(r, scale, nb);                                             \
        double *restrict cell = sums + (int64_t)ix * (WIDTH);                          \
        const double *restrict row = rows + i * (WIDTH);                               \
        for (int32_t j = 0; j < (WIDTH); j++) cell[j] += row[j];                       \
    }                                                                                  \
    *cycle = k;

typedef void (*accumulator)(int64_t, int64_t, const double *restrict, const double *restrict,
                            double, double, int32_t, double *restrict, double *restrict,
                            int32_t);

#define DEFINE_ACCUMULATE(WIDTH)                                                       \
    static void accumulate_##WIDTH(int64_t first, int64_t last, const double *restrict t,  \
                                   const double *restrict rows, double period,         \
                                   double scale, int32_t nb, double *restrict cycle,   \
                                   double *restrict sums, int32_t width) {             \
        (void)width;                                                                   \
        ACCUMULATE_BODY(WIDTH)                                                         \
    }

DEFINE_ACCUMULATE(3)
DEFINE_ACCUMULATE(5)
DEFINE_ACCUMULATE(7)
DEFINE_ACCUMULATE(9)
DEFINE_ACCUMULATE(11)
DEFINE_ACCUMULATE(13)
DEFINE_ACCUMULATE(15)
DEFINE_ACCUMULATE(17)
DEFINE_ACCUMULATE(19)
DEFINE_ACCUMULATE(21)
DEFINE_ACCUMULATE(23)
DEFINE_ACCUMULATE(25)
DEFINE_ACCUMULATE(27)
DEFINE_ACCUMULATE(29)
DEFINE_ACCUMULATE(31)

static void accumulate_any(int64_t first, int64_t last, const double *restrict t,
                           const double *restrict rows, double period, double scale,
                           int32_t nb, double *restrict cycle, double *restrict sums,
                           int32_t width) {
    ACCUMULATE_BODY(width)
}

static accumulator choose(int32_t width) {
    switch (width) {
    case 3: return accumulate_3;
    case 5: return accumulate_5;
    case 7: return accumulate_7;
    case 9: return accumulate_9;
    case 11: return accumulate_11;
    case 13: return accumulate_13;
    case 15: return accumulate_15;
    case 17: return accumulate_17;
    case 19: return accumulate_19;
    case 21: return accumulate_21;
    case 23: return accumulate_23;
    case 25: return accumulate_25;
    case 27: return accumulate_27;
    case 29: return accumulate_29;
    case 31: return accumulate_31;
    default: return accumulate_any;
    }
}

static int search_range(int64_t n, const double *t, int32_t m, const double *rows,
                        int64_t count, const double *periods, const int32_t *nbins,
                        int32_t lo, double n_trans, double threshold, int plain,
                        double *out_mes, int32_t *out_dur, int32_t *out_bin, int64_t budget) {
    int32_t width = 2 * m + 1;
    accumulator add = choose(width);
    for (int64_t start = 0; start < count;) {
        // Consecutive periods share each cache-sized chunk of cadence rows.
        int64_t block = 0, cells = 0;
        while (start + block < count && block < MAX_BLOCK) {
            int64_t need = (int64_t)nbins[start + block] * width;
            if (block > 0 && (cells + need) * 8 > budget) break;
            cells += need;
            block++;
        }
        double *memory = calloc((size_t)cells, sizeof(double));
        if (!memory) return 1;
        double *sums[MAX_BLOCK], scale[MAX_BLOCK], cycle[MAX_BLOCK];
        int64_t offset = 0;
        for (int64_t q = 0; q < block; q++) {
            sums[q] = memory + offset;
            offset += (int64_t)nbins[start + q] * width;
            scale[q] = (double)nbins[start + q] / periods[start + q];
            cycle[q] = 0.0;
        }
        for (int64_t first = 0; first < n; first += CHUNK) {
            int64_t last = first + CHUNK < n ? first + CHUNK : n;
            for (int64_t q = 0; q < block; q++)
                add(first, last, t, rows, periods[start + q], scale[q], nbins[start + q],
                    &cycle[q], sums[q], width);
        }
        for (int64_t q = 0; q < block; q++)
            evaluate(m, width, nbins[start + q], sums[q], lo, n_trans, threshold, plain,
                     out_mes + start + q, out_dur + start + q, out_bin + start + q);
        free(memory);
        start += block;
    }
    return 0;
}

int v3_fold_search(int64_t n, const double *t, int32_t n_dur, const double *num,
                   const double *den, int64_t n_per, const double *periods,
                   const int32_t *nbins, const int32_t *lo, const int32_t *hi,
                   double n_trans, double threshold, int64_t budget, double *out_mes,
                   int32_t *out_dur, int32_t *out_bin) {
    for (int64_t p = 0; p < n_per;) {
        int64_t end = p;
        while (end < n_per && lo[end] == lo[p] && hi[end] == hi[p]) end++;
        int32_t m = hi[p] - lo[p];
        if (m <= 0) {
            for (int64_t q = p; q < end; q++) {
                out_mes[q] = threshold;
                out_dur[q] = -1;
                out_bin[q] = -1;
            }
            p = end;
            continue;
        }
        int64_t width = 2 * m + 1;
        double *rows = malloc((size_t)n * width * sizeof(double));
        if (!rows) return 1;
        for (int64_t i = 0; i < n; i++) {
            rows[i * width] = 1.0;
            for (int32_t j = 0; j < m; j++) {
                rows[i * width + 1 + 2 * j] = num[(int64_t)(lo[p] + j) * n + i];
                rows[i * width + 2 + 2 * j] = den[(int64_t)(lo[p] + j) * n + i];
            }
        }
        int plain = threshold >= 0.0;
        for (int64_t i = 0; i < n && plain; i++)
            for (int32_t j = 0; j < m; j++) {
                double a = rows[i * width + 1 + 2 * j], b = rows[i * width + 2 + 2 * j];
                if (!isfinite(a) || !isfinite(b) || !(b > 0.0)) plain = 0;
            }
        int failed = search_range(n, t, m, rows, end - p, periods + p, nbins + p, lo[p],
                                  n_trans, threshold, plain, out_mes + p, out_dur + p,
                                  out_bin + p, budget);
        free(rows);
        if (failed) return 1;
        p = end;
    }
    return 0;
}

// NumPy's float remainder for |b| < 2P: b, b - P (exact by Sterbenz) or b + P.
static inline double remainder_numpy(double b, double period) {
    if (b >= 0.0 && b < period) return b == 0.0 ? 0.0 : b;
    if (b >= period && b < 2.0 * period) return b - period;
    if (b < 0.0 && b > -period) return b + period;
    double r = fmod(b, period);
    if (r != 0.0) {
        if ((period < 0.0) != (r < 0.0)) r += period;
    } else {
        r = copysign(0.0, period);
    }
    return r;
}

// check_tce_caused_by_single_event's in-transit test.
static inline int in_transit(double t, double period, double peak, double dur) {
    double half = period / 2.0, cycle = floor(t / period);
    double x = fold(t, period, &cycle);
    return fabs(remainder_numpy((x - peak) + half, period) - half) < dur;
}

// Number of in-transit cadences; out_max receives their NaN-propagating maximum SES.
static int64_t transit_scan(int64_t n, const double *t, const double *ses, double period,
                            double peak, double dur, int first, double *out_max) {
    double half = period / 2.0, cycle = 0.0, best = 0.0;
    int64_t count = 0;
    int nan = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = fold(t[i], period, &cycle);
        double shift = remainder_numpy((x - peak) + half, period) - half;
        if (fabs(shift) < dur) {
            if (first) return 1;
            double s = ses[i];
            if (isnan(s))
                nan = 1;
            else if (count == 0 || s > best)
                best = s;
            count++;
        }
    }
    *out_max = nan ? NAN : best;
    return count;
}

// OpenBLAS 0.3.23 ddot_k_ARMV8's order (NumPy's on Apple M1) over [lo, hi) only: skipped products are +-0.

// np.convolve(a, v, "same") at the outputs in need[], each in dot_armv8's order; 1 if a is not finite.
typedef struct {
    const double *x, *y;
    int64_t lo, main, hi, full;
} span;

static span output_span(int64_t n, const double *a, const double *kernel, int64_t k_lo,
                        int64_t k_hi, int64_t o, int finite) {
    int64_t left = n / 2, xs, ys, len;
    if (o <= left) {
        xs = 0;
        ys = left - o;
        len = n - left + o;
    } else {
        xs = o - left;
        ys = 0;
        len = n - (o - left);
    }
    int64_t lo = finite ? k_lo - ys : 0, hi = finite ? k_hi - ys : len;
    if (lo < 0) lo = 0;
    if (hi > len) hi = len;
    if (hi < lo) hi = lo;
    span s = {a + xs, kernel + ys, lo, 0, hi, len & ~(int64_t)3};
    s.main = hi < s.full ? hi : s.full;
    if (s.main < lo) s.main = lo;
    return s;
}

static inline void lanes(const span *s, int64_t from, int64_t to, double *l0, double *l1) {
    for (int64_t m = from; m < to; m++) {
        if (m & 1)
            *l1 = *l1 + s->x[m] * s->y[m];
        else
            *l0 = *l0 + s->x[m] * s->y[m];
    }
}

static inline double finish(const span *s, double l0, double l1) {
    double sum = s->full > 0 ? l0 + l1 : 0.0;
    for (int64_t m = s->lo > s->full ? s->lo : s->full; m < s->hi; m++)
        sum = fma(s->x[m], s->y[m], sum);
    return 0.0 + sum;
}

// Shared middle of 8 or 16 outputs, two per NEON register; vmulq/vaddq round as the scalar loop does.
#if defined(__aarch64__)
#define V3_MIDDLE(WIDTH, PAIRS) \
static int64_t middle##WIDTH(const span *s, int right, int64_t start, int64_t stop, double *l0, \
                       double *l1) { \
    float64x2_t even[PAIRS], odd[PAIRS]; \
    for (int p = 0; p < PAIRS; p++) { \
        even[p] = right ? (float64x2_t){l0[2 * p], l0[2 * p + 1]} \
                        : (float64x2_t){l0[2 * p + 1], l0[2 * p]}; \
        odd[p] = right ? (float64x2_t){l1[2 * p], l1[2 * p + 1]} \
                       : (float64x2_t){l1[2 * p + 1], l1[2 * p]}; \
    } \
    int64_t m = start; \
    if (right) { \
        const double *x = s[0].x, *y = s[0].y; \
        for (; m + 1 < stop; m += 2) { \
            float64x2_t ye = vdupq_n_f64(y[m]), yo = vdupq_n_f64(y[m + 1]); \
            for (int p = 0; p < PAIRS; p++) { \
                even[p] = vaddq_f64(even[p], vmulq_f64(vld1q_f64(x + m + 2 * p), ye)); \
                odd[p] = vaddq_f64(odd[p], vmulq_f64(vld1q_f64(x + m + 1 + 2 * p), yo)); \
            } \
        } \
    } else { \
        const double *x = s[0].x, *y = s[0].y;  /* y = kernel + ys of output 0 */ \
        for (; m + 1 < stop; m += 2) { \
            float64x2_t xe = vdupq_n_f64(x[m]), xo = vdupq_n_f64(x[m + 1]); \
            for (int p = 0; p < PAIRS; p++) { \
                /* (kernel[ys - 2p + m], kernel[ys - 2p - 1 + m]) */ \
                float64x2_t ke = vld1q_f64(y + m - 2 * p - 1); \
                float64x2_t ko = vld1q_f64(y + m - 2 * p); \
                even[p] = vaddq_f64(even[p], vmulq_f64(xe, ke)); \
                odd[p] = vaddq_f64(odd[p], vmulq_f64(xo, ko)); \
            } \
        } \
    } \
    for (int p = 0; p < PAIRS; p++) { \
        l0[2 * p] = right ? vgetq_lane_f64(even[p], 0) : vgetq_lane_f64(even[p], 1); \
        l0[2 * p + 1] = right ? vgetq_lane_f64(even[p], 1) : vgetq_lane_f64(even[p], 0); \
        l1[2 * p] = right ? vgetq_lane_f64(odd[p], 0) : vgetq_lane_f64(odd[p], 1); \
        l1[2 * p + 1] = right ? vgetq_lane_f64(odd[p], 1) : vgetq_lane_f64(odd[p], 0); \
    } \
    return m; \
}
V3_MIDDLE(8, 4)
V3_MIDDLE(16, 8)
#undef V3_MIDDLE

#endif

#if defined(__aarch64__)
// Gapped outputs share one dot operand; each lane keeps its output's even/odd order.
static int64_t middle_gather4(const span *s, int right, int64_t start, int64_t stop,
                              double *l0, double *l1) {
    const double *v0 = right ? s[0].x : s[0].y, *v1 = right ? s[1].x : s[1].y;
    const double *v2 = right ? s[2].x : s[2].y, *v3 = right ? s[3].x : s[3].y;
    const double *common = right ? s[0].y : s[0].x;
    int direct = right ? (v1 == v0 + 1 && v3 == v2 + 1)
                       : (v1 + 1 == v0 && v3 + 1 == v2);
    int flip = direct && !right;
    float64x2_t e0 = vld1q_f64(l0), e1 = vld1q_f64(l0 + 2);
    float64x2_t o0 = vld1q_f64(l1), o1 = vld1q_f64(l1 + 2);
    if (flip) {
        e0 = vextq_f64(e0, e0, 1); e1 = vextq_f64(e1, e1, 1);
        o0 = vextq_f64(o0, o0, 1); o1 = vextq_f64(o1, o1, 1);
    }
    int64_t m = start;
    if (direct) {
        const double *p0 = right ? v0 : v1, *p1 = right ? v2 : v3;
        for (; m + 1 < stop; m += 2) {
            float64x2_t coeff = vld1q_f64(common + m);
            e0 = vaddq_f64(e0, vmulq_laneq_f64(vld1q_f64(p0 + m), coeff, 0));
            o0 = vaddq_f64(o0, vmulq_laneq_f64(vld1q_f64(p0 + m + 1), coeff, 1));
            e1 = vaddq_f64(e1, vmulq_laneq_f64(vld1q_f64(p1 + m), coeff, 0));
            o1 = vaddq_f64(o1, vmulq_laneq_f64(vld1q_f64(p1 + m + 1), coeff, 1));
        }
    } else {
        for (; m + 1 < stop; m += 2) {
            float64x2_t coeff = vld1q_f64(common + m);
            float64x2_t a0 = vld1q_f64(v0 + m), a1 = vld1q_f64(v1 + m);
            float64x2_t a2 = vld1q_f64(v2 + m), a3 = vld1q_f64(v3 + m);
            e0 = vaddq_f64(e0, vmulq_laneq_f64(vzip1q_f64(a0, a1), coeff, 0));
            o0 = vaddq_f64(o0, vmulq_laneq_f64(vzip2q_f64(a0, a1), coeff, 1));
            e1 = vaddq_f64(e1, vmulq_laneq_f64(vzip1q_f64(a2, a3), coeff, 0));
            o1 = vaddq_f64(o1, vmulq_laneq_f64(vzip2q_f64(a2, a3), coeff, 1));
        }
    }
    if (flip) {
        e0 = vextq_f64(e0, e0, 1); e1 = vextq_f64(e1, e1, 1);
        o0 = vextq_f64(o0, o0, 1); o1 = vextq_f64(o1, o1, 1);
    }
    vst1q_f64(l0, e0); vst1q_f64(l0 + 2, e1);
    vst1q_f64(l1, o0); vst1q_f64(l1 + 2, o1);
    return m;
}
#endif

// `width` (4, 8 or 16) outputs: prefix, shared middle, suffix, fused remainder.
static void convolve_group(int width, int64_t n, const double *a, const double *kernel,
                           int64_t k_lo, int64_t k_hi, const int64_t *need, int finite,
                           double *out) {
    span s[16];
    double l0[16] = {0.0}, l1[16] = {0.0};
    int64_t start = 0, stop = INT64_MAX;
    for (int k = 0; k < width; k++) {
        s[k] = output_span(n, a, kernel, k_lo, k_hi, need[k], finite);
        if (s[k].lo > start) start = s[k].lo;
        if (s[k].main < stop) stop = s[k].main;
    }
    start += start & 1;  // shared middle starts on an even element
    if (stop < start) stop = start;
    for (int k = 0; k < width; k++) {
        int64_t first = start < s[k].main ? start : s[k].main;
        lanes(&s[k], s[k].lo, first, &l0[k], &l1[k]);
    }
    int64_t m = start;
#if defined(__aarch64__)
    if (width == 16) {
        m = middle16(s, need[0] > n / 2, start, stop, l0, l1);
    } else if (width == 8) {
        m = middle8(s, need[0] > n / 2, start, stop, l0, l1);
    } else if (width == 4 && finite &&
               ((need[0] > n / 2 && need[1] > n / 2 && need[2] > n / 2 && need[3] > n / 2) ||
                (need[0] <= n / 2 && need[1] <= n / 2 && need[2] <= n / 2 && need[3] <= n / 2))) {
        m = middle_gather4(s, need[0] > n / 2, start, stop, l0, l1);
    } else
#endif
    {
        const double *x0 = s[0].x, *y0 = s[0].y, *x1 = s[1].x, *y1 = s[1].y;
        const double *x2 = s[2].x, *y2 = s[2].y, *x3 = s[3].x, *y3 = s[3].y;
        for (; m + 1 < stop; m += 2) {
            l0[0] = l0[0] + x0[m] * y0[m];
            l1[0] = l1[0] + x0[m + 1] * y0[m + 1];
            l0[1] = l0[1] + x1[m] * y1[m];
            l1[1] = l1[1] + x1[m + 1] * y1[m + 1];
            l0[2] = l0[2] + x2[m] * y2[m];
            l1[2] = l1[2] + x2[m + 1] * y2[m + 1];
            l0[3] = l0[3] + x3[m] * y3[m];
            l1[3] = l1[3] + x3[m + 1] * y3[m + 1];
        }
    }
    for (int k = 0; k < width; k++) {
        int64_t from = m > s[k].lo ? m : s[k].lo;
        if (start >= s[k].main) from = s[k].main;  // prefix already covered it
        lanes(&s[k], from, s[k].main, &l0[k], &l1[k]);
        out[k] = finish(&s[k], l0[k], l1[k]);
    }
}

int v3_convolve_same(int64_t n, const double *a, const double *kernel, int64_t k_lo,
                     int64_t k_hi, int64_t count, const int64_t *need, double *out) {
    int finite = 1;
    for (int64_t i = 0; i < n; i++)
        if (!isfinite(a[i])) finite = 0;
    int64_t c = 0, left = n / 2;
    while (c + 4 <= count) {
        int wide = c + 8 <= count && need[c + 7] == need[c] + 7 &&
                   (need[c] > left || need[c + 7] <= left);
        int wider = c + 16 <= count && need[c + 15] == need[c] + 15 &&
                    (need[c] > left || need[c + 15] <= left);
#if !defined(__aarch64__)
        wide = wider = 0;
#endif
        int width = wider ? 16 : wide ? 8 : 4;
        convolve_group(width, n, a, kernel, k_lo, k_hi, need + c, finite, out + c);
        c += width;
    }
    for (; c < count; c++) {
        span s = output_span(n, a, kernel, k_lo, k_hi, need[c], finite);
        double l0 = 0.0, l1 = 0.0;
        lanes(&s, s.lo, s.main, &l0, &l1);
        out[c] = finish(&s, l0, l1);
    }
    return !finite;
}

// Batch calculate_SES's direct convolutions in NumPy's order; only storage is reused.
int v3_ses_sums(int64_t levels, int64_t n, const double *flux, const double *sig2,
                const double *templates, int64_t count, const int64_t *need,
                double *num, double *den) {
    double *work = malloc((2 * n + 2 * count) * sizeof(double));
    if (!work) return -1;
    double *weighted = work, *squared = work + n;
    double *cn = work + 2 * n, *cd = cn + count;
    for (int64_t j = 0; j < count; j++) num[j] = den[j] = 0.0;
    for (int64_t i = 0; i < levels; i++) {
        const double *f = flux + i * n, *s = sig2 + i * n, *t = templates + i * n;
        int64_t lo = n, hi = 0, slo = n, shi = 0;
        for (int64_t j = 0; j < n; j++) {
            weighted[j] = s[j] * f[j];
            squared[j] = t[j] * t[j];
            if (t[j] != 0.0) { if (lo == n) lo = j; hi = j + 1; }
            if (squared[j] != 0.0) { if (slo == n) slo = j; shi = j + 1; }
        }
        if (!hi) lo = 0;
        if (!shi) slo = 0;
        v3_convolve_same(n, weighted, t, lo, hi, count, need, cn);
        v3_convolve_same(n, s, squared, slo, shi, count, need, cd);
        double weight = ldexp(1.0, -(i + 1 < levels ? i + 1 : levels - 1));
        for (int64_t j = 0; j < count; j++) {
            num[j] = num[j] + weight * cn[j];
            den[j] = den[j] + weight * cd[j];
        }
    }
    free(work);
    return 0;
}

// In-transit maximum over cadences within dur + 1e-6 d of a predicted mid-transit (t time-ordered).
static int64_t transit_max_windows(int64_t n, const double *t, const double *ses,
                                   double period, double peak, double dur, int first,
                                   double *out_max) {
    double half = period / 2.0, best = 0.0, reach = dur + 1e-6;
    int64_t count = 0, i = 0;
    int nan = 0;
    double k = floor((t[0] - peak - reach) / period) - 1.0;
    while (i < n) {
        double high = k * period + peak + reach;
        if (high < t[i]) {
            double later = floor((t[i] - peak - reach) / period);
            k = later > k + 1.0 ? later : k + 1.0;
            continue;
        }
        double low = k * period + peak - reach;
        if (t[i] < low) {
            // Gallop to the first cadence at or after the window start.
            int64_t step = 1, j = i;
            while (j + step < n && t[j + step] < low) {
                j += step;
                step *= 2;
            }
            int64_t a = j, b = j + step < n ? j + step : n;
            while (b - a > 1) {
                int64_t mid = a + (b - a) / 2;
                if (t[mid] < low)
                    a = mid;
                else
                    b = mid;
            }
            i = b;
        }
        double cycle = k;
        for (; i < n && t[i] <= high; i++) {
            double x = fold(t[i], period, &cycle);
            double shift = remainder_numpy((x - peak) + half, period) - half;
            if (fabs(shift) < dur) {
                if (first) return 1;
                double s = ses[i];
                if (isnan(s))
                    nan = 1;
                else if (count == 0 || s > best)
                    best = s;
                count++;
            }
        }
        k += 1.0;
    }
    *out_max = nan ? NAN : best;
    return count;
}

// Peak location and single-event veto for every period with a winner, as upstream after its duration loop.
void v3_single_events(int64_t n, const double *t, const double *ses, int64_t count,
                      const double *periods, const int32_t *nbins, const int32_t *bins,
                      const int32_t *durations, const double *tdurs, const double *mes,
                      double p_lim, double frac, const int8_t *usable,
                      const int64_t *starts, const int64_t *decisive, double *peaks,
                      int8_t *vetoed) {
    for (int64_t c = 0; c < count; c++) {
        double P = periods[c];
        int32_t nb = nbins[c], b = bins[c], d = durations[c];
        double peak = nb == 1 ? 0.0 : b == nb - 1 ? P : (double)b * (P / (double)(nb - 1));
        double dur = tdurs[d], best;
        const double *row = ses + (int64_t)d * n;
        int windows = 2.0 * (dur + 1e-6) < 0.25 * P;
        peaks[c] = peak;
        if (usable[d]) {
            int kind = P < p_lim ? 0 : 1;
            int decided = 0;
            for (int64_t j = starts[2 * d + kind]; j < starts[2 * d + kind + 1]; j++) {
                int64_t i = decisive[j];
                int passes = kind == 0 ? row[i] > mes[c] : row[i] / mes[c] > frac;
                if (passes && in_transit(t[i], P, peak, dur)) {
                    decided = 1;
                    break;
                }
            }
            if (decided) {
                vetoed[c] = 1;
                continue;
            }
            int64_t inside = windows ? transit_max_windows(n, t, row, P, peak, dur, 1, &best)
                                     : transit_scan(n, t, row, P, peak, dur, 1, &best);
            vetoed[c] = inside ? 0 : -1;
            continue;
        }
        int64_t inside = windows ? transit_max_windows(n, t, row, P, peak, dur, 0, &best)
                                 : transit_scan(n, t, row, P, peak, dur, 0, &best);
        if (inside == 0)
            vetoed[c] = -1;
        else if (P < p_lim)
            vetoed[c] = best > mes[c];
        else
            vetoed[c] = best / mes[c] > frac;
    }
}

// ses.calc_mes for chosen rows; bins share an epoch loop but keep the original order and bounds.
#define BIN_BATCH 64
void v3_bins_mes(int64_t n, const double *t, const double *num, const double *den,
                 int64_t count, const double *periods, const int32_t *nbins,
                 const int32_t *durations, const int32_t *bins, double n_trans,
                 const int32_t *table, int64_t table_size, double grid, double *mes) {
#if defined(__aarch64__)
    // Keep both floor(low/grid) conversions inside int64's range.
    int vector_setup = n > 0 && grid >= 1e-12 && grid <= 1e6 &&
                       isfinite(t[n - 1]) && fabs(t[n - 1]) <= 1e6;
#endif
    for (int64_t base = 0; base < count; base += BIN_BATCH) {
        int w = count - base < BIN_BATCH ? count - base : BIN_BATCH;
        double scale[BIN_BATCH], start[BIN_BATCH], stop[BIN_BATCH];
        double sn[BIN_BATCH] = {0}, sd[BIN_BATCH] = {0}, counts[BIN_BATCH] = {0};
        double low[BIN_BATCH], high[BIN_BATCH];
        int64_t pos[BIN_BATCH];
        int done[BIN_BATCH] = {0};
        const double *nr[BIN_BATCH], *dr[BIN_BATCH];
        for (int q = 0; q < w; q++) {
            int64_t c = base + q;
            scale[q] = (double)nbins[c] / periods[c];
            start[q] = (double)bins[c] / scale[q];
            stop[q] = (double)(bins[c] + 1) / scale[q];
            nr[q] = num + (int64_t)durations[c] * n;
            dr[q] = den + (int64_t)durations[c] * n;
        }
        for (double k = 0.0;; k += 1.0) {
            int active = 0;
            for (int q = 0; q < w; q++) {
#if defined(__aarch64__)
                if (vector_setup && q + 1 < w && !done[q] && !done[q + 1]) {
                    float64x2_t kp = vmulq_n_f64(vld1q_f64(periods + base + q), k);
                    float64x2_t a = vsubq_f64(vaddq_f64(kp, vld1q_f64(start + q)),
                                             vdupq_n_f64(1e-6));
                    float64x2_t z = vaddq_f64(vaddq_f64(kp, vld1q_f64(stop + q)),
                                             vdupq_n_f64(1e-6));
                    vst1q_f64(low + q, a);
                    vst1q_f64(high + q, z);
                    if (low[q] <= t[n - 1] && low[q + 1] <= t[n - 1] &&
                        low[q] >= -1e6 && low[q + 1] >= -1e6) {
                        int64x2_t indices = vsubq_s64(vcvtq_s64_f64(vrndmq_f64(
                            vdivq_f64(a, vdupq_n_f64(grid)))), vdupq_n_s64(1));
                        int64_t js[2];
                        vst1q_s64(js, indices);
                        for (int lane = 0; lane < 2; lane++) {
                            int r = q + lane;
                            int64_t j = js[lane];
                            if (j >= table_size) { done[r] = 1; pos[r] = n; }
                            else { pos[r] = j > 0 ? table[j] : 0; active++; }
                        }
                        q++;
                        continue;
                    }
                }
#endif
                if (done[q]) { pos[q] = n; continue; }
                low[q] = k * periods[base + q] + start[q] - 1e-6;
                high[q] = k * periods[base + q] + stop[q] + 1e-6;
                if (!(low[q] <= t[n - 1])) { done[q] = 1; pos[q] = n; continue; }
                int64_t j = (int64_t)floor(low[q] / grid) - 1;
                if (j >= table_size) { done[q] = 1; pos[q] = n; continue; }
                pos[q] = j > 0 ? table[j] : 0;
                active++;
            }
            if (!active) break;
            for (int q = 0; q < w; q++) {
                int64_t i = pos[q], c = base + q;
                if (i < n) i += t[i] < low[q];
                while (i < n && t[i] < low[q]) i++;
                for (; i < n && t[i] <= high[q]; i++) {
                    double cycle = k;
                    double x = fold(t[i], periods[c], &cycle);
                    if (bin_of(x, scale[q], nbins[c]) != bins[c]) continue;
                    counts[q] += 1.0;
                    sn[q] += nr[q][i];
                    sd[q] += dr[q][i];
                }
            }
        }
        for (int q = 0; q < w; q++) {
            double numerator = 0.0, denominator = 1.0;
            if (!(counts[q] < n_trans)) { numerator = sn[q]; denominator = sd[q]; }
            mes[base + q] = numerator / sqrt(denominator);
        }
    }
}

// One level of pywt 1.9.0 swt_ (periodized a-trous convolution) in its summation order; 1 when F > N.
int v3_swt_level(int64_t N, const double *input, int64_t L, const double *filter,
                 int64_t fstep, double *output) {
    int64_t F = L * fstep;
    if (F > N) return 1;
    for (int64_t o = 0; o < N; o++) {
        int64_t i = F / 2 + o;
        double sum = 0.0;
        if (i < N) {
            for (int64_t t = 0; t < L; t++) {
                int64_t index = i - t * fstep;
                sum = fma(filter[t], input[index < 0 ? index + N : index], sum);
            }
        } else {
            int64_t top = (i - N) / fstep;
            if (top > L - 1) top = L - 1;
            for (int64_t t = top; t >= 0; t--) sum = fma(filter[t], input[i - N - t * fstep], sum);
            for (int64_t t = top + 1; t < L; t++) sum = fma(filter[t], input[i - t * fstep], sum);
        }
        output[o] = sum;
    }
    return 0;
}

// Smallest set bit above x, or -1.
static inline int64_t next_set(const uint64_t *bits, int64_t words, int64_t x) {
    int64_t w = (x + 1) >> 6;
    if (w >= words) return -1;
    uint64_t m = bits[w] & (~0ULL << ((x + 1) & 63));
    while (!m) {
        if (++w >= words) return -1;
        m = bits[w];
    }
    return (w << 6) + __builtin_ctzll(m);
}

// Largest set bit below x, or -1.
static inline int64_t previous_set(const uint64_t *bits, int64_t x) {
    if (x <= 0) return -1;
    int64_t w = (x - 1) >> 6;
    uint64_t m = bits[w] & (~0ULL >> (63 - ((x - 1) & 63)));
    while (!m) {
        if (--w < 0) return -1;
        m = bits[w];
    }
    return (w << 6) + 63 - __builtin_clzll(m);
}

// The index np.pad(..., mode="reflect") reads, without allocating the padded sequence.
static inline int32_t reflected_rank(const int32_t *rank, int64_t r, int64_t half, int64_t n) {
    if (r >= half) return rank[r - half];
    if (n == 1) return rank[0];
    int64_t period = 2 * n - 2, index = (half - r) % period;
    if (index >= n) index = period - index;
    return rank[index];
}

// utils.running_median_insort on the reflect-padded sequence, by rank: the identical double.
int v3_running_kth(int64_t len, const int32_t *rank, int64_t n, const double *sorted,
                   int64_t window, double *out) {
    int64_t half = window / 2, words = (n + 63) / 64;
    int32_t *count = calloc((size_t)(n > 0 ? n : 1), sizeof(int32_t));
    uint64_t *bits = calloc((size_t)(words > 0 ? words : 1), sizeof(uint64_t));
    if (!count || !bits) {
        free(count);
        free(bits);
        return 1;
    }
    int64_t cur = -1, below = 0, first = 0;
    if (n > 0 && half >= (n == 1 ? 1 : 2 * n - 2)) {
        // No outputs precede half: count complete reflection cycles directly.
        int64_t period = n == 1 ? 1 : 2 * n - 2;
        int64_t cycles = half / period, remain = half % period;
        for (int64_t j = 0; j < n; j++)
            count[rank[j]] = cycles * (j == 0 || j == n - 1 ? 1 : 2);
        for (int64_t d = 1; d <= remain; d++) {
            int64_t j = d < n ? d : period - d;
            count[rank[j]]++;
        }
        memset(bits, 0xff, words * sizeof(uint64_t));
        if (n & 63) bits[words - 1] >>= 64 - (n & 63);
        cur = 0;
        first = half;
    }
    for (int64_t r = first; r < len; r++) {
        if (r >= window) {
            int64_t y = reflected_rank(rank, r - window, half, n);
            if (--count[y] == 0) bits[y >> 6] &= ~(1ULL << (y & 63));
            if (y < cur) {
                below--;
            } else if (y == cur && count[y] == 0) {
                int64_t next = next_set(bits, words, y);
                if (next >= 0) {
                    cur = next;
                } else {
                    cur = previous_set(bits, y);
                    if (cur >= 0) below -= count[cur];
                }
            }
        }
        int64_t x = reflected_rank(rank, r, half, n);
        if (count[x]++ == 0) bits[x >> 6] |= 1ULL << (x & 63);
        if (cur < 0) {
            cur = x;
            below = 0;
        } else if (x < cur) {
            below++;
        }
        if (r < half) continue;
        int64_t k = r < window ? (r + 1) / 2 : half;
        while (k < below) {
            cur = previous_set(bits, cur);
            below -= count[cur];
        }
        while (k >= below + count[cur]) {
            below += count[cur];
            cur = next_set(bits, words, cur);
        }
        out[r - half] = sorted[cur];
    }
    free(count);
    free(bits);
    return 0;
}

// numpy.interp at x = 0..n-1 with NumPy's precomputed slopes; multiply and add stay separate.
static void interp_grid(int64_t n, int64_t m, const double *xp, const double *fp,
                        double *slopes, double *out) {
    for (int64_t j = 0; j + 1 < m; j++)
        slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    int64_t j = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = (double)i;
        while (j + 1 < m && xp[j + 1] <= x) j++;
        if (x < xp[0]) out[i] = fp[0];
        else if (j == m - 1 || x == xp[j]) out[i] = fp[j];
        else out[i] = slopes[j] * (x - xp[j]) + fp[j];
    }
}

typedef int (*rank_fn)(int64_t, const double *, int32_t *, double *);

// fast.calc_var_stat's MAD branch, then 1 / sig**2; 1 where the Python path must decide.
static int mad_weights(int64_t n, const double *x, const int32_t *rank,
                       const double *ordered, int64_t window, rank_fn sort,
                       double *out, double *work, int32_t *ranks) {
    if (window < 15001) {
        if (v3_running_kth(n + window / 2, rank, n, ordered, window, out)) return -1;
        for (int64_t j = 0; j < n; j++) out[j] = 1.4826 * out[j];
    } else {
        double dx = (double)window / 15001.0;
        int64_t m = (int64_t)((double)n / dx);
        if (m < 2 || m > n) return 1;
        double *xp = work, *values = work + n, *sorted = work + 2 * n;
        double *median = work + 3 * n, *slopes = work + 4 * n;
        double step = (double)n / (double)(m - 1);
        for (int64_t j = 0; j < m; j++) xp[j] = (double)j * step;
        xp[m - 1] = (double)n;
        for (int64_t j = 0; j < m; j++) {
            double t = xp[j];
            int64_t k = (int64_t)t;
            double value = k >= n - 1 ? x[n - 1]
                         : t == (double)k ? x[k]
                         : (x[k + 1] - x[k]) * (t - (double)k) + x[k];
            values[j] = fabs(value);
        }
        if (sort(m, values, ranks, sorted)) return 1;
        if (v3_running_kth(m + 15001 / 2, ranks, m, sorted, 15001, median)) return -1;
        for (int64_t j = 0; j < m; j++) median[j] = 1.4826 * median[j];
        interp_grid(n, m, xp, median, slopes, out);
    }
    for (int64_t j = 0; j < n; j++) {
        if (out[j] == 0.0 || !isfinite(out[j])) return 1;
        double squared = out[j] * out[j];
        if (squared == 0.0 || !isfinite(squared)) return 1;
        out[j] = 1.0 / squared;
        if (!isfinite(out[j])) return 1;
    }
    return 0;
}

// calc_var_stat's MAD weights for (level, window) pairs of one segment; 1 when the Python path is needed.
int v3_mad_pairs(int64_t pairs, const int64_t *level, const int64_t *window, int64_t n,
                 const double *flux, const int32_t *rank, const double *ordered,
                 void *sort, double *sig2) {
    double *work = malloc((size_t)(5 * n) * sizeof(double));
    int32_t *ranks = malloc((size_t)n * sizeof(int32_t));
    int status = -1;
    if (!work || !ranks) goto done;
    status = 0;
    for (int64_t p = 0; p < pairs && !status; p++) {
        int64_t i = level[p];
        status = mad_weights(n, flux + i * n, rank + i * n, ordered + i * n, window[p],
                             (rank_fn)sort, sig2 + p * n, work, ranks);
    }
done:
    free(work);
    free(ranks);
    return status;
}

// upstream kep_period_sampling's recurrence, operation for operation, with libm pow.
int64_t v3_period_sampling(double p_min, double limit, double baseline, double d_leastsq,
                           double rho_star, double *out, int64_t capacity) {
    double Pt = p_min;
    int64_t count = 0;
    if (count < capacity) out[count] = Pt;
    count++;
    while (Pt < limit) {
        double d_dur = d_leastsq * (13.0 / 24.0) * pow(Pt / 365.0, 1.0 / 3.0) *
                       pow(rho_star, -1.0 / 3.0);
        Pt += 4.0 * d_dur * (Pt / baseline);
        if (count < capacity) out[count] = Pt;
        count++;
    }
    return count;
}
