// Exact replicas of the Vetter residual models, bitwise; no fast-math or contraction (v3_nonlinear_ld is the integral).

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// scipy.special.expit: 1 / (1 + exp(-x)).
static inline double expit(double x) {
    // These tails already round to exactly 1 or 0 in float64.
    if (x >= 37.0) return 1.0;
    if (x <= -710.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

static inline double one_minus_exp(double x) {
    return x <= -38.0 ? 1.0 : 1.0 - exp(x);
}

// jump(t, a, b, t0) = a * expit(b*(t-t0)).
static inline double jump(double t, double a, double b, double t0) {
    return a * expit(b * (t - t0));
}

// box(t, a, b, tdur, t0) = jump(t - tdur/2., ...) - jump(t + tdur/2., ...).
static inline double box(double t, double a, double b, double tdur, double t0) {
    double half = tdur / 2.0;
    return jump(t - half, a, b, t0) - jump(t + half, a, b, t0);
}

// poly + box: before the ingress |box| is under a quarter ulp of pl, so the sum is pl and the exps are skipped.
static inline double poly_box(double t, double pl, double a, double b, double tdur,
                              double t0) {
    double half = tdur / 2.0;
    if (b * ((t - half) - t0) <= -40.0 && b * ((t + half) - t0) <= -40.0 &&
        fabs(a) <= 1.0 && fabs(pl) >= 0.5 && isfinite(pl))
        return pl;
    return pl + box(t, a, b, tdur, t0);
}

// spsd(t, a, b, t0, d, tdur) = jump(t, a, b, t0) * (1. - np.exp(-d*(t - tdur/2.))).
static inline double spsd(double t, double a, double b, double t0, double d, double tdur) {
    double half = tdur / 2.0;
    return jump(t, a, b, t0) * one_minus_exp((-d) * (t - half));
}

// sine(t, a, tdur, t0) = -a*np.cos((t-t0)/(2*tdur)).
static inline double sine_term(double t, double a, double tdur, double t0) {
    return (-a) * cos((t - t0) / (2.0 * tdur));
}

// (c1*x + c2*x**2. + c0): NumPy's x**2. is x*x.
static inline double poly(double x, double c0, double c1, double c2) {
    return ((c1 * x) + (c2 * (x * x))) + c0;
}

// One entry point; p holds the parameters of kind 0 trap, 1 box, 2 spsd, 3 sine as the wrapper documents.
int v3_residual(int kind, int64_t n, const double *x, const double *data, const double *err,
                const double *p, double *out) {
    int nan = 0;
    for (int64_t i = 0; i < n; i++) {
        double model, t = x[i];
        switch (kind) {
        case 0: model = poly_box(t, poly(t, p[4], p[5], p[6]), p[0], p[1], p[2], p[3]); break;
        case 1: model = poly_box(t, poly(t, p[3], p[4], p[5]), p[0], 5e3, p[1], p[2]); break;
        case 2: model = poly(t, p[4], p[5], p[6]) + spsd(t, p[0], 9999.0, p[2], p[1], p[3]); break;
        default: model = poly(t, p[3], p[4], p[5]) + sine_term(t, p[0], p[1], p[2]); break;
        }
        if (isnan(model)) nan = 1;
        out[i] = kind == 3 ? (data[i] - model) / (err[i] * err[i]) : (data[i] - model) / err[i];
    }
    return nan;
}

// NumPy 1.26 DOUBLE_pairwise_sum for one block of at most 8192 elements.
static double pairwise_sum(const double *a, int64_t n) {
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; i++) res += a[i];
        return res;
    } else if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        int64_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    } else {
        int64_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

// v3_residual, then np.sum(out) and (out*out).sum() for n <= 8192 (one pairwise block).
int v3_residual_sums(int kind, int64_t n, const double *x, const double *data,
                     const double *err, const double *p, double *out, double *squares,
                     double *sums) {
    int nan = v3_residual(kind, n, x, data, err, p, out);
    for (int64_t i = 0; i < n; i++) squares[i] = out[i] * out[i];
    sums[0] = pairwise_sum(out, n);
    sums[1] = pairwise_sum(squares, n);
    return nan;
}

// Parameter.from_internal and its value clipping, in their original order.
static double external(double v, double low, double high) {
    double value;
    if (low == -INFINITY && high == INFINITY) value = v;
    else if (high == INFINITY) value = low - 1.0 + sqrt(v * v + 1.0);
    else if (low == -INFINITY) value = high + 1.0 - sqrt(v * v + 1.0);
    else value = low + (sin(v) + 1.0) * (high - low) / 2.0;
    if (!isfinite(value)) return value;
    if (value > high) value = high;
    else if (value < low) value = low;
    return value;
}

// Scalar.gradient's two-point evaluations; nonfinite intermediates take the Python path.
int v3_direct_gradient(int kind, int64_t n, const double *t, const double *data,
                       const double *err, int m, const double *x, const double *h,
                       double f0, const double *low, const double *high, int np,
                       const int32_t *map, const double *fixed, double *scratch,
                       double *squares, double *g, double *last) {
    if (m < 1 || m > 7 || np < 1 || np > 7) return 1;
    double p[7], sums[2];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            // Only the previous and current perturbed variables change.
            if (i && j != i && j != i - 1) continue;
            double value = external(x[j] + (i == j ? h[j] : 0.0), low[j], high[j]);
            if (!isfinite(value)) return 1;
            last[j] = value;
        }
        for (int j = 0; j < np; j++) p[j] = map[j] < 0 ? fixed[j] : last[map[j]];
        int nan = v3_residual_sums(kind, n, t, data, err, p, scratch, squares, sums);
        if (nan || !isfinite(sums[0])) return 1;
        g[i] = (sums[1] - f0) / ((x[i] + h[i]) - x[i]);
    }
    return 0;
}

// The base objective and its finite differences in one call (fg: f0, gradient, last step).
int v3_value_gradient(int kind, int64_t n, const double *t, const double *data,
                       const double *err, int m, const double *x, const double *low,
                       const double *high, int np, const int32_t *map,
                       const double *fixed, double *scratch, double *squares,
                       double *fg, double *last) {
    if (m < 1 || m > 7 || np < 1 || np > 7) return 1;
    double p[7], h[7], sums[2];
    for (int j = 0; j < m; j++) {
        last[j] = external(x[j], low[j], high[j]);
        if (!isfinite(last[j])) return 1;
        h[j] = (x[j] + 1e-8) - x[j] == 0.0
                   ? 0x1p-26 * (x[j] >= 0.0 ? 1.0 : -1.0) * fmax(1.0, fabs(x[j]))
                   : 1e-8;
    }
    for (int j = 0; j < np; j++) p[j] = map[j] < 0 ? fixed[j] : last[map[j]];
    int nan = v3_residual_sums(kind, n, t, data, err, p, scratch, squares, sums);
    if (nan || !isfinite(sums[0])) return 1;
    fg[0] = sums[1];
    fg[m + 1] = h[m - 1];
    return v3_direct_gradient(kind, n, t, data, err, m, x, h, fg[0], low, high,
                               np, map, fixed, scratch, squares, fg + 1, last);
}

// Occulted flux for batman's nonlinear limb darkening, integrated accurately (batman 2.5.2 uses a midpoint rule).

#define LD_CLAMP 0.99995
#define LD_XM 0.9
#define LD_K 36.0
#define LD_MIN_NODES 4
#define LD_MAX_NODES 64

typedef struct { double c1, c2, c3, c4, norm, sc, ic; } ld_law;

static inline double ld_intensity_s(const ld_law *law, double s) {
    if (s < law->sc) s = law->sc;  /* batman's clamp, x <= 0.99995 */
    double s2 = s * s;
    return (1.0 - law->c1 * (1.0 - s) - law->c2 * (1.0 - s2) - law->c3 * (1.0 - s2 * s)
            - law->c4 * (1.0 - s2 * s2)) / law->norm;
}

static inline double ld_s(double x) { return sqrt(sqrt(fmax(1.0 - x * x, 0.0))); }

// pi * integral of I(s) 4 s^3 ds, the antiderivative for the covered disc.
static inline double ld_disc(const ld_law *law, double s) {
    double a = 1.0 - law->c1 - law->c2 - law->c3 - law->c4;
    double s2 = s * s, s4 = s2 * s2;
    return 4.0 * M_PI / law->norm * s4 *
           (a / 4.0 + s * (law->c1 / 5.0 + s * (law->c2 / 6.0 + s * (law->c3 / 7.0 + s * law->c4 / 8.0))));
}

static inline double ld_theta(double d, double p, double x) {
    double arg = (d * d + x * x - p * p) / (2.0 * d * x);
    return acos(arg > 1.0 ? 1.0 : (arg < -1.0 ? -1.0 : arg));
}

static int ld_nodes(double delta, double h) {
    if (!(h > 0.0)) return LD_MIN_NODES;
    double eta = delta > 0.0 ? acosh(1.0 + delta / h) : 1e-300;
    double complex z = 1.0 + I * (2.0 * eta / M_PI);
    double rho = cabs(z + csqrt(z - 1.0) * csqrt(z + 1.0));
    double n = ceil(LD_K / (2.0 * log(rho)));
    return n < LD_MIN_NODES ? LD_MIN_NODES : (n > LD_MAX_NODES ? LD_MAX_NODES : (int)n);
}

// Distance from [a, b] to the nearest point of `points` outside it.
static double ld_gap(const double *points, int count, double a, double b, double none) {
    double gap = none;
    for (int i = 0; i < count; i++) {
        double g = points[i] < a ? a - points[i] : (points[i] > b ? points[i] - b : -1.0);
        if (g > 0.0 && g < gap) gap = g;
    }
    return gap;
}

// The partial ring over [a, b], below XM in x, above in s.
static double ld_ring(const ld_law *law, double d, double p, double a, double b, double lo,
                      double hi, const double *cosines, const double *weights,
                      const int32_t *offsets) {
    double singular[5] = {0.0, lo, hi, LD_CLAMP, 1.0};
    double sum = 0.0;
    if (b <= LD_XM) {
        double c = 0.5 * (a + b), h = 0.5 * (b - a);
        int n = ld_nodes(ld_gap(singular, 5, a, b, 1.0), h);
        const double *cs = cosines + offsets[n], *ws = weights + offsets[n];
        for (int j = 0; j < n; j++) {
            double x = c - h * cs[j];
            sum += ws[j] * ld_intensity_s(law, ld_s(x)) * 2.0 * x * ld_theta(d, p, x);
        }
        return h * sum;
    }
    double sa = ld_s(a), sb = ld_s(b);
    double mapped[6];
    for (int i = 0; i < 4; i++) mapped[i] = ld_s(singular[i]);  /* s = 1 is x = 0 */
    mapped[4] = 1.0;
    mapped[5] = 1.0;
    double c = 0.5 * (sa + sb), h = 0.5 * (sa - sb);
    int n = ld_nodes(ld_gap(mapped, 6, sb, sa, 1.0), h);
    const double *cs = cosines + offsets[n], *ws = weights + offsets[n];
    for (int j = 0; j < n; j++) {
        double s = c - h * cs[j], s2 = s * s;
        double x = sqrt(fmax(1.0 - s2 * s2, 0.0));
        sum += ws[j] * ld_intensity_s(law, s) * 4.0 * s2 * s * ld_theta(d, p, x);
    }
    return h * sum;
}

// Flux at separation d for planet radius p, as batman._nonlinear_ld.
static double ld_flux(const ld_law *law, double whole, double di, double p,
                      const double *cosines, const double *weights, const int32_t *offsets) {
    double x_in = fmax(di - p, 0.0), x_out = fmin(di + p, 1.0);
    if (x_in >= 1.0 || x_out - x_in < 1e-7) return 1.0;  /* batman's two exits */
    double delta = 0.0;
    if (di < p) {  /* covered disc, x < p - d */
        double top = fmin(p - di, 1.0);
        if (top <= LD_CLAMP) {
            delta += whole - ld_disc(law, ld_s(top));
        } else {
            delta += whole - ld_disc(law, law->sc) + law->ic * M_PI * (top * top - LD_CLAMP * LD_CLAMP);
        }
    }
    double lo = fabs(di - p), hi = x_out;
    if (lo < hi) {
        double cuts[4] = {lo, 0.0, 0.0, hi};
        int count = 1;
        if (lo < LD_XM && LD_XM < hi) cuts[count++] = LD_XM;
        if (lo < LD_CLAMP && LD_CLAMP < hi) cuts[count++] = LD_CLAMP;
        cuts[count++] = hi;
        for (int k = 0; k + 1 < count; k++)
            delta += ld_ring(law, di, p, cuts[k], cuts[k + 1], lo, hi, cosines, weights, offsets);
    }
    return 1.0 - delta;
}

static ld_law ld_make(double c1, double c2, double c3, double c4) {
    ld_law law = {c1, c2, c3, c4, (-c1 / 10.0 - c2 / 6.0 - 3.0 * c3 / 14.0 - c4 / 4.0 + 0.5) * 2.0 * M_PI,
                  ld_s(LD_CLAMP), 0.0};
    law.ic = ld_intensity_s(&law, law.sc);
    return law;
}

// Flux at separations d[0..n) for planet radius p, as batman._nonlinear_ld.
int v3_nonlinear_ld(int64_t n, const double *d, double p, double c1, double c2, double c3,
                    double c4, const double *cosines, const double *weights,
                    const int32_t *offsets, double *flux) {
    ld_law law = ld_make(c1, c2, c3, c4);
    double whole = ld_disc(&law, 1.0);
    for (int64_t i = 0; i < n; i++)
        flux[i] = ld_flux(&law, whole, d[i], p, cosines, weights, offsets);
    return 0;
}

// The same flux by piecewise Chebyshev interpolants on its analytic regions.
#define LD_CHEB 24
#define LD_CHEB_TOL 1e-13
#define LD_GAP 2e-3
#define LD_MIN_WIDTH 2e-4
#define LD_MAX_PIECES 512
#define LD_BUCKETS 2048

typedef struct {
    double key[5];
    int used, pieces;
    double end;  /* pieces cover [0, end) contiguously; gaps have ok = 0 */
    double a[LD_MAX_PIECES], b[LD_MAX_PIECES];
    int ok[LD_MAX_PIECES];
    double coef[LD_MAX_PIECES][LD_CHEB];
    int bucket[LD_BUCKETS];  /* first piece whose end exceeds the bucket start */
} ld_table;

static _Thread_local ld_table ld_cache[2];
static _Thread_local int ld_clock;

// Chebyshev coefficients on [a, b] from LD_CHEB first-kind points.
static void ld_fit(const ld_law *law, double whole, double p, double a, double b,
                   const double *cosines, const double *weights, const int32_t *offsets,
                   double *coef) {
    double values[LD_CHEB];
    for (int j = 0; j < LD_CHEB; j++) {
        double x = cos(M_PI * (j + 0.5) / LD_CHEB);
        values[j] = ld_flux(law, whole, 0.5 * (a + b) + 0.5 * (b - a) * x, p, cosines,
                            weights, offsets);
    }
    for (int k = 0; k < LD_CHEB; k++) {
        double sum = 0.0;
        for (int j = 0; j < LD_CHEB; j++) sum += values[j] * cos(M_PI * k * (j + 0.5) / LD_CHEB);
        coef[k] = (k == 0 ? 1.0 : 2.0) * sum / LD_CHEB;
    }
}

static int ld_converged(const double *coef) {
    double tail = fabs(coef[LD_CHEB - 1]) + fabs(coef[LD_CHEB - 2]) + fabs(coef[LD_CHEB - 3]);
    return tail <= LD_CHEB_TOL;
}

// Append a piece; returns 0 when the table is full.
static int ld_piece(ld_table *t, double a, double b, int ok) {
    if (t->pieces >= LD_MAX_PIECES) return 0;
    t->a[t->pieces] = a;
    t->b[t->pieces] = b;
    t->ok[t->pieces] = ok;
    t->pieces++;
    return 1;
}

// Pieces of the analytic region [lo, hi), in order; 0 when the table filled up.
static int ld_region(ld_table *t, const ld_law *law, double whole, double p, double lo,
                     double hi, const double *cosines, const double *weights,
                     const int32_t *offsets) {
    double end = t->pieces ? t->b[t->pieces - 1] : 0.0;
    if (lo > end && !ld_piece(t, end, lo, 0)) return 0;
    double stack_a[64], stack_b[64];
    int top = 0;
    stack_a[top] = lo;
    stack_b[top++] = hi;
    // Depth-first, right half pushed first, so pieces come out in order.
    while (top > 0) {
        double a = stack_a[--top], b = stack_b[top];
        if (t->pieces >= LD_MAX_PIECES) return 0;
        double *coef = t->coef[t->pieces];
        ld_fit(law, whole, p, a, b, cosines, weights, offsets, coef);
        int ok = ld_converged(coef);
        if (!ok && b - a > LD_MIN_WIDTH && top + 2 <= 64) {
            double m = 0.5 * (a + b);
            stack_a[top] = m;
            stack_b[top++] = b;
            stack_a[top] = a;
            stack_b[top++] = m;
            continue;
        }
        ld_piece(t, a, b, ok);
    }
    return 1;
}

static void ld_build(ld_table *t, const ld_law *law, double whole, double p,
                     const double *cosines, const double *weights, const int32_t *offsets) {
    t->pieces = 0;
    double inside = LD_CLAMP - p - LD_GAP;
    double over_lo = 1.0 - p + LD_GAP, over_hi = LD_CLAMP + p - LD_GAP;
    int room = 1;
    if (inside > LD_MIN_WIDTH)
        room = ld_region(t, law, whole, p, 0.0, inside, cosines, weights, offsets);
    if (room && over_hi - over_lo > LD_MIN_WIDTH && over_lo > (t->pieces ? t->b[t->pieces - 1] : 0.0))
        ld_region(t, law, whole, p, over_lo, over_hi, cosines, weights, offsets);
    t->end = t->pieces ? t->b[t->pieces - 1] : 0.0;
    int k = 0;
    for (int j = 0; j < LD_BUCKETS; j++) {
        double start = t->end * j / LD_BUCKETS;
        while (k + 1 < t->pieces && t->b[k] <= start) k++;
        t->bucket[j] = k;
    }
}

static const ld_table *ld_lookup(double p, double c1, double c2, double c3, double c4,
                                 const ld_law *law, double whole, const double *cosines,
                                 const double *weights, const int32_t *offsets) {
    double key[5] = {p, c1, c2, c3, c4};
    for (int i = 0; i < 2; i++)
        if (ld_cache[i].used && memcmp(ld_cache[i].key, key, sizeof key) == 0) return &ld_cache[i];
    ld_table *t = &ld_cache[ld_clock];
    ld_clock ^= 1;
    memcpy(t->key, key, sizeof key);
    t->used = 1;
    ld_build(t, law, whole, p, cosines, weights, offsets);
    return t;
}

static inline double ld_clenshaw(const double *coef, double a, double b, double d) {
    double x = (2.0 * d - a - b) / (b - a);
    double b1 = 0.0, b2 = 0.0;
    for (int k = LD_CHEB - 1; k >= 1; k--) {
        double t = 2.0 * x * b1 - b2 + coef[k];
        b2 = b1;
        b1 = t;
    }
    return x * b1 - b2 + coef[0];
}

int v3_nonlinear_ld_fast(int64_t n, const double *d, double p, double c1, double c2,
                         double c3, double c4, const double *cosines, const double *weights,
                         const int32_t *offsets, double *flux) {
    ld_law law = ld_make(c1, c2, c3, c4);
    double whole = ld_disc(&law, 1.0);
    if (!(p > 0.0) || !(p < 0.9) || !isfinite(c1 + c2 + c3 + c4))
        return v3_nonlinear_ld(n, d, p, c1, c2, c3, c4, cosines, weights, offsets, flux);
    const ld_table *t = ld_lookup(p, c1, c2, c3, c4, &law, whole, cosines, weights, offsets);
    double scale = t->end > 0.0 ? LD_BUCKETS / t->end : 0.0;
    for (int64_t i = 0; i < n; i++) {
        double di = d[i];
        if (di >= 0.0 && di < t->end) {
            int j = (int)(di * scale);
            int k = t->bucket[j < LD_BUCKETS - 1 ? j : LD_BUCKETS - 1];
            while (t->b[k] <= di) k++;
            if (t->ok[k] && t->a[k] <= di) {
                flux[i] = ld_clenshaw(t->coef[k], t->a[k], t->b[k], di);
                continue;
            }
        }
        flux[i] = ld_flux(&law, whole, di, p, cosines, weights, offsets);
    }
    return 0;
}
