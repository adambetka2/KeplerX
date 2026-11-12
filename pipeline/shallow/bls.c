/*
 * Adapted from Astropy 7.2.2 BLS. Keep operation order and compile with
 * -ffp-contract=off for arm64 parity with BoxLeastSquares.power(objective="snr").
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { OBJECTIVE, DEPTH, DEPTH_ERR, DURATION, PHASE, DEPTH_SNR, LOG_LIKELIHOOD, OUTPUTS };

typedef struct {
    int N, n_durations, oversample;
    const double *x, *y, *ivar, *durations;
    double min_t, sum_y, sum_ivar, bin_duration;
} Series;

static double least(const double *values, int n)
{
    double result = values[0];
    for (int i = 1; i < n; ++i) result = fmin(result, values[i]);
    return result;
}

static double greatest(const double *values, int n)
{
    double result = values[0];
    for (int i = 1; i < n; ++i) result = fmax(result, values[i]);
    return result;
}

static int bin_count(const Series *s, double period)
{
    return (int)ceil(period / s->bin_duration) + s->oversample;
}

/* The index loop vectorizes only while it is separate from the scatter. */
static void bin_pair(const Series *s, double period_a, double period_b, int *index_a,
                     int *index_b, double *sums_a, double *sums_b)
{
    if (!index_b) {
        for (int n = 0; n < s->N; ++n) {
            double wrapped = fma(-period_a, floor(s->x[n] / period_a), s->x[n]);
            index_a[n] = (int)(wrapped / s->bin_duration) + 1;
        }
        memset(sums_a, 0, 2 * (size_t)(bin_count(s, period_a) + 1) * sizeof(double));
        for (int n = 0; n < s->N; ++n) {
            double *a = sums_a + 2 * index_a[n];
            a[0] = fma(s->y[n], s->ivar[n], a[0]);
            a[1] += s->ivar[n];
        }
        return;
    }
    for (int n = 0; n < s->N; ++n) {
        double wrapped_a = fma(-period_a, floor(s->x[n] / period_a), s->x[n]);
        double wrapped_b = fma(-period_b, floor(s->x[n] / period_b), s->x[n]);
        index_a[n] = (int)(wrapped_a / s->bin_duration) + 1;
        index_b[n] = (int)(wrapped_b / s->bin_duration) + 1;
    }
    memset(sums_a, 0, 2 * (size_t)(bin_count(s, period_a) + 1) * sizeof(double));
    memset(sums_b, 0, 2 * (size_t)(bin_count(s, period_b) + 1) * sizeof(double));
    for (int n = 0; n < s->N; ++n) {
        double *a = sums_a + 2 * index_a[n], *b = sums_b + 2 * index_b[n];
        a[0] = fma(s->y[n], s->ivar[n], a[0]);
        a[1] += s->ivar[n];
        b[0] = fma(s->y[n], s->ivar[n], b[0]);
        b[1] += s->ivar[n];
    }
}

static void cumulate(const Series *s, int n_bins, double *sums)
{
    for (int n = 1, ind = n_bins - s->oversample; n <= s->oversample; ++n, ++ind) {
        sums[2 * ind] = sums[2 * n];
        sums[2 * ind + 1] = sums[2 * n + 1];
    }
    for (int n = 1; n <= n_bins; ++n) {
        sums[2 * n] += sums[2 * n - 2];
        sums[2 * n + 1] += sums[2 * n - 1];
    }
}

static void record(const Series *s, const double *sums, double period, int dur, int n,
                   double *out, int stride)
{
    double y_in = sums[2 * (n + dur)] - sums[2 * n];
    double ivar_in = sums[2 * (n + dur) + 1] - sums[2 * n + 1];
    double y_out = s->sum_y - y_in;
    double ivar_out = s->sum_ivar - ivar_in;
    y_in /= ivar_in;
    y_out /= ivar_out;
    double depth = y_out - y_in;
    double depth_err = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
    double duration = dur * s->bin_duration;
    out[DEPTH * stride] = depth;
    out[DEPTH_ERR * stride] = depth_err;
    out[DURATION * stride] = duration;
    out[PHASE * stride] = fmod(fma(n, s->bin_duration, 0.5 * duration) + s->min_t, period);
    out[DEPTH_SNR * stride] = depth / depth_err;
    out[LOG_LIKELIHOOD * stride] = 0.5 * ivar_in * depth * depth;
}

/* Astropy keeps the first model whose SNR exceeds all earlier ones, with durations
   in order and phases in order within each duration; the scan preserves that order
   and skips blocks of phases that cannot raise the maximum. */
static void search(const Series *s, double period, const double *sums, double *score,
                   double *out, int stride)
{
    int n_bins = bin_count(s, period), best_dur = -1, best_n = 0;
    double best = -INFINITY;
    for (int k = 0; k < s->n_durations; ++k) {
        int dur = (int)round(s->durations[k] / s->bin_duration);
        int n_max = n_bins - dur;
        for (int n = 0; n <= n_max; ++n) {
            double y_in = sums[2 * (n + dur)] - sums[2 * n];
            double ivar_in = sums[2 * (n + dur) + 1] - sums[2 * n + 1];
            double y_out = s->sum_y - y_in;
            double ivar_out = s->sum_ivar - ivar_in;
            y_in /= ivar_in;
            y_out /= ivar_out;
            double snr = (y_out - y_in) / sqrt(1.0 / ivar_in + 1.0 / ivar_out);
            int valid = (ivar_in >= DBL_EPSILON) & (ivar_out >= DBL_EPSILON) & (y_out >= y_in);
            score[n] = valid ? snr : -INFINITY;
        }
        for (int block = 0; block <= n_max; block += 16) {
            int end = block + 16 <= n_max + 1 ? block + 16 : n_max + 1, above = 0;
            for (int n = block; n < end; ++n) above |= score[n] > best;
            if (!above) continue;
            for (int n = block; n < end; ++n) {
                if (score[n] > best) {
                    best = score[n];
                    best_dur = dur;
                    best_n = n;
                }
            }
        }
    }
    out[OBJECTIVE * stride] = best;
    if (best_dur < 0) {
        for (int i = DEPTH; i < OUTPUTS; ++i) out[i * stride] = NAN;
        return;
    }
    record(s, sums, period, best_dur, best_n, out, stride);
}

/* Fills out[i * n_periods + p] with output i for period p. Returns 1 or 2 for
   invalid periods or durations as astropy does, and -1 when memory runs out. */
int bls(int N, const double *t, const double *y, const double *ivar, int n_periods,
        const double *periods, int n_durations, const double *durations, int oversample,
        double *out)
{
    double min_period = least(periods, n_periods);
    double min_duration = least(durations, n_durations);
    if (min_period < DBL_EPSILON) return 1;
    if (greatest(durations, n_durations) > min_period || min_duration < DBL_EPSILON) return 2;

    Series s = {.N = N, .n_durations = n_durations, .oversample = oversample, .y = y,
                .ivar = ivar, .durations = durations, .min_t = INFINITY,
                .bin_duration = min_duration / (double)oversample};
    size_t bins = (size_t)bin_count(&s, greatest(periods, n_periods)) + 1;
    double *values = malloc(((size_t)N + 5 * bins) * sizeof(double));
    int *index_a = malloc(2 * (size_t)N * sizeof(int));
    if (!values || !index_a) {
        free(values);
        free(index_a);
        return -1;
    }
    double *x = values, *sums_a = x + N, *sums_b = sums_a + 2 * bins;
    double *score = sums_b + 2 * bins;
    int *index_b = index_a + N;

    for (int i = 0; i < N; ++i) {
        s.min_t = fmin(s.min_t, t[i]);
        s.sum_y = fma(y[i], ivar[i], s.sum_y);
        s.sum_ivar += ivar[i];
    }
    for (int i = 0; i < N; ++i) x[i] = t[i] - s.min_t;
    s.x = x;
    for (int p = 0; p < n_periods; p += 2) {
        int q = p + 1 < n_periods ? p + 1 : p;
        bin_pair(&s, periods[p], periods[q], index_a, q == p ? NULL : index_b,
                 sums_a, sums_b);
        cumulate(&s, bin_count(&s, periods[p]), sums_a);
        search(&s, periods[p], sums_a, score, out + p, n_periods);
        if (q == p) break;
        cumulate(&s, bin_count(&s, periods[q]), sums_b);
        search(&s, periods[q], sums_b, score, out + q, n_periods);
    }
    free(values);
    free(index_a);
    return 0;
}
