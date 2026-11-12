#include <limits.h>
#include <stdint.h>
#include "../../pipeline/shallow/bls.c"

static int batch_dimensions(int N, int n_series, int n_periods,
                            int n_durations, int oversample)
{
    if (N <= 0 || n_series != 3 || n_periods <= 0 || n_durations <= 0 ||
        oversample <= 0) return 3;
    if ((size_t)N > SIZE_MAX / (3 * sizeof(double)) ||
        (size_t)n_periods > SIZE_MAX / (3 * OUTPUTS * sizeof(double))) return 3;
    return 0;
}

static int batch_pointers(const double *t, const double *y, const double *ivar,
                          const double *periods, const double *durations, double *out)
{
    return t && y && ivar && periods && durations && out ? 0 : 3;
}

static int batch_series_values(int N, const double *t, const double *y,
                               const double *ivar)
{
    for (int n = 0; n < N; ++n)
        if (!isfinite(t[n]) || !isfinite(y[n])) return 3;
    for (size_t n = 0; n < 3 * (size_t)N; ++n)
        if (!isfinite(ivar[n]) || ivar[n] <= 0) return 3;
    return 0;
}

static int batch_grid_values(int n_periods, const double *periods,
                             int n_durations, const double *durations)
{
    for (int p = 0; p < n_periods; ++p)
        if (!isfinite(periods[p])) return 3;
    for (int d = 0; d < n_durations; ++d)
        if (!isfinite(durations[d])) return 3;
    if (least(periods, n_periods) < DBL_EPSILON) return 1;
    if (least(durations, n_durations) < DBL_EPSILON ||
        greatest(durations, n_durations) > least(periods, n_periods)) return 2;
    return 0;
}

static int batch_guard(int N, const double *t, const double *y, int n_series,
                       const double *ivar, int n_periods, const double *periods,
                       int n_durations, const double *durations, int oversample,
                       double *out)
{
    int status = batch_dimensions(N, n_series, n_periods, n_durations, oversample);
    if (status) return status;
    status = batch_pointers(t, y, ivar, periods, durations, out);
    if (status) return status;
    status = batch_series_values(N, t, y, ivar);
    if (status) return status;
    return batch_grid_values(n_periods, periods, n_durations, durations);
}

static void batch_initialize(Series *series, int N, const double *t,
                             const double *y, const double *ivar, int n_durations,
                             const double *durations, int oversample)
{
    for (int k = 0; k < 3; ++k) {
        series[k] = (Series){.N = N, .n_durations = n_durations,
                            .oversample = oversample, .y = y,
                            .ivar = ivar + (size_t)k * N, .durations = durations,
                            .min_t = INFINITY,
                            .bin_duration = least(durations, n_durations) / oversample};
        for (int n = 0; n < N; ++n) {
            series[k].min_t = fmin(series[k].min_t, t[n]);
            series[k].sum_y = fma(y[n], series[k].ivar[n], series[k].sum_y);
            series[k].sum_ivar += series[k].ivar[n];
        }
    }
}

static int batch_sizes(int N, int n_periods, const Series *s,
                       const double *periods, size_t *bins)
{
    double count = ceil(greatest(periods, n_periods) / s->bin_duration);
    if (!isfinite(count) || count > INT_MAX / 2 - s->oversample - 1) return 3;
    *bins = (size_t)(int)count + s->oversample + 1;
    if ((size_t)N > SIZE_MAX / (3 * sizeof(double)) ||
        (size_t)N > SIZE_MAX / (2 * sizeof(int)) ||
        (size_t)n_periods > SIZE_MAX / (3 * OUTPUTS * sizeof(double)) ||
        *bins > (SIZE_MAX / sizeof(double) - (size_t)N) / 5) return 3;
    return 0;
}

static int batch_index(double x, double period, double bin_duration,
                        int bins, int *index)
{
    double wrapped = fma(-period, floor(x / period), x);
    double value = wrapped / bin_duration;
    if (!isfinite(value) || value < -1 || value >= bins) return 3;
    *index = (int)value + 1;
    return 0;
}

static int batch_indices(const Series *s, double period_a, double period_b,
                         int *index_a, int *index_b)
{
    int bins_a = bin_count(s, period_a), bins_b = bin_count(s, period_b);
    for (int n = 0; n < s->N; ++n) {
        int status = batch_index(s->x[n], period_a, s->bin_duration,
                                  bins_a, index_a + n);
        if (status) return status;
        if (index_b) {
            status = batch_index(s->x[n], period_b, s->bin_duration,
                                  bins_b, index_b + n);
            if (status) return status;
        }
    }
    return 0;
}

static void batch_scatter(const Series *s, double period_a, double period_b,
                          const int *index_a, const int *index_b,
                          double *sums_a, double *sums_b)
{
    memset(sums_a, 0, 2 * (size_t)(bin_count(s, period_a) + 1) * sizeof(double));
    if (!index_b) {
        for (int n = 0; n < s->N; ++n) {
            double *a = sums_a + 2 * index_a[n];
            a[0] = fma(s->y[n], s->ivar[n], a[0]);
            a[1] += s->ivar[n];
        }
        return;
    }
    memset(sums_b, 0, 2 * (size_t)(bin_count(s, period_b) + 1) * sizeof(double));
    for (int n = 0; n < s->N; ++n) {
        double *a = sums_a + 2 * index_a[n], *b = sums_b + 2 * index_b[n];
        a[0] = fma(s->y[n], s->ivar[n], a[0]);
        a[1] += s->ivar[n];
        b[0] = fma(s->y[n], s->ivar[n], b[0]);
        b[1] += s->ivar[n];
    }
}

static int batch_search(Series *series, int n_series, int n_periods,
                        const double *periods, int *index_a, int *index_b,
                        double *sums_a, double *sums_b, double *score, double *out)
{
    for (int p = 0; p < n_periods; p += 2) {
        int q = p + 1 < n_periods ? p + 1 : p;
        int *second = q == p ? NULL : index_b;
        int status = batch_indices(series, periods[p], periods[q], index_a, second);
        if (status) return status;
        for (int k = 0; k < n_series; ++k) {
            Series *s = series + k;
            double *result = out + (size_t)k * OUTPUTS * n_periods;
            batch_scatter(s, periods[p], periods[q], index_a, second, sums_a, sums_b);
            cumulate(s, bin_count(s, periods[p]), sums_a);
            search(s, periods[p], sums_a, score, result + p, n_periods);
            if (q == p) continue;
            cumulate(s, bin_count(s, periods[q]), sums_b);
            search(s, periods[q], sums_b, score, result + q, n_periods);
        }
        if (q == p) break;
    }
    return 0;
}

int bls_batch(int N, const double *t, const double *y, int n_series,
              const double *ivar, int n_periods, const double *periods,
              int n_durations, const double *durations, int oversample, double *out)
{
    int status = batch_guard(N, t, y, n_series, ivar, n_periods, periods,
                              n_durations, durations, oversample, out);
    if (status) return status;
    Series series[3] = {0};
    batch_initialize(series, N, t, y, ivar, n_durations, durations, oversample);
    size_t bins;
    status = batch_sizes(N, n_periods, series, periods, &bins);
    if (status) return status;
    double *values = malloc(((size_t)N + 5 * bins) * sizeof(double));
    int *index_a = malloc(2 * (size_t)N * sizeof(int));
    if (!values || !index_a) {
        free(values);
        free(index_a);
        return -1;
    }
    for (int n = 0; n < N; ++n) values[n] = t[n] - series[0].min_t;
    for (int k = 0; k < n_series; ++k) series[k].x = values;
    status = batch_search(series, n_series, n_periods, periods, index_a, index_a + N,
                          values + N, values + N + 2 * bins, values + N + 4 * bins,
                          out);
    free(values);
    free(index_a);
    return status;
}
