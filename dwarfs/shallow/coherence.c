/* Optional native membership metadata; bls.c and its seven outputs are unchanged. */
#include "../../pipeline/shallow/bls.c"

static void selected_bins(const Series *s, double period, const double *score,
                          const double *out, int stride, long long *metadata)
{
    metadata[0] = metadata[1] = -1;
    if (!isfinite(out[OBJECTIVE * stride])) return;
    int dur = (int)round(s->durations[0] / s->bin_duration);
    for (int n = 0; n <= bin_count(s, period) - dur; ++n) {
        if (score[n] == out[OBJECTIVE * stride]) {
            metadata[0] = n;
            metadata[1] = dur;
            return;
        }
    }
}

int bls_metadata(int N, const double *t, const double *y, const double *ivar,
                 int n_periods, const double *periods, int n_durations,
                 const double *durations, int oversample, double *out,
                 long long *metadata)
{
    if (N < 1 || n_periods < 1 || n_durations != 1 || oversample < 1) return 3;
    double min_period = least(periods, n_periods);
    double min_duration = durations[0];
    if (min_period < DBL_EPSILON) return 1;
    if (min_duration > min_period || min_duration < DBL_EPSILON) return 2;
    Series s = {.N = N, .n_durations = n_durations, .oversample = oversample, .y = y,
                .ivar = ivar, .durations = durations, .min_t = INFINITY,
                .bin_duration = min_duration / (double)oversample};
    size_t bins = (size_t)bin_count(&s, greatest(periods, n_periods)) + 1;
    double *values = malloc(((size_t)N + 5 * bins) * sizeof(double));
    int *index_a = malloc(2 * (size_t)N * sizeof(int));
    if (!values || !index_a) { free(values); free(index_a); return -1; }
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
        bin_pair(&s, periods[p], periods[q], index_a, index_b, sums_a, sums_b);
        cumulate(&s, bin_count(&s, periods[p]), sums_a);
        search(&s, periods[p], sums_a, score, out + p, n_periods);
        selected_bins(&s, periods[p], score, out + p, n_periods, metadata + 2 * p);
        if (q == p) break;
        cumulate(&s, bin_count(&s, periods[q]), sums_b);
        search(&s, periods[q], sums_b, score, out + q, n_periods);
        selected_bins(&s, periods[q], score, out + q, n_periods, metadata + 2 * q);
    }
    free(values);
    free(index_a);
    return 0;
}

static int key_lower(int N, const double *x, double period, double delta,
                     long long cycle, int bin)
{
    int lower = 0, upper = N;
    while (lower < upper) {
        int middle = lower + (upper - lower) / 2;
        double q = floor(x[middle] / period);
        int b = (int)(fma(-period, q, x[middle]) / delta) + 1;
        if (q < cycle || (q == cycle && b < bin)) lower = middle + 1;
        else upper = middle;
    }
    return lower;
}

static void window_sums(int N, const double *x, const double *weight,
                        const double *response, double period, double delta,
                        long long cycle, int first, int last, double *w, double *r)
{
    if (first > last) return;
    int lower = key_lower(N, x, period, delta, cycle, first);
    int upper = key_lower(N, x, period, delta, cycle, last + 1);
    double amount = weight[upper] - weight[lower];
    if (upper > lower && (!isfinite(amount) || amount <= 0.0)) { *w = NAN; return; }
    *w += amount;
    *r += response[upper] - response[lower];
}

static void one_event(int N, const double *x, const double *weight,
                      const double *response, double period, double delta,
                      long long event, int start, int duration, int oversample,
                      double *w, double *r)
{
    int M = (int)ceil(period / delta), stop = start + duration;
    int a = start + 1, b = stop < M - 1 ? stop : M - 1;
    window_sums(N, x, weight, response, period, delta, event, a, b, w, r);
    a = start - M + 2 > 1 ? start - M + 2 : 1;
    b = stop - M + 1 < oversample ? stop - M + 1 : oversample;
    window_sums(N, x, weight, response, period, delta, event + 1, a, b, w, r);
}

static void event_statistics(int N, const double *x, const double *weight,
                             const double *response, double period, double delta,
                             int start, int duration, int oversample, double native_snr,
                             double *out)
{
    for (int k = 0; k < 7; ++k) out[k] = NAN;
    out[1] = 0.0;
    long long last = (long long)floor(x[N - 1] / period), count = 0;
    double inside = 0.0, signal = 0.0, mean = 0.0, scatter = 0.0;
    for (long long event = -1; event <= last; ++event) {
        double w = 0.0, r = 0.0;
        one_event(N, x, weight, response, period, delta, event, start, duration,
                  oversample, &w, &r);
        if (!isfinite(w) || !isfinite(r)) { out[3] = count; out[6] = 5.0; return; }
        if (w == 0.0) continue;
        ++count;
        double next = inside + w, mu = r / w, difference = mu - mean;
        scatter += difference * difference * inside * w / next;
        mean += difference * w / next;
        inside = next;
        signal += r;
    }
    out[3] = (double)count;
    double outside = weight[N] - inside;
    out[6] = count == 0 ? 1.0 : outside <= 0.0 ? 2.0 : 0.0;
    if (out[6] != 0.0) return;
    double depth = (response[N] - signal) / outside - signal / inside;
    double information = inside * outside / weight[N];
    out[0] = depth * sqrt(information);
    out[2] = scatter;
    out[4] = information;
    out[5] = depth;
    out[6] = count == 1 ? 3.0 : count == 2 ? 4.0 : 0.0;
    if (count >= 3)
        out[1] = fmax(0.0, native_snr) / sqrt(fmax(1.0, scatter / (count - 1)));
}

int rank_events(int N, const double *x, const double *weight, const double *response,
                int count, const double *periods, const long long *metadata,
                double delta, int oversample, const double *native_snr, double *out)
{
    if (N < 1 || count < 0 || delta <= 0.0 || oversample < 1) return 3;
    for (int p = 0; p < count; ++p) {
        int start = (int)metadata[2 * p], duration = (int)metadata[2 * p + 1];
        if (start < 0 || duration < 1 || !isfinite(periods[p])) return 3;
        event_statistics(N, x, weight, response, periods[p], delta, start, duration,
                         oversample, native_snr[p], out + 7 * p);
    }
    return 0;
}
