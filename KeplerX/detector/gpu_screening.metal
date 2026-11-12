// GPU bounds for the period search (screening only); the CPU resolves every period that could pass, bitwise. No fast math.

#include <metal_stdlib>
using namespace metal;

#define FOLD_DURATIONS 16
#define FOLD_INTS 7680   // threadgroup histogram entries (30 KB)
#define FOLD_GROUP 32    // periods per fold group at most
#define WINDOW_LANES 16  // periods per window tile at most
#define WINDOW_PAD 2e-3f // days added to each side of a tile's epoch range

// Durations of the tile or group (1..16); pipelines are specialised by it.
constant int DURATIONS [[function_constant(0)]];

struct Tile {
    int period;   // first period of the tile (search order)
    int lanes;    // consecutive periods in the tile
    int bins;     // consecutive bins in the tile
    int b0;       // first bin
    int d0;       // first duration (absolute template index)
    int m;        // durations in this chunk
    int partial;  // slot of (first period, d0) for this bin tile
    int tiles;    // bin tiles of the period group (partials per cell)
};

struct Group {
    int period;   // first period of the group (search order)
    int count;    // consecutive periods in the group
    int rows;     // histogram bins per period (the first period's bins)
    int d0;       // first duration; every period has the same durations
    int cell;     // cell of (first period, d0); each period's cells follow
    int m;        // durations
    int pad0, pad1;
};

struct Period {
    int nb;       // phase bins
    int epochs;   // floor(max(t) / P) + 1
    float ph, pl; // period as a double-float
    float wh, wl; // P / nbins as a double-float
};

struct Params {
    int n;          // cadences
    int table_size; // entries in the time table
    float grid;     // time-table step, days
    float edge;     // days around a bin edge where membership is uncertain
    float n_trans;  // bins with fewer cadences have MES 0
    int n_dur;      // durations per cadence row of (numerator, denominator)
    float threshold; // largest float at most the search threshold (or -inf)
};

struct Top {
    float best;   // largest upper bound
    float second; // second-largest upper bound
    float low;    // largest lower bound
    int arg;      // bin of best
};

static inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

static inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    return float2(s, b - (s - a));
}

static inline float2 ff_add(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    float2 t = two_sum(a.y, b.y);
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    return quick_two_sum(s.x, s.y);
}

// Double-float sum without renormalisation: error about 2^-47, enough for the 1e-6 d screen.
static inline float2 ff_add_sloppy(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    s.y += a.y + b.y;
    return quick_two_sum(s.x, s.y);
}

static inline float2 ff_mul_f(float2 a, float b) {
    float p = a.x * b;
    float e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return quick_two_sum(p, e);
}

static inline Top empty() {
    Top t;
    t.best = -1.0f;
    t.second = -1.0f;
    t.low = 0.0f;
    t.arg = -1;
    return t;
}

// a holds lower bins than b: on equal bounds a keeps the best slot.
static inline Top merge(Top a, Top b) {
    Top r;
    if (b.best > a.best) {
        r.best = b.best;
        r.arg = b.arg;
        r.second = max(a.best, b.second);
    } else {
        r.best = a.best;
        r.arg = a.arg;
        r.second = max(b.best, a.second);
    }
    r.low = max(a.low, b.low);
    return r;
}

// One bin's bounds from its exact integer sums over `count` cadences.
static inline Top bin_bounds(int count, int sn, int sd, float factor, float n_trans, int b) {
    Top here = empty();
    if (count >= 4096) {
        here.best = INFINITY;
        here.arg = b;
        return here;
    }
    float upper = 0.0f, lower = 0.0f;
    if ((float)count >= n_trans) {
        float c = (float)count, fn = (float)sn, fd = (float)sd;
        float en = 0.5001f * c + fabs(fn) * 1.2e-7f + 1.0f;
        float ed = 0.5001f * c + fd * 1.2e-7f + 1.0f;
        float hi = fn + en, lo = fn - en;
        if (hi > 0.0f) upper = hi * rsqrt(max(fd - ed, 1e-37f)) * factor * 1.00001f;
        if (lo > 0.0f) lower = lo * rsqrt(fd + ed) * factor * 0.99999f;
    }
    here.best = upper;
    here.low = lower;
    here.arg = b;
    return here;
}

// bin_bounds' upper bound alone (the same expression).
static inline float bin_upper(int count, int sn, int sd, float factor, float n_trans) {
    if (count >= 4096) return INFINITY;
    float upper = 0.0f;
    if ((float)count >= n_trans) {
        float c = (float)count, fn = (float)sn, fd = (float)sd;
        float en = 0.5001f * c + fabs(fn) * 1.2e-7f + 1.0f;
        float ed = 0.5001f * c + fd * 1.2e-7f + 1.0f;
        float hi = fn + en;
        if (hi > 0.0f) upper = hi * rsqrt(max(fd - ed, 1e-37f)) * factor * 1.00001f;
    }
    return upper;
}

// One cell's Top over `count` consecutive histogram rows, by one SIMD group (result in lane 0).
static inline Top reduce_cell(threadgroup atomic_uint *hist, int row0, int width, int j,
                              int count, int valid, int b0, float factor, float n_trans,
                              float tau, uint lane) {
    float best = -1.0f;
    int arg = -1;
    for (int c = 0; c < count; c += 32) {
        int r = c + (int)lane;
        float u = -1.0f;
        if (r < count && r < valid) {
            int base = (row0 + r) * width;
            int n = (int)atomic_load_explicit(&hist[base], memory_order_relaxed);
            int sn = as_type<int>(atomic_load_explicit(&hist[base + 1 + 2 * j], memory_order_relaxed));
            int sd = as_type<int>(atomic_load_explicit(&hist[base + 2 + 2 * j], memory_order_relaxed));
            u = bin_upper(n, sn, sd, factor, n_trans);
        }
        float m = simd_max(u);
        if (m > best) {
            best = m;
            arg = b0 + c + (int)simd_min(u == m ? lane : 32u);
        }
    }
    Top acc = empty();
    if (!(best > tau)) {
        acc.best = best;
        acc.second = best;
        acc.low = 0.0f;
        acc.arg = arg;
        return acc;
    }
    for (int c = 0; c < count; c += 32) {
        int r = c + (int)lane;
        Top t = empty();
        if (r < count && r < valid) {
            int base = (row0 + r) * width;
            int n = (int)atomic_load_explicit(&hist[base], memory_order_relaxed);
            int sn = as_type<int>(atomic_load_explicit(&hist[base + 1 + 2 * j], memory_order_relaxed));
            int sd = as_type<int>(atomic_load_explicit(&hist[base + 2 + 2 * j], memory_order_relaxed));
            t = bin_bounds(n, sn, sd, factor, n_trans, b0 + r);
        }
        // Lane i merges lane i + offset, the same cell's higher bins.
        for (ushort offset = 16; offset >= 1; offset /= 2) {
            Top o;
            o.best = simd_shuffle_down(t.best, offset);
            o.second = simd_shuffle_down(t.second, offset);
            o.low = simd_shuffle_down(t.low, offset);
            o.arg = simd_shuffle_down(t.arg, offset);
            if (lane + offset < 32) t = merge(t, o);
        }
        acc = merge(acc, t);
    }
    return acc;
}

// Combine each cell's partials (one per bin tile) in bin order.
kernel void merge_partials(device const float4 *partials [[buffer(0)]],
                           device const int *first [[buffer(1)]],
                           device const int *length [[buffer(2)]],
                           device float4 *cells [[buffer(3)]],
                           constant int &count [[buffer(4)]],
                           uint c [[thread_position_in_grid]]) {
    if ((int)c >= count) return;
    Top t = empty();
    for (int i = 0; i < length[c]; i++) {
        float4 v = partials[first[c] + i];
        Top o;
        o.best = v.x;
        o.low = v.y;
        o.second = v.z;
        o.arg = as_type<int>(v.w);
        t = merge(t, o);
    }
    cells[c] = float4(t.best, t.low, t.second, as_type<float>(t.arg));
}

// The bin (k, b) of a cadence at time t, or -1 outside every bin; `ambiguous` within `edge` of a bin edge.
static inline int fold_bin(float2 t, float2 P, float2 w, float inverse_period,
                           float inverse_width, int nb, float edge, thread bool &ambiguous) {
    float k = floor(t.x * inverse_period);
    float2 r = ff_add(t, -ff_mul_f(P, k));
    if (r.x + r.y < 0.0f) {
        k -= 1.0f;
        r = ff_add(t, -ff_mul_f(P, k));
    } else if ((r.x - P.x) + (r.y - P.y) >= 0.0f) {
        k += 1.0f;
        r = ff_add(t, -ff_mul_f(P, k));
    }
    int b = clamp((int)floor(r.x * inverse_width), 0, nb - 1);
    float2 rel = ff_add(r, -ff_mul_f(w, (float)b));
    float before = rel.x + rel.y;                    // days after the bin start
    float after = (rel.x - w.x) + (rel.y - w.y);     // days after the bin end
    if (before < 0.0f) {
        b -= 1;
        after = before;
        before = (rel.x + w.x) + (rel.y + w.y);
    } else if (after >= 0.0f) {
        b += 1;
        before = after;
        after = (rel.x - 2.0f * w.x) + (rel.y - 2.0f * w.y);
    }
    bool origin = t.x == 0.0f && t.y == 0.0f;
    if (!origin && (fabs(before) < edge || fabs(after) < edge)) ambiguous = true;
    if (b < 0 || b >= nb || before < 0.0f || after >= 0.0f) {
        ambiguous = true;
        return -1;
    }
    return b;
}

// Guarded interior test before fold_bin (docs/keplerX-2026-10-28-phase-fold-fma-theory.md).
static inline float fold_margin(float ph, float wh, float edge) {
    return (0x1p-20f * (ph + wh) + 4.0f * edge) * (1.0f + 0x1p-20f);  // 16u(P + w) + 4 edge
}

static inline int interior_bin(float2 t, float2 P, float2 w, float inverse_period,
                               float inverse_width, int nb, float margin) {
    float k = floor(t.x * inverse_period);
    float rh = fma(-k, P.x, t.x);
    float rl = fma(-k, P.y, t.y);
    float b = floor((rh + rl) * inverse_width);
    float d = fma(-b, w.x, rh) + fma(-b, w.y, rl);
    return b >= 0.0f && b < (float)nb && d > margin && d < w.x - margin ? (int)b : -1;
}

kernel void fold(device const float2 *times [[buffer(0)]],
                 device const int4 *pairs [[buffer(2)]],
                 device const Group *groups [[buffer(4)]],
                 device const Period *periods [[buffer(5)]],
                 device atomic_int *flags [[buffer(7)]],
                 constant Params &prm [[buffer(8)]],
                 device const float *factors [[buffer(9)]],
                 device float4 *cells [[buffer(10)]],
                 uint g [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint threads [[threads_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]],
                 uint nsimd [[simdgroups_per_threadgroup]]) {
    threadgroup atomic_uint hist[FOLD_INTS];
    threadgroup atomic_int unsure[FOLD_GROUP];
    threadgroup float4 shape[FOLD_GROUP];    // (P, w) as double-floats
    threadgroup float2 inverse[FOLD_GROUP];  // float 1 / P and nbins / P
    threadgroup int bins[FOLD_GROUP];
    threadgroup float margin[FOLD_GROUP];
    Group group = groups[g];
    const int width = 1 + 2 * DURATIONS;
    for (int x = (int)tid; x < group.count * group.rows * width; x += (int)threads)
        atomic_store_explicit(&hist[x], 0u, memory_order_relaxed);
    for (int x = (int)tid; x < group.count; x += (int)threads) {
        atomic_store_explicit(&unsure[x], 0, memory_order_relaxed);
        Period per = periods[group.period + x];
        shape[x] = float4(per.ph, per.pl, per.wh, per.wl);
        inverse[x] = float2(1.0f / per.ph, (float)per.nb / per.ph);
        bins[x] = per.nb;
        margin[x] = fold_margin(per.ph, per.wh, prm.edge);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int fold_base = ((group.d0 & 1) * (prm.n_dur / 2 + 1) + (group.d0 >> 1)) * prm.n;
    for (int i = (int)tid; i < prm.n; i += (int)threads) {
        float2 t = times[i];
        device const int4 *word = pairs + fold_base + i;
        int2 v[FOLD_DURATIONS + 1];
        for (int j = 0; j < DURATIONS; j += 2) {
            int4 block = word[(j / 2) * prm.n];
            v[j] = block.xy;
            v[j + 1] = block.zw;
        }
        for (int q = 0; q < group.count; q++) {
            float4 pw = shape[q];
            float2 inv = inverse[q];
            int b = interior_bin(t, pw.xy, pw.zw, inv.x, inv.y, bins[q], margin[q]);
            if (b < 0) {
                bool ambiguous = false;
                b = fold_bin(t, pw.xy, pw.zw, inv.x, inv.y, bins[q], prm.edge, ambiguous);
                if (ambiguous) atomic_store_explicit(&unsure[q], 1, memory_order_relaxed);
                if (b < 0) continue;
            }
            int base = (q * group.rows + b) * width;
            atomic_fetch_add_explicit(&hist[base], 1u, memory_order_relaxed);
            for (int j = 0; j < DURATIONS; j++) {
                atomic_fetch_add_explicit(&hist[base + 1 + 2 * j], as_type<uint>(v[j].x), memory_order_relaxed);
                atomic_fetch_add_explicit(&hist[base + 2 + 2 * j], as_type<uint>(v[j].y), memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = (int)tid; x < group.count; x += (int)threads)
        if (atomic_load_explicit(&unsure[x], memory_order_relaxed))
            atomic_store_explicit(&flags[group.period + x], 1, memory_order_relaxed);
    for (int x = (int)simd; x < group.count * DURATIONS; x += (int)nsimd) {
        int q = x / DURATIONS, j = x % DURATIONS;
        Top t = reduce_cell(hist, q * group.rows, width, j, bins[q], bins[q], 0,
                            factors[group.d0 + j], prm.n_trans, prm.threshold, lane);
        if (lane == 0)
            cells[group.cell + q * DURATIONS + j] = float4(t.best, t.low, t.second, as_type<float>(t.arg));
    }
}

kernel void window_fold(device const float2 *times [[buffer(0)]],
                        device const int *table [[buffer(1)]],
                        device const int4 *pairs [[buffer(2)]],
                        device const Tile *tiles [[buffer(4)]],
                        device const Period *periods [[buffer(5)]],
                        device float4 *partials [[buffer(6)]],
                        device atomic_int *flags [[buffer(7)]],
                        constant Params &prm [[buffer(8)]],
                        device const float *factors [[buffer(9)]],
                        threadgroup atomic_uint *hist [[threadgroup(0)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint nsimd [[simdgroups_per_threadgroup]]) {
    threadgroup atomic_int unsure[WINDOW_LANES];
    threadgroup float4 shape[WINDOW_LANES];    // (P, w) as double-floats
    threadgroup float2 inverse[WINDOW_LANES];  // float 1 / P and nbins / P
    threadgroup int bins[WINDOW_LANES];
    threadgroup float margin[WINDOW_LANES];
    threadgroup int epochs_of[WINDOW_LANES];
    Tile tile = tiles[g];
    const int width = 1 + 2 * DURATIONS;
    const int G = tile.lanes, R = tile.bins, b0 = tile.b0;
    for (int x = (int)tid; x < G * R * width; x += (int)threads)
        atomic_store_explicit(&hist[x], 0u, memory_order_relaxed);
    for (int x = (int)tid; x < G; x += (int)threads) {
        Period per = periods[tile.period + x];
        shape[x] = float4(per.ph, per.pl, per.wh, per.wl);
        inverse[x] = float2(1.0f / per.ph, (float)per.nb / per.ph);
        bins[x] = per.nb;
        margin[x] = fold_margin(per.ph, per.wh, prm.edge);
        epochs_of[x] = per.epochs;
        atomic_store_explicit(&unsure[x], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float pmin = INFINITY, pmax = 0.0f, wmin = INFINITY, wmax = 0.0f;
    int epochs = 0;
    for (int q = 0; q < G; q++) {
        pmin = min(pmin, shape[q].x);
        pmax = max(pmax, shape[q].x);
        wmin = min(wmin, shape[q].z);
        wmax = max(wmax, shape[q].z);
        epochs = max(epochs, epochs_of[q]);
    }
    // Float epoch ranges err by under 3e-4 d below 4096 d; WINDOW_PAD keeps each range a superset.
    const float lo0 = (float)b0 * wmin - WINDOW_PAD;
    const float hi0 = (float)(b0 + R) * wmax + WINDOW_PAD;
    const float inverse_grid = 1.0f / prm.grid;
    const int last = prm.table_size - 1;
    const int base_pair = ((tile.d0 & 1) * (prm.n_dur / 2 + 1) + (tile.d0 >> 1)) * prm.n;
    // Each SIMD group finds 32 epochs' cadence ranges at once; the tile's threads share their cadences.
    for (int k0 = 0; k0 <= epochs; k0 += 32) {
        int k = k0 + (int)lane, L = 0, count = 0;
        if (k <= epochs) {
            float fk = (float)k;
            L = table[clamp((int)floor(fma(fk, pmin, lo0) * inverse_grid) - 1, 0, last)];
            int H = table[clamp((int)floor(fma(fk, pmax, hi0) * inverse_grid) + 2, 0, last)];
            int L1 = table[clamp((int)floor(fma(fk + 1.0f, pmin, lo0) * inverse_grid) - 1, 0, last)];
            count = max(0, min(H, L1) - L);
        }
        int offset = simd_prefix_exclusive_sum(count);
        int total = simd_sum(count);
        for (int first = (int)simd * 32; first < total; first += (int)nsimd * 32) {
            // Position p's epoch is the last lane whose offset is at most p (shuffles in uniform flow).
            int p = first + (int)lane, e = 0;
            for (int step = 16; step >= 1; step /= 2)
                if (simd_shuffle(offset, (ushort)(e + step)) <= p) e += step;
            int i = simd_shuffle(L, (ushort)e) + (p - simd_shuffle(offset, (ushort)e));
            if (p >= total) continue;
            float2 t = times[i];
            device const int4 *word = pairs + base_pair + i;
            int2 v[FOLD_DURATIONS + 1];
            for (int j = 0; j < DURATIONS; j += 2) {
                int4 block = word[(j / 2) * prm.n];
                v[j] = block.xy;
                v[j + 1] = block.zw;
            }
            for (int q = 0; q < G; q++) {
                float4 pw = shape[q];
                float2 inv = inverse[q];
                int b = interior_bin(t, pw.xy, pw.zw, inv.x, inv.y, bins[q], margin[q]);
                if (b < 0) {
                    bool ambiguous = false;
                    b = fold_bin(t, pw.xy, pw.zw, inv.x, inv.y, bins[q], prm.edge, ambiguous);
                    if (ambiguous) atomic_store_explicit(&unsure[q], 1, memory_order_relaxed);
                    if (b < 0) continue;
                }
                int r = b - b0;
                if (r < 0 || r >= R) continue;
                int base = (q * R + r) * width;
                atomic_fetch_add_explicit(&hist[base], 1u, memory_order_relaxed);
                for (int j = 0; j < DURATIONS; j++) {
                    atomic_fetch_add_explicit(&hist[base + 1 + 2 * j], as_type<uint>(v[j].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&hist[base + 2 + 2 * j], as_type<uint>(v[j].y), memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = (int)tid; x < G; x += (int)threads)
        if (atomic_load_explicit(&unsure[x], memory_order_relaxed))
            atomic_store_explicit(&flags[tile.period + x], 1, memory_order_relaxed);
    for (int cell = (int)simd; cell < G * DURATIONS; cell += (int)nsimd) {
        int q = cell / DURATIONS, j = cell - q * DURATIONS;
        Top acc = reduce_cell(hist, q * R, width, j, R, bins[q] - b0, b0,
                              factors[tile.d0 + j], prm.n_trans, prm.threshold, lane);
        if (lane == 0)
            partials[tile.partial + (q * DURATIONS + j) * tile.tiles] =
                float4(acc.best, acc.low, acc.second, as_type<float>(acc.arg));
    }
}
