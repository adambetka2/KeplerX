// Native transit likelihood: juliet 2.2.10+spacelab.1 with batman 2.5.3+spacelab.2, agreeing to rounding.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <Accelerate/Accelerate.h>

#define PI 3.14159265358979323846
#define G_SI 6.67408e-11
#define LOG2PI 1.8378770664093453
#define REJECTED (-1e101)

typedef struct {
    double period, t0, p, b, rho, q1, q2, ecc, omega_deg, mflux, sigma_w,
        dilution;
} Parameters;

typedef struct {
    double tc, period, a, inc, ecc, omega, p, u1, u2, ld_norm;
    double n, tp, sin_inc, cos_omega, sin_omega, root, circular, mc, step_tol;
    double before, after; /* window half-widths around conjunction (days) */
    int whole_orbit, empty;
} Orbit;

// batman's limb darkening, formulas unchanged.

static double ellpic_bulirsch(double n, double k)
{
    double kc = sqrt(1. - k * k);
    double p = sqrt(n + 1.);
    double m0 = 1.;
    double c = 1.;
    double d = 1. / p;
    double e = kc;
    double f, g;
    int nit = 0;
    while (nit < 10000) {
        double ip = 1. / p;
        f = c;
        c = d * ip + c;
        g = e * ip;
        d = 2. * (f * g + d);
        p = g + p;
        g = m0;
        m0 = kc + m0;
        if (fabs(g - kc) > 1.0e-8 * g) { /* |1 - kc/g| > 1e-8, g >= 1 */
            kc = 2. * sqrt(e);
            e = kc * m0;
        } else
            return 0.5 * PI * (c * m0 + d) / (m0 * (m0 + p));
        nit++;
    }
    return 0;
}

static double ellec(double k)
{
    double m1 = 1.0 - k * k;
    double ee1 = 1.0 + m1 * (0.44325141463 + m1 * (0.06260601220 +
                 m1 * (0.04757383546 + m1 * 0.01736506451)));
    double ee2 = m1 * (0.24998368310 + m1 * (0.09200180037 +
                 m1 * (0.04069697526 + m1 * 0.00526449639))) * log(1.0 / m1);
    return ee1 + ee2;
}

static double ellk(double k)
{
    double m1 = 1.0 - k * k;
    double ek1 = 1.38629436112 + m1 * (0.09666344259 + m1 * (0.03590092383 +
                 m1 * (0.03742563713 + m1 * 0.01451196212)));
    double ek2 = (0.5 + m1 * (0.12498593597 + m1 * (0.06880248576 +
                 m1 * (0.03328355346 + m1 * 0.00441787012)))) * log(m1);
    return ek1 - ek2;
}

static void hastings_log(double m1, double L, double *K, double *E)
{
    *K = 1.38629436112 + m1 * (0.09666344259 + m1 * (0.03590092383 +
         m1 * (0.03742563713 + m1 * 0.01451196212))) -
         (0.5 + m1 * (0.12498593597 + m1 * (0.06880248576 +
         m1 * (0.03328355346 + m1 * 0.00441787012)))) * L;
    *E = 1.0 + m1 * (0.44325141463 + m1 * (0.06260601220 +
         m1 * (0.04757383546 + m1 * 0.01736506451))) -
         m1 * (0.24998368310 + m1 * (0.09200180037 +
         m1 * (0.04069697526 + m1 * 0.00526449639))) * L;
}

static void hastings(double k, double *K, double *E)
{
    double m1 = 1.0 - k * k;
    hastings_log(m1, log(m1), K, E);
}

// batman _quadratic_ld for one separation d; c1, c2 are u1, u2.
static double quadratic_flux(double d, double p, double c1, double c2, double omega)
{
    const double tol = 1.0e-14;
    double kap0 = 0.0, kap1 = 0.0;
    double lambdad = 0.0, lambdae = 0.0, etad = 0.0;
    d = fabs(d);
    if (fabs(p - d) < tol) d = p;
    if (fabs(p - 1.0 - d) < tol) d = p - 1.0;
    if (fabs(1.0 - p - d) < tol) d = 1.0 - p;
    if (d < tol) d = 0.0;
    double x1 = (p - d) * (p - d);
    double x2 = (p + d) * (p + d);
    double x3 = p * p - d * d;
    if (d >= 1.0 + p) return 1.0;
    if (p >= 1.0 && d <= p - 1.0) {
        lambdad = 0.0;
        etad = 0.5;
        lambdae = 1.0;
        return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae +
                      (c1 + 2.0 * c2) * (lambdad + 2.0 / 3.0) + c2 * etad) / omega;
    }
    if (d >= fabs(1.0 - p) && d <= 1.0 + p) {
        kap1 = acos(fmin((1.0 - p * p + d * d) / 2.0 / d, 1.0));
        kap0 = acos(fmin((p * p + d * d - 1.0) / 2.0 / p / d, 1.0));
        lambdae = p * p * kap0 + kap1;
        lambdae = (lambdae - 0.50 * sqrt(fmax(4.0 * d * d -
                   (1.0 + d * d - p * p) * (1.0 + d * d - p * p), 0.0))) / PI;
    }
    if (d == p) {
        if (d < 0.5) {
            double q = 2.0 * p;
            double Kk = ellk(q);
            double Ek = ellec(q);
            lambdad = 1.0 / 3.0 + 2.0 / 9.0 / PI * (4.0 * (2.0 * p * p - 1.0) * Ek +
                      (1.0 - 4.0 * p * p) * Kk);
            etad = p * p / 2.0 * (p * p + 2.0 * d * d);
            return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                          c2 * etad) / omega;
        } else if (d > 0.5) {
            double q = 0.5 / p;
            double Kk = ellk(q);
            double Ek = ellec(q);
            lambdad = 1.0 / 3.0 + 16.0 * p / 9.0 / PI * (2.0 * p * p - 1.0) * Ek -
                      (32.0 * p * p * p * p - 20.0 * p * p + 3.0) / 9.0 / PI / p * Kk;
            etad = 1.0 / 2.0 / PI * (kap1 + p * p * (p * p + 2.0 * d * d) * kap0 -
                   (1.0 + 5.0 * p * p + d * d) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)));
        } else {
            lambdad = 1.0 / 3.0 - 4.0 / PI / 9.0;
            etad = 3.0 / 32.0;
            return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                          c2 * etad) / omega;
        }
        return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                      c2 * etad) / omega;
    }
    if ((d > 0.5 + fabs(p - 0.5) && d < 1.0 + p) ||
        (p > 0.5 && d > fabs(1.0 - p) && d < p)) {
        double q = sqrt((1.0 - x1) / 4.0 / d / p);
        double Kk, Ek;
        hastings(q, &Kk, &Ek);
        double n = 1.0 / x1 - 1.0;
        double Pk = ellpic_bulirsch(n, q);
        lambdad = 1.0 / 9.0 / PI / sqrt(p * d) * (((1.0 - x2) * (2.0 * x2 + x1 - 3.0) -
                  3.0 * x3 * (x2 - 2.0)) * Kk + 4.0 * p * d * (d * d + 7.0 * p * p - 4.0) *
                  Ek - 3.0 * x3 / x1 * Pk);
        if (d < p) lambdad += 2.0 / 3.0;
        etad = 1.0 / 2.0 / PI * (kap1 + p * p * (p * p + 2.0 * d * d) * kap0 -
               (1.0 + 5.0 * p * p + d * d) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)));
        return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                      c2 * etad) / omega;
    }
    if (p <= 1.0 && d <= (1.0 - p)) {
        etad = p * p / 2.0 * (p * p + 2.0 * d * d);
        lambdae = p * p;
        double r1 = 1.0 / (1.0 - x1), rx1 = 1.0 / x1;
        double q = sqrt((x2 - x1) * r1);
        double Kk, Ek;
        hastings(q, &Kk, &Ek);
        double n = x2 * rx1 - 1.0;
        double Pk = ellpic_bulirsch(n, q);
        lambdad = 2.0 / 9.0 / PI * sqrt(r1) * ((1.0 - 5.0 * d * d + p * p +
                  x3 * x3) * Kk + (1.0 - x1) * (d * d + 7.0 * p * p - 4.0) * Ek -
                  3.0 * x3 * rx1 * Pk);
        if (fabs(p + d - 1.0) <= tol)
            lambdad = 2.0 / 3.0 / PI * acos(1.0 - 2.0 * p) - 4.0 / 9.0 / PI *
                      sqrt(p * (1.0 - p)) * (3.0 + 2.0 * p - 8.0 * p * p);
        if (d < p) lambdad += 2.0 / 3.0;
    }
    return 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                  c2 * etad) / omega;
}

// batman's bracketed solver for M already reduced to [-pi, pi].
static double bracketed_E(double M, double e)
{
    double lower = M - e, upper = M + e, E = M;
    double previous_step = upper - lower;
    for (int i = 0; i < 128; i++) {
        double f = E - e * sin(E) - M;
        if (f == 0.) return E;
        if (f > 0.) upper = E;
        else lower = E;
        if (upper - lower <= 8. * DBL_EPSILON * (1. + fabs(E)))
            return lower + (upper - lower) / 2.;
        double proposed = E - f / (1. - e * cos(E));
        if (proposed == E) return E;
        if (!isfinite(proposed) || proposed <= lower || proposed >= upper ||
            fabs(proposed - E) > previous_step / 2.)
            proposed = lower + (upper - lower) / 2.;
        previous_step = fabs(proposed - E);
        E = proposed;
    }
    return lower + (upper - lower) / 2.;
}

// Eccentric anomaly of true anomaly f, branch-continuous through +-pi.
static double eccentric_of_true(double f, double e)
{
    return 2. * atan2(sqrt(1. - e) * sin(f / 2.), sqrt(1. + e) * cos(f / 2.));
}

// Time from conjunction (true anomaly fc) to true anomaly fc + psi.
static double time_from_conjunction(const Orbit *o, double fc, double psi)
{
    if (o->circular) return psi / o->n;
    double e = o->ecc;
    double E1 = eccentric_of_true(fc, e), E2 = eccentric_of_true(fc + psi, e);
    double dE = E2 - E1;
    if (psi > 0 && dE < 0) dE += 2. * PI;
    if (psi < 0 && dE > 0) dE -= 2. * PI;
    double dM = dE - e * (sin(E2) - sin(E1));
    return dM / o->n;
}

// Smallest orbital radius over true anomalies [f0, f1] (f1 - f0 < 2 pi).
static double radius_floor(const Orbit *o, double f0, double f1)
{
    double e = o->ecc;
    double turns = ceil(f0 / (2. * PI));
    double best = fmax(cos(f0), cos(f1));
    if (turns * 2. * PI <= f1) best = 1.;
    return o->a * (1. - e * e) / (1. + e * best);
}

// Arcs where r_min^2 (sin^2 psi sin^2 i + cos^2 i) >= (1 + p)^2 cannot overlap the star.
#define ARCS 64
static void window(Orbit *o)
{
    const double fc = PI / 2. - o->omega;
    const double ci = cos(o->inc), c2 = ci * ci, s2 = o->sin_inc * o->sin_inc;
    const double reach = (1. + o->p) * (1. + o->p) * (1. + 1e-9);
    int first = -1, last = -1;
    o->empty = 0;
    o->whole_orbit = 0;
    for (int k = 0; k < ARCS; k++) {
        double a0 = -PI / 2. + k * PI / ARCS, a1 = a0 + PI / ARCS;
        double r = radius_floor(o, fc + a0, fc + a1) * (1. - 1e-9);
        double s = (a0 <= 0. && a1 >= 0.) ? 0. : fmin(sin(a0) * sin(a0), sin(a1) * sin(a1));
        if (r * r * (s * s2 + c2) * (1. - 1e-9) < reach) {
            if (first < 0) first = k;
            last = k;
        }
    }
    if (first < 0) {
        o->empty = 1;
        return;
    }
    double lo = -PI / 2. + first * PI / ARCS, hi = -PI / 2. + (last + 1) * PI / ARCS;
    // Each side of conjunction has its own radius floor: an eccentric orbit is closer on its periastron side.
    for (int pass = 0; pass < 3; pass++) {
        double r_hi = radius_floor(o, fc + fmin(fmax(lo, 0.), hi), fc + hi);
        double r_lo = radius_floor(o, fc + lo, fc + fmax(fmin(hi, 0.), lo));
        for (int side = 0; side < 2; side++) {
            double ratio = (1. + o->p) / (side ? r_hi : r_lo) * (1. + 1e-9);
            double x = (ratio * ratio - c2) / s2;
            if (!(x < 1.)) continue;
            double bound = asin(sqrt(fmax(x, 0.))) * (1. + 1e-9) + 1e-12;
            if (side) hi = fmin(hi, bound);
            else lo = fmax(lo, -bound);
        }
    }
    if (!(lo < hi)) {
        o->empty = 1;
        return;
    }
    o->before = -time_from_conjunction(o, fc, lo) * (1. + 1e-9) + 1e-9;
    o->after = time_from_conjunction(o, fc, hi) * (1. + 1e-9) + 1e-9;
}

static int setup(const Parameters *q, Orbit *o)
{
    if (q->ecc > 1.) return 0; /* juliet's ecclim */
    double a = pow((q->rho * G_SI * ((q->period * 24. * 3600.) * (q->period * 24. * 3600.))) /
                   (3. * PI), 1. / 3.);
    double ecc_factor = (1. + q->ecc * sin(q->omega_deg * PI / 180.)) / (1. - q->ecc * q->ecc);
    double inc_inv_factor = (q->b / a) * ecc_factor;
    if (q->b > 1. + q->p || inc_inv_factor >= 1.) return 0;
    if (!isfinite(q->ecc) || q->ecc < 0. || q->ecc >= 1.) return -1;
    o->tc = q->t0;
    o->period = q->period;
    o->a = a;
    o->inc = (acos(inc_inv_factor) * 180. / PI) * PI / 180.;
    o->ecc = q->ecc;
    o->omega = q->omega_deg * PI / 180.;
    o->p = fabs(q->p);
    o->u1 = 2. * sqrt(q->q1) * q->q2;
    o->u2 = sqrt(q->q1) * (1. - 2. * q->q2);
    o->ld_norm = 1.0 - o->u1 / 3.0 - o->u2 / 6.0;
    o->n = 2. * PI / o->period;
    o->circular = o->ecc < 1.0e-5;
    double f = PI / 2. - o->omega, M;
    if (o->circular) M = f;
    else {
        double E = 2. * atan(sqrt((1. - o->ecc) / (1. + o->ecc)) * tan(f / 2.));
        M = E - o->ecc * sin(E);
    }
    o->tp = o->tc - o->period * M / 2. / PI;
    o->sin_inc = sin(o->inc);
    o->cos_omega = cos(o->omega);
    o->sin_omega = sin(o->omega);
    o->root = sqrt(1. - o->ecc * o->ecc);
    o->mc = remainder(o->n * (o->tc - o->tp), 2. * PI);
    o->step_tol = fmin(4e-9, sqrt(2e-17 * (1. - o->ecc) / fmax(o->ecc, 1e-300)));
    window(o);
    return 1;
}

typedef struct {
    double E, s, c, M;
    int warm;
    double E1, E2, E3, E4;
    int expanded;
} Kepler;

// sin and cos of |x| <= 0.2 by Taylor terms (truncation below 3e-19).
static inline void small_rotation(double x, double *sx, double *cx)
{
    double x2 = x * x;
    *sx = x * (1. - x2 * (1. / 6.) * (1. - x2 * (1. / 20.) * (1. - x2 * (1. / 42.) *
          (1. - x2 * (1. / 72.) * (1. - x2 * (1. / 110.) * (1. - x2 * (1. / 156.)))))));
    *cx = 1. - x2 * 0.5 * (1. - x2 * (1. / 12.) * (1. - x2 * (1. / 30.) * (1. - x2 *
          (1. / 56.) * (1. - x2 * (1. / 90.) * (1. - x2 * (1. / 132.) * (1. - x2 *
          (1. / 182.)))))));
}

// Taylor terms of E(M) about a solution, with w = 1 - e cos E.
static void expand(Kepler *k, double e)
{
    double w = 1. - e * k->c, iw = 1. / w, iw2 = iw * iw, s = k->s, c = k->c;
    k->E1 = iw;
    k->E2 = -0.5 * e * s * iw2 * iw;
    k->E3 = (3. * e * e * s * s - e * c * w) * iw2 * iw2 * iw * (1. / 6.);
    k->E4 = (e * s * w * w + 10. * e * e * s * c * w - 15. * e * e * e * s * s * s) * iw2 * iw2 *
            iw2 * iw * (1. / 24.);
    k->expanded = 1;
}

// Solve E - e sin E = M from a nearby solution: third-order predictor, rotation, then Newton.
static void solve(const Orbit *o, double M, const Kepler *k0, Kepler *k)
{
    const double e = o->ecc;
    double dM = M - k0->M;
    if (k0->warm && fabs(dM) < 0.15) {
        if (!k0->expanded) expand((Kepler *)k0, e);
        double delta = dM * (k0->E1 + dM * (k0->E2 + dM * (k0->E3 + dM * k0->E4)));
        if (fabs(delta) <= 0.2) {
            double x = k0->E + delta, sd, cd;
            small_rotation(delta, &sd, &cd);
            double s = k0->s * cd + k0->c * sd, c = k0->c * cd - k0->s * sd;
            for (int i = 0; i < 4; i++) {
                double step = (x - e * s - M) / (1. - e * c);
                x -= step;
                if (fabs(step) <= o->step_tol) {
                    k->E = x;
                    k->s = s - step * c;
                    k->c = c + step * s;
                    k->M = M;
                    k->warm = 1;
                    k->expanded = 0;
                    return;
                }
                if (fabs(step) > 0.2) break;
                double ss, cs, s1, c1;
                small_rotation(-step, &ss, &cs);
                s1 = s * cs + c * ss;
                c1 = c * cs - s * ss;
                s = s1;
                c = c1;
            }
        }
    }
    double x = bracketed_E(remainder(M, 2. * PI), e);
    x += M - remainder(M, 2. * PI);
    k->E = x;
    k->s = sin(x);
    k->c = cos(x);
    k->M = M;
    k->warm = 1;
    k->expanded = 0;
}

// Sky separation at time t from the nearest conjunction; 100 behind the star (batman's convention).
static double separation(const Orbit *o, double t, double centre, const Kepler *k0,
                         Kepler *k)
{
    double sf, cf, r;
    if (o->circular) {
        double f = ((t - o->tp) / o->period - (int)((t - o->tp) / o->period)) * 2. * PI;
        sf = sin(f);
        cf = cos(f);
        r = o->a * (1.0 - o->ecc * o->ecc) / (1.0 + o->ecc * cf);
        double su = sf * o->cos_omega + cf * o->sin_omega;
        if (su * o->sin_inc <= 0.) return 100.;
        return r * sqrt(1.0 - su * su * o->sin_inc * o->sin_inc);
    }
    solve(o, o->mc + o->n * (t - centre), k0, k);
    double X = o->a * (k->c - o->ecc), Y = o->a * o->root * k->s;
    r = o->a * (1. - o->ecc * k->c);
    double z = Y * o->cos_omega + X * o->sin_omega; /* r sin(f + omega) */
    if (z <= 0.) return 100.;
    double h = z * o->sin_inc;
    return sqrt(r * r - h * h);
}

// numpy's pairwise sum for n <= 128 contiguous doubles.
static double small_sum(const double *a, int n)
{
    if (n < 8) {
        double res = 0.;
        for (int i = 0; i < n; i++) res += a[i];
        return res;
    }
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    int i;
    for (i = 8; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
}

static double pairwise_sum(const double *a, int64_t n)
{
    if (n <= 128) return small_sum(a, (int)n);
    int64_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(a, half) + pairwise_sum(a + half, n - half);
}

// Bulirsch's Pi (batman's ellpic_bulirsch) for m arguments at once, iterations interleaved.
#define BATCH 1024
static void ellpic_rows(int m, const double *n, const double *k, double *out)
{
    double kc[BATCH], pp[BATCH], m0[BATCH], c[BATCH], dd[BATCH], e[BATCH];
    int live[BATCH];
    for (int b = 0; b < m; b++) {
        kc[b] = sqrt(1. - k[b] * k[b]);
        pp[b] = sqrt(n[b] + 1.);
        m0[b] = 1.;
        c[b] = 1.;
        dd[b] = 1. / pp[b];
        e[b] = kc[b];
        live[b] = 1;
        out[b] = 0; /* batman's non-convergence value */
    }
    int remaining = m;
    for (int it = 0; it < 10000 && remaining; it++) {
        for (int b = 0; b < m; b++) {
            if (!live[b]) continue;
            double ip = 1. / pp[b];
            double f = c[b];
            c[b] = dd[b] * ip + c[b];
            double g = e[b] * ip;
            dd[b] = 2. * (f * g + dd[b]);
            pp[b] = g + pp[b];
            g = m0[b];
            m0[b] = kc[b] + m0[b];
            if (fabs(g - kc[b]) > 1.0e-8 * g) {
                kc[b] = 2. * sqrt(e[b]);
                e[b] = kc[b] * m0[b];
            } else {
                out[b] = 0.5 * PI * (c[b] * m0[b] + dd[b]) / (m0[b] * (m0[b] + pp[b]));
                live[b] = 0;
                remaining--;
            }
        }
    }
}

// batman's fully-inside branch for a batch of separations.
static void inside_fluxes(const Orbit *o, int m, const double *d, double *flux)
{
    const double p = o->p, c1 = o->u1, c2 = o->u2;
    double x1[BATCH], x3[BATCH], r1[BATCH], rx1[BATCH], K[BATCH], E[BATCH];
    double n[BATCH], q[BATCH], Pk[BATCH];
    for (int b = 0; b < m; b++) {
        double x2 = (p + d[b]) * (p + d[b]);
        x1[b] = (p - d[b]) * (p - d[b]);
        x3[b] = p * p - d[b] * d[b];
        r1[b] = 1.0 / (1.0 - x1[b]);
        rx1[b] = 1.0 / x1[b];
        q[b] = sqrt((x2 - x1[b]) * r1[b]);
        K[b] = 1.0 - q[b] * q[b];
        n[b] = x2 * rx1[b] - 1.0;
    }
    vvlog(E, K, &m);
    for (int b = 0; b < m; b++) hastings_log(1.0 - q[b] * q[b], E[b], &K[b], &E[b]);
    ellpic_rows(m, n, q, Pk);
    for (int b = 0; b < m; b++) {
        double db = d[b];
        double etad = p * p / 2.0 * (p * p + 2.0 * db * db);
        double lambdae = p * p;
        double lambdad = 2.0 / 9.0 / PI * sqrt(r1[b]) * ((1.0 - 5.0 * db * db + p * p +
                         x3[b] * x3[b]) * K[b] + (1.0 - x1[b]) * (db * db + 7.0 * p * p - 4.0) *
                         E[b] - 3.0 * x3[b] * rx1[b] * Pk[b]);
        if (db < p) lambdad += 2.0 / 3.0;
        flux[b] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae + (c1 + 2.0 * c2) * lambdad +
                         c2 * etad) / o->ld_norm;
    }
}

// batman's limb-crossing branch (zones 3 and 3.1) for a batch of separations.
static void limb_fluxes(const Orbit *o, int m, const double *d, double *flux)
{
    const double p = o->p, c1 = o->u1, c2 = o->u2;
    double kap0[BATCH], kap1[BATCH], lambdae[BATCH], x1[BATCH], x2[BATCH], K[BATCH];
    double E[BATCH], n[BATCH], q[BATCH], Pk[BATCH];
    for (int b = 0; b < m; b++) {
        double db = d[b];
        kap1[b] = fmin((1.0 - p * p + db * db) / 2.0 / db, 1.0);
        kap0[b] = fmin((p * p + db * db - 1.0) / 2.0 / p / db, 1.0);
    }
    vvacos(kap1, kap1, &m);
    vvacos(kap0, kap0, &m);
    for (int b = 0; b < m; b++) {
        double db = d[b];
        double w = 1.0 + db * db - p * p;
        lambdae[b] = (p * p * kap0[b] + kap1[b] - 0.50 * sqrt(fmax(4.0 * db * db - w * w, 0.0))) / PI;
        x1[b] = (p - db) * (p - db);
        x2[b] = (p + db) * (p + db);
        q[b] = sqrt((1.0 - x1[b]) / 4.0 / db / p);
        K[b] = 1.0 - q[b] * q[b];
        n[b] = 1.0 / x1[b] - 1.0;
    }
    vvlog(E, K, &m);
    for (int b = 0; b < m; b++) hastings_log(1.0 - q[b] * q[b], E[b], &K[b], &E[b]);
    ellpic_rows(m, n, q, Pk);
    for (int b = 0; b < m; b++) {
        double db = d[b], x3 = p * p - db * db;
        double lambdad = 1.0 / 9.0 / PI / sqrt(p * db) * (((1.0 - x2[b]) * (2.0 * x2[b] +
                         x1[b] - 3.0) - 3.0 * x3 * (x2[b] - 2.0)) * K[b] + 4.0 * p * db *
                         (db * db + 7.0 * p * p - 4.0) * E[b] - 3.0 * x3 / x1[b] * Pk[b]);
        if (db < p) lambdad += 2.0 / 3.0;
        double etad = 1.0 / 2.0 / PI * (kap1[b] + p * p * (p * p + 2.0 * db * db) * kap0[b] -
                      (1.0 + 5.0 * p * p + db * db) / 4.0 * sqrt((1.0 - x1[b]) * (x2[b] - 1.0)));
        flux[b] = 1.0 - ((1.0 - c1 - 2.0 * c2) * lambdae[b] + (c1 + 2.0 * c2) * lambdad +
                         c2 * etad) / o->ld_norm;
    }
}

// Snap d as batman does and name its branch: 1 inside, 2 limb-crossing, 0 other.
static int branch(double *d, double p)
{
    const double tol = 1.0e-14;
    double x = fabs(*d);
    if (fabs(p - x) < tol) x = p;
    if (fabs(p - 1.0 - x) < tol) x = p - 1.0;
    if (fabs(1.0 - p - x) < tol) x = 1.0 - p;
    if (x < tol) x = 0.0;
    *d = x;
    if (!(p < 1.0) || x == p) return 0;
    if (x < 1.0 - p && fabs(p + x - 1.0) > tol) return 1;
    if (x > 1.0 - p && x < 1.0 + p && (x > 0.5 + fabs(p - 0.5) || (p > 0.5 && x < p)))
        return 2;
    return 0;
}

// Exposure-averaged model flux for photons [start, stop) near `centre`.
static void photon_models(const Orbit *o, int64_t start, int64_t stop, const double *t,
                          double centre, int ss, const double *off, double *model,
                          double *scratch)
{
    Kepler chain = {0., 0., 1., 0., 0, 0., 0., 0., 0., 0}, k;
    const double edge = 1.0 + o->p;
    // Samples are solved in groups under 0.15 rad of mean anomaly, each group chained from the previous.
    const double spacing = ss > 1 ? o->n * (off[ss - 1] - off[0]) / (ss - 1) : 0.;
    int group = ss;
    if (spacing * ss >= 0.15) group = (int)fmax(1., floor(0.15 / spacing) - 1.);
    double inside[BATCH], limb[BATCH], fluxes[BATCH];
    int slot[BATCH], limb_slot[BATCH];
    for (int64_t i = start; i < stop; i++) {
        int any = 0;
        for (int g0 = 0; g0 < ss; g0 += group) {
            int g1 = g0 + group < ss ? g0 + group : ss, mid = (g0 + g1) / 2;
            scratch[mid] = separation(o, off[mid] + t[i], centre, &chain, &k);
            chain = k;
            for (int j = g0; j < g1; j++)
                if (j != mid) scratch[j] = separation(o, off[j] + t[i], centre, &chain, &k);
        }
        for (int j = 0; j < ss; j++) any |= scratch[j] < edge;
        if (!any) {
            model[i] = 1.0;
            continue;
        }
        int m = 0, l = 0;
        for (int j = 0; j < ss; j++) {
            if (scratch[j] >= edge) {
                scratch[j] = 1.0;
                continue;
            }
            double d = scratch[j];
            int kind = branch(&d, o->p);
            if (kind == 1) {
                inside[m] = d;
                slot[m++] = j;
            } else if (kind == 2) {
                limb[l] = d;
                limb_slot[l++] = j;
            } else
                scratch[j] = quadratic_flux(scratch[j], o->p, o->u1, o->u2, o->ld_norm);
        }
        inside_fluxes(o, m, inside, fluxes);
        for (int b = 0; b < m; b++) scratch[slot[b]] = fluxes[b];
        limb_fluxes(o, l, limb, fluxes);
        for (int b = 0; b < l; b++) scratch[limb_slot[b]] = fluxes[b];
        any = 0;
        for (int j = 0; j < ss; j++) any |= scratch[j] != 1.0;
        model[i] = any ? small_sum(scratch, ss) / ss : 1.0;
    }
}

// Model every photon whose exposure can reach the stellar disc; 1 elsewhere.
static void transit_photons(const Orbit *o, int64_t n, const double *t, double half_span,
                            double *model, int ss, const double *off, double *scratch,
                            int64_t *evaluated)
{
    for (int64_t i = 0; i < n; i++) model[i] = 1.0;
    *evaluated = 0;
    if (n == 0 || o->empty) return;
    if (o->whole_orbit) {
        for (int64_t i = 0; i < n; i++) {
            double centre = o->tc + rint((t[i] - o->tc) / o->period) * o->period;
            photon_models(o, i, i + 1, t, centre, ss, off, model, scratch);
        }
        *evaluated = n;
        return;
    }
    double lo_edge = o->before + half_span, hi_edge = o->after + half_span;
    long k0 = (long)floor((t[0] - o->tc - hi_edge) / o->period) - 1;
    long k1 = (long)ceil((t[n - 1] - o->tc + lo_edge) / o->period) + 1;
    int64_t i = 0, done = 0;
    for (long k = k0; k <= k1 && i < n; k++) {
        double centre = o->tc + k * o->period;
        while (i < n && t[i] < centre - lo_edge) i++;
        int64_t i0 = i;
        while (i < n && t[i] <= centre + hi_edge) i++;
        if (i > i0) {
            photon_models(o, i0, i, t, centre, ss, off, model, scratch);
            done += i - i0;
        }
    }
    *evaluated = done;
}

// Public: one log-likelihood, juliet's convention (rejections are -1e101).
double validator_loglike(int64_t n, const double *t, const double *y, const double *e2,
                         int32_t ss, const double *off, double half_span,
                         const double *theta, double *model, double *terms,
                         int64_t *evaluated)
{
    Parameters q = {theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
                    theta[6], theta[7], theta[8], theta[9], theta[10], theta[11]};
    Orbit o;
    double scratch[1024];
    *evaluated = 0;
    if (ss > BATCH) return NAN;
    int status = setup(&q, &o);
    if (status == 0) return REJECTED;
    if (status < 0) return NAN;
    transit_photons(&o, n, t, half_span, model, ss, off, scratch, evaluated);
    const double D = q.dilution;
    const double scale = 1. / (1. + D * q.mflux);
    const double jitter = (q.sigma_w * 1e-6) * (q.sigma_w * 1e-6);
    for (int64_t i = 0; i < n; i++) {
        double m = (model[i] * D + (1. - D)) * scale;
        double r = y[i] - m;
        double tau = 1. / (e2[i] + jitter);
        terms[i] = -log(tau) + tau * (r * r);
    }
    double value = -0.5 * ((double)n * LOG2PI + pairwise_sum(terms, n));
    if (isnan(value)) return REJECTED;
    return value;
}

// Public: the model flux only (exposure-averaged, before dilution/offset).
int validator_model(int64_t n, const double *t, int32_t ss, const double *off,
                    double half_span, const double *theta, double *model,
                    int64_t *evaluated, int32_t everywhere)
{
    Parameters q = {theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
                    theta[6], theta[7], theta[8], theta[9], theta[10], theta[11]};
    Orbit o;
    double scratch[1024];
    if (ss > BATCH) return -2;
    int status = setup(&q, &o);
    if (status <= 0) return status;
    if (everywhere) {
        o.whole_orbit = 1;
        o.empty = 0;
    }
    transit_photons(&o, n, t, half_span, model, ss, off, scratch, evaluated);
    return 1;
}

// Public: many log-likelihoods (rows of 12 parameters).
void validator_loglike_rows(int64_t rows, const double *theta, int64_t n, const double *t,
                            const double *y, const double *e2, int32_t ss,
                            const double *off, double half_span, double *model,
                            double *terms, double *out, int64_t *evaluated)
{
    for (int64_t j = 0; j < rows; j++)
        out[j] = validator_loglike(n, t, y, e2, ss, off, half_span, theta + 12 * j, model,
                                   terms, evaluated + j);
}
