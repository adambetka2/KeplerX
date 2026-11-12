/* Adapted from batman's _rsky.c (Laura Kreidberg).
 * The reviewed Spacelab.2 Kepler solver is unchanged.
 */

#define M_PI 3.14159265358979323846
#define DBL_EPSILON 2.220446049250313080847263336181640625e-16
#define isfinite(x) __builtin_isfinite(x)

extern double sin(double), cos(double), atan(double), tan(double), sqrt(double);
extern double remainder(double, double), fabs(double);

static double getE(double M, double e)
{
    M = remainder(M, 2.*M_PI);
    double lower = M - e, upper = M + e, E = M;
    double previous_step = upper - lower;
    for (int i = 0; i < 128; i++) {
        double f = E - e*sin(E) - M;
        if (f == 0.) return E;
        if (f > 0.) upper = E;
        else lower = E;
        if (upper - lower <= 8.*DBL_EPSILON*(1. + fabs(E)))
            return lower + (upper - lower)/2.;
        double proposed = E - f/(1. - e*cos(E));
        if (proposed == E) return E;
        if (!isfinite(proposed) || proposed <= lower || proposed >= upper ||
            fabs(proposed - E) > previous_step/2.)
            proposed = lower + (upper - lower)/2.;
        previous_step = fabs(proposed - E);
        E = proposed;
    }
    return lower + (upper - lower)/2.;
}

void sky(const double *time, double *output, int count, double tc, double per,
         double a, double inc, double ecc, double omega, int transit, int f_only)
{
    const double n = 2.*M_PI/per;
    const double f0 = M_PI/2. - omega;
    double M0;
    if (ecc < 1.0e-5) M0 = f0;
    else {
        const double E0 = 2.*atan(sqrt((1. - ecc)/(1. + ecc))*tan(f0/2.));
        M0 = E0 - ecc*sin(E0);
    }
    const double tp = tc - per*M0/2./M_PI;
    const double ratio = sqrt((1. + ecc)/(1. - ecc));
    const double si = sin(inc);
    for (int i = 0; i < count; i++) {
        const double t = time[i];
        double f;
        if (ecc < 1.0e-5)
            f = ((t - tp)/per - (int)((t - tp)/per))*2.*M_PI;
        else {
            const double M = n*(t - tp);
            const double E = getE(M, ecc);
            f = 2.*atan(ratio*tan(E/2.));
        }
        if (f_only) output[i] = f;
        else {
            const double so = sin(f + omega);
            if (transit == 1 && so*si <= 0.) output[i] = 100.;
            else if (transit == 2 && so*si >= 0.) output[i] = 100.;
            else output[i] = a*(1.0 - ecc*ecc)/(1.0 + ecc*cos(f))
                             *sqrt(1.0 - so*so*si*si);
        }
    }
}
