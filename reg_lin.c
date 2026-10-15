/* TP Calcul scientifique - Regression lineaire simple, version C (+OpenMP)
   Compilation : gcc -O2 reg_lin.c -o reg_lin -lm
                 gcc -O3 -fopenmp reg_lin.c -o reg_lin_omp -lm
   Usage : ./reg_lin fichier.bin [nrep]                                  */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Lecture du fichier binaire : int64 n, n doubles x, n doubles y */
int lire_binaire(const char *nom, long *n, double **x, double **y)
{
    FILE *f = fopen(nom, "rb");
    if (!f) { printf("Erreur ouverture %s\n", nom); return 1; }
    long long nn;
    fread(&nn, sizeof(long long), 1, f);
    *n = (long)nn;
    *x = malloc(*n * sizeof(double));
    *y = malloc(*n * sizeof(double));
    if (!*x || !*y) { printf("Erreur malloc\n"); return 1; }
    fread(*x, sizeof(double), *n, f);
    fread(*y, sizeof(double), *n, f);
    fclose(f);
    return 0;
}

/* Calcul de a et b par les formules avec moyennes (2 passes) */
void regression(long n, const double *x, const double *y, double *a, double *b)
{
    double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0;
    long i;
    #pragma omp parallel for reduction(+:sx,sy)
    for (i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
    double xm = sx / n, ym = sy / n;
    #pragma omp parallel for reduction(+:sxy,sxx)
    for (i = 0; i < n; i++) {
        double dx = x[i] - xm;
        sxy += dx * (y[i] - ym);
        sxx += dx * dx;
    }
    *a = sxy / sxx;
    *b = ym - (*a) * xm;
}

/* R^2 et ecart-type des residus */
void statistiques(long n, const double *x, const double *y, double a, double b,
                  double *r2, double *sigma)
{
    double sy = 0.0, ssres = 0.0, sstot = 0.0;
    long i;
    #pragma omp parallel for reduction(+:sy)
    for (i = 0; i < n; i++) sy += y[i];
    double ym = sy / n;
    #pragma omp parallel for reduction(+:ssres,sstot)
    for (i = 0; i < n; i++) {
        double r = y[i] - (a * x[i] + b);
        ssres += r * r;
        sstot += (y[i] - ym) * (y[i] - ym);
    }
    *r2 = 1.0 - ssres / sstot;
    *sigma = sqrt(ssres / (n - 2));
}

double chrono(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

int main(int argc, char **argv)
{
    if (argc < 2) { printf("Usage : %s fichier.bin [nrep]\n", argv[0]); return 1; }
    int nrep = (argc > 2) ? atoi(argv[2]) : 1;
    long n; double *x, *y, a = 0, b = 0, r2, sigma;
    if (lire_binaire(argv[1], &n, &x, &y)) return 1;
#ifdef _OPENMP
    printf("OpenMP : %d threads\n", omp_get_max_threads());
#endif
    double t0 = chrono();
    for (int k = 0; k < nrep; k++) regression(n, x, y, &a, &b);
    double t1 = chrono();
    statistiques(n, x, y, a, b, &r2, &sigma);
    printf("n = %ld  a = %.6f  b = %.6f  R2 = %.6f  sigma = %.4f\n", n, a, b, r2, sigma);
    printf("temps regression (%d rep) = %.3f s  (%.4f s / rep)\n", nrep, t1 - t0, (t1 - t0) / nrep);
    free(x); free(y);
    return 0;
}
