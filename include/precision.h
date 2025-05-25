#ifndef PRECISION_H
#define PRECISION_H

#include <cfloat>
#include <cmath>
#include <limits>

#ifdef DOUBLE

/**
 * @brief Defines the floating-point type as double.
 */
typedef double PRECISION;
#define ONE 1.0
#define EULER 0.57721566490153286060
#define EPSILON 1.E-30
#define MAX_VAL DBL_MAX
#define ABS(X) fabs(X)
#define EXP(X) exp(X)
#define LOG(X) log(X)
#else

/**
 * @brief Defines the floating-point type as float.
 */
typedef float PRECISION;
#define ONE 1.0f
#define EULER 0.5772156649f
#define EPSILON 1.E-30f
#define MAX_VAL FLT_MAX
#define ABS(X) fabsf(X)
#define EXP(X) expf(X)
#define LOG(X) logf(X)
#endif

#endif // PRECISION_H
