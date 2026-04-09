#pragma once
#ifndef CUFINUFFT_EITHERPREC_H
#define CUFINUFFT_EITHERPREC_H

// Stripped down version of cufinufft_eitherprec.h just for pointers handling in our loader
#include "cufinufft_opts.h"

#ifdef SINGLE
typedef float cufinufft_real;
#else
typedef double cufinufft_real;
#endif

// Opaque handle to the plan
typedef struct cufinufft_plan_s* cufinufft_plan;

// Option structure uses standard primitives, defined in opts.h which we have
// The plan structure is opaque, so we don't need its internal types.

#endif // CUFINUFFT_EITHERPREC_H
