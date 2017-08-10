/* Time-stamp: <debugMacro.h 2017-06-20 08:14:34 Hidenori Kuwakado> */
/* Macros for debugging */

#ifndef DEBUGMACRO_H
#define DEBUGMACRO_H

#include <assert.h> // __ASSERT_VOID_CAST, ...
#include <stdio.h> // fprintf()
#include <stdlib.h> // exit()
#include <cuda_runtime.h> // cudaSuccess


#ifndef DO
#define DO (1)
#else
#error "DO" has been defined.
#endif


#ifndef xDO
#define xDO (0)
#else
#error "xDO" has been defined.
#endif


#ifndef ____MTS
#define ____MTS(x) #x
#else
#error "____MTS()" has been defined.
#endif


#ifndef ___MTS
#define ___MTS(x) ____MTS(x) // Tricky, but necessary macro for printing __LINE__
#else
#error "___MTS()" has been defined.
#endif


#ifndef dprintf
#define dprintf(...) do { printf(__FILE__ "(" ___MTS(__LINE__) ") " __VA_ARGS__); } while (0)
#else
#error "dprintf()" has been defined.
#endif


#ifndef MustBe
/* This macro needs assert.h. */
#define MustBe(expr) \
  ((expr) \
   ? __ASSERT_VOID_CAST(0) \
   : __assert_fail(__STRING(expr), __FILE__, __LINE__, __ASSERT_FUNCTION))
#else
#error "MustBe()" has been defined.
#endif


/* Macro for checking the return value of a CUDA function */
#ifndef MustBeCudaSuccess
#define MustBeCudaSuccess(rv) \
    do { \
        if ((rv) != cudaSuccess) { \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(rv), rv, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
#error "MustBeCudaSuccess()" has been defined.
#endif


#endif

/* end of file */
