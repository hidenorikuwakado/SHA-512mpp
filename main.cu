/* Time-stamp: <main.cu 2017-08-04 11:16:16 Hidenori Kuwakado> */

#include <assert.h>
#include <cuda_runtime.h>
#include <getopt.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "debugMacro.h"

#include "commonParameter.h"
#include "sha512.h"
#include "sha512mpp.h"


/* ------------------------------------------------------------ */

#define PROGRAM_NAME "sha512mpp"
#define PROGRAM_VERSION "1.0"
#define PROGRAM_COPYRIGHT "Copyright (C) 2017 Hidenori Kuwakado"

/* The number of elements in an array */
#define NELMS(a) (sizeof(a)/sizeof(a[0]))

#define ToLogicalStr(a) (((a) == true) ? "true" : (((a) == false) ? "false" : "unknown"))

/* ------------------------------------------------------------ */


/* Local functions */
static double getClock(void);
static void usage(void);
static double
computeThroughtput(
    const double start,
    const double finish,
    const size_t numBytesPerData,
    const int numData);
static uint8_t toyRandom(void);


/* ------------------------------------------------------------ */

int
main(int argc, char* argv[])
{
    /* How many GPUs? */
    int numDevices = 0;
    cudaError_t rv = cudaGetDeviceCount(&numDevices);
    MustBeCudaSuccess(rv);

    option_t opt = {
        .help = false,
        .lifetime = 5 * 60, // 5 [min] = 300 [s]
        .numBytesOfInputData = 2 * 1024 * 1024, // 2 [MiB]
        .numInputData = 1,
        .numDevices = numDevices, // Use all the GPUs.
        .sha512 = false,
        .version = false,
    };

    /* Only long command-line options are acceptable. */
    while (1) {
        enum longOptionIdentifier { // option name
            Help, Lifetime, NumDevices, NumInputData, NumMiBytes, Sha512, Version,
            LASTDUMMY
        };
        assert(LASTDUMMY < '?');
        const struct option longopts[] = {
            { "help", no_argument, NULL, Help },
            { "lifetime", required_argument, NULL, Lifetime },
            { "numDevices", required_argument, NULL, NumDevices },
            { "numInputData", required_argument, NULL, NumInputData },
            { "numMiBytes", required_argument, NULL, NumMiBytes },
            { "sha512", no_argument, NULL, Sha512 },
            { "version", no_argument, NULL, Version },
            { NULL, 0, NULL, 0 },
        };
        int longindex = 0;
        int c = getopt_long_only(argc, argv, "", longopts, &longindex);
        if (c == -1) {
            break;
        }
        switch (c) {
        case Help:
            opt.help = true;
            usage();
            exit(EXIT_SUCCESS);
            break;
        case Lifetime:
            if (atoi(optarg) >= 0) {
                opt.lifetime = (time_t)atoi(optarg);
            } else {
                fprintf(stderr,
                        "Error: --lifetime requires a non-negative integer (in seconds): %d\n",
                    atoi(optarg));
                exit(EXIT_FAILURE);
            }
            break;
        case NumDevices:
            opt.numDevices = atoi(optarg);
            if (opt.numDevices < 0) {
                fprintf(stderr, "Error: --numDevices requires a non-negative integer: %d\n",
                    opt.numDevices);
                exit(EXIT_FAILURE);
            } else if (opt.numDevices == 0) {
                /* Use all the GPUs. */
                opt.numDevices = numDevices;
            } else if (opt.numDevices > numDevices) {
                fprintf(stderr, "Error: --numDevices is too large (<= %d)\n", numDevices);
                exit(EXIT_FAILURE);
            }
            break;
        case NumInputData:
            opt.numInputData = atoi(optarg);
            if (opt.numInputData <= 0) {
                fprintf(stderr, "Error: --numData requires a positive integer: %d\n", atoi(optarg));
                exit(EXIT_FAILURE);
            }
            break;
        case NumMiBytes:
            if (atoi(optarg) < 2) {
                /* The size less than 2 MiBytes is not sufficient for
                 * 4-way GPU and the 128x32768 matrix. */
                fprintf(stderr, "Error: --numMiBytes requires an integer larger 2: %d.\n", atoi(optarg));
                exit(EXIT_FAILURE);
            } else {
                /* in Bytes (not MiBytes) */
                opt.numBytesOfInputData = (size_t)atoi(optarg) * (size_t)(1024 * 1024);
                if (DO) {
                    /* We assume that the byte length is a multiple of
                     * input-block length. */
                    assert(opt.numBytesOfInputData % NumBytesOfInputBlock == 0);
                }
            }
            break;
        case Sha512:
            opt.sha512 = true;
            break;
        case Version:
            opt.version = true;
            printf("%s %s\n", PROGRAM_NAME, PROGRAM_VERSION);
            printf("%s\n", PROGRAM_COPYRIGHT);
            exit(EXIT_SUCCESS);
            break;
        case '?':
            /* fall through */
        default:
            /* The given (unrecognized) option is displayed by getopt_long_only(). */
            usage();
            exit(EXIT_FAILURE);
        }
    }
    if (xDO) {
        dprintf("--help: %s\n", ToLogicalStr(opt.help));
        dprintf("--liftime: %ld\n", opt.lifetime);
        dprintf("--numDevices: %d\n", opt.numDevices);
        dprintf("--numInputData: %d\n", opt.numInputData);
        dprintf("--numMibytes: %zu\n", opt.numBytesOfInputData / (1024 * 1024));
        dprintf("--sha512: %s\n", ToLogicalStr(opt.sha512));
        dprintf("--version: %s\n", ToLogicalStr(opt.version));
    }

    /* We assume numBytesOfData is a multiple of an input-data
     * block. */
    uint8_t* const inputData = (uint8_t *)malloc(opt.numBytesOfInputData);
    MustBe(inputData != NULL);
    for (size_t i = 0; i < opt.numBytesOfInputData; ++i) {
        inputData[i] = toyRandom();
    }
    if (xDO) {
        const char* inputDataFile = "inputData.dat";
        dprintf("Write data file: %s\n", inputDataFile);
        FILE* const fp = fopen(inputDataFile, "w");
        MustBe(fp != NULL);
        for (size_t i = 0; i < opt.numBytesOfInputData; ++i) {
            fputc(inputData[i], fp);
        }
        fclose(fp);
    }

    /* To compute hashing throughput [MiB/s] */
    double start = 0.0;
    double secondStart = 0.0;
    double finish = 0.0;
    int totalNumInputData = 0;
    /* A resulting hash value */
    uint8_t digest[NumBytesOfSHA512OutputBlock] = { 0x00 };

    /* Body of a hash computation */
    if (opt.sha512 == true) {
        /* SHA-512 */
        start = getClock();
        /* Execute the computation of "opt.numInputData" data at least
         * for "opt.liftime" seconds. */
        do {
            for (int n = 0; n < opt.numInputData; ++n) {
                computeDigestWithSha512(digest, inputData, &opt);
            }
            finish = getClock();
            totalNumInputData += opt.numInputData;
        } while (finish < start + (double)opt.lifetime);
    } else {
        /* Repeat numData times. */
        start = getClock();
        /* The first call of cudaMalloc() takes much time. */
        for (int deviceIdx = 0; deviceIdx < numDevices; ++deviceIdx) {
            rv = cudaSetDevice(deviceIdx);
            MustBeCudaSuccess(rv);
            uint32_t* dummy = NULL;
            rv = cudaMalloc((void**)&dummy, sizeof(uint32_t));
            MustBeCudaSuccess(rv);
            rv = cudaFree(dummy);
            MustBeCudaSuccess(rv);
        }
        secondStart = getClock();
        /* Execute the computation of "opt.numInputData" data at least
         * for "opt.liftime" seconds. */
        do {
            for (int n = 0; n < opt.numInputData; ++n) {
                computeDigestWithSha512mp(digest, inputData, &opt);
            }
            finish = getClock();
            totalNumInputData += opt.numInputData;
        } while (finish < secondStart + (double)opt.lifetime);
    }

    /* Throught in MiBytes per second */
    printf("The number of columns of the matrix: %d\n", NUM_COLS_MATRIX);
    printf("The number of data: %d\n", opt.numInputData);
    printf("The size of data: %zu [Bytes] = %zu [MiB]\n",
           opt.numBytesOfInputData, opt.numBytesOfInputData / (1024 * 1024));
    if (opt.sha512 != true) {
        printf("The number of devices: %d\n", opt.numDevices);
    }
    printf("The total number of data: %d\n", totalNumInputData);
    
    /* Results */
    const double throughput = computeThroughtput(start, finish, opt.numBytesOfInputData, totalNumInputData);
    printf("Elapsed: %f [s]\n", finish - start);
    printf("Throughput: %f [MiB/s]\n", throughput);
    /* Results except for initilaizing GPU memeory. */
    if (opt.sha512 != true) {
        const double throughput2 = computeThroughtput(secondStart, finish, opt.numBytesOfInputData, totalNumInputData);
        printf("Elapsed2: %f [s]\n", finish - secondStart);
        printf("Throughput2: %f [MiB/s]\n", throughput2);
    }
    fputs("Digest: ", stdout);
    for (size_t i = 0; i < NELMS(digest); ++i) {
        printf("%02"PRIx8, digest[i]);
    }
    putchar('\n');

    /* Clean up */
    free(inputData);

    return EXIT_SUCCESS;
}


/* ------------------------------------------------------------ */

static void
usage(void)
{
    printf("Usage: %s [long option]\n", PROGRAM_NAME);
    /* Option strings are defined in struct option longopts[]. */
    printf("--help   This message is displayed.\n");
    printf("--lifetime n   Perform this program at least n [s].\n");
    printf("--numData n   Perform hashing n times for benchmark.\n");
    printf("--numDevices n   The number of used GPUs (all: 0).\n");
    printf("--numMiBytes n   The byte length of a data to be hashed.\n");
    printf("--sha512   No preprocesing is done (i.e., normal SHA-512).\n");
    printf("--version   The version and the copyright are displayed.\n");
}


static double
getClock(void)
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}


static double
computeThroughtput(
    const double start,
    const double finish,
    const size_t numBytesPerData,
    const int numData)
{
    if (xDO) {
        dprintf("start  %f\n", start);
        dprintf("finish %f\n", finish);
        dprintf("numBytesPerData %zu\n", numBytesPerData);
        dprintf("numData %d\n", numData);
    }
    if (start < finish) {
        /* Averaget throught in MiBytes per second */
        double totalMiBytes = ((double)numBytesPerData / (1024.0 * 1024.0)) * (double)numData;
        if (xDO) {
            dprintf("totalMiBytes %f\n", totalMiBytes);
        }
        return totalMiBytes / (finish - start);
    } else {
        return -1.0;
    }
}


/* Generate random-like data. Since the implementation of rand()
 * depends on a system, the value may be changed. If so, we cannot
 * compare the results to check the correctness of computation.
 * This code is based on the following site:
 * https://en.wikipedia.org/wiki/Linear_congruential_generator */
static uint8_t
toyRandom(void)
{
    static uint32_t x = 0x36c979U; // meaningless
    const uint32_t a = 1664525U;
	const uint32_t c = 1013904223U;
    x = a * x + c;
    return (uint8_t)(x >> 3);
}

/* end of file */
