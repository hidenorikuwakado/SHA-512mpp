/* Time-stamp: <sha512.cu 2017-07-04 07:33:56 Hidenori Kuwakado> */
/*
 * Compute a hash with SHA-512. Usually, this module usually just
 * calls SHA512 function of OpenSSL.
 */

#include <assert.h>
#include <limits.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include "debugMacro.h"
#include "commonParameter.h"
#include "sha512.h"


/* ------------------------------------------------------------ */

void
computeDigestWithSha512
(uint8_t* const digest,
 const uint8_t* const inputData,
 const option_t* const opt)
{
    if (xDO) {
        dprintf("numBytesOfInputData: %zu\n", opt->numBytesOfInputData);
        if (opt->numBytesOfInputData < 16) {
            dprintf("data: ");
            for (size_t i = 0; i < opt->numBytesOfInputData; ++i) {
                printf("%02x", inputData[i]);
            }
        } else {
            dprintf("data (fisrt/last 8 bytes): ");
            for (size_t i = 0; i < 8; ++i) {
                printf("%02x", inputData[i]);
            }
            fputs(" ... ", stdout);
            for (size_t i = opt->numBytesOfInputData - 8; i < opt->numBytesOfInputData; ++i) {
                printf("%02x", inputData[i]);
            }
        }
        putchar('\n');
    }

    /* SHA512 has been declared in openssl/sha.h as follows:
     * unsigned char *SHA512(
     *   const unsigned char *d,
     *   size_t n,
     *   unsigned char *md);
     * where n is the byte length of data pointed by d. */
    SHA512((unsigned char *)inputData, opt->numBytesOfInputData, (unsigned char *)digest);
}

/* end of file */
