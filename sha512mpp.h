/* Time-stamp: <sha512mpp.h 2017-08-04 11:16:42 Hidenori Kuwakado> */

#ifndef SHA512MPP_H
#define SHA512MPP_H

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t

/* Command-line options */
typedef struct {
    bool help;
    time_t lifetime; // The shortest running time (in second)
    size_t numBytesOfInputData; // The number of bytes of a message to be hashed
    int numInputData; // The number of repeated
    int numDevices; // The number of used GPUs
    bool sha512; // Perform normal SHA-512
    bool version;
} option_t;


void
computeDigestWithSha512mp
(uint8_t* const digest,
 const uint8_t* const data,
 const option_t* const opt);

#endif

/* end of file */
