/* Time-stamp: <sha512.h 2017-08-04 11:18:46 Hidenori Kuwakado> */

#ifndef SHA512_H
#define SHA512_H

#include <stddef.h> // size_t 
#include <stdint.h> // uint8_t

#include "sha512mpp.h" // option_t

void
computeDigestWithSha512
(uint8_t* const digest,
 const uint8_t* const message,
 const option_t* const opt);

#endif

/* end of file */
