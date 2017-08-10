/* Time-stamp: <sha512mpp.cu 2017-08-04 11:16:31 Hidenori Kuwakado> */

#include <assert.h>
#include <cuda_runtime.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <openssl/rand.h>
#include <openssl/sha.h>


#include "debugMacro.h"

/* NUM_COMLUMNS is given with the compiling option. */
#include "commonParameter.h"
#if NUM_COLUMNS == 4096
#include "matrix.128x4096.h"
#elif NUM_COLUMNS == 8192
#include "matrix.128x8192.h"
#elif NUM_COLUMNS == 16384
#include "matrix.128x16384.h"
#elif NUM_COLUMNS == 32768
#include "matrix.128x32768.h"
#else
#error "No found: matrix file"
#endif
#include "sha512mpp.h"

/* ------------------------------------------------------------ */

static uint8_t*
performMessagePreprocessing(
    size_t* numBytesOfPseudoInputData,
    const uint8_t* const inputData,
    const option_t* const opt);

static size_t
computeNumBlocksOfInputPadding(
    const option_t* const opt);

__global__ void
computeOutputData(
    uint32_t* const outputData_g,
    const uint32_t* const inputData_g,
    const uint32_t* const mat_g);

static void
emulateComputeOutput(
    uint8_t* const output,
    const uint8_t* const input,
    const size_t numBytesOfInput,
    const uint8_t* const mat);

/* ------------------------------------------------------------ */

void
computeDigestWithSha512mp(
    uint8_t* const digest,
    const uint8_t* const inputData,
    const option_t* const opt)
{
    if (xDO) {
        dprintf("inputDataByteLength: %zu\n", opt->numBytesOfInputData);
        dprintf("inputData (first 16 bytes): ");
        for (size_t i = 0; i < 16; ++i) {
            printf("%02"PRIx8, inputData[i]);
        }
        putchar('\n');
    }

    /* Perform the message preprocessing. */
    size_t numBytesOfPseudoInputData = 0;
    uint8_t* pseudoInputData = performMessagePreprocessing(&numBytesOfPseudoInputData, inputData, opt);
    MustBe(pseudoInputData != NULL);

    /* Execute SHA-512 for pseudo-input data. */
    SHA512((unsigned char *)pseudoInputData, numBytesOfPseudoInputData, (unsigned char *)digest);

    /* Clean up */
    free(pseudoInputData);
}


static uint8_t*
performMessagePreprocessing(
    size_t* numBytes,
    const uint8_t* const inputData,
    const option_t* const opt)
{
    /* Suffix rules for memory variables
     * _h: host memory (i.e., PC's memory)
     * _g: global memory on GPU
     * _c: constant memory on GPU
     * _s: shared memory on GPU  */
    cudaError_t rv;

    /* Allocate the memory for the matrix. */
    uint32_t *mat_g[opt->numDevices]; // Not uint8_t
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);
        /* mat_h is declared and initilized in matrix.*.h */
        rv = cudaMalloc((void **)&mat_g[gpuIdx], sizeof(mat_h));
        MustBeCudaSuccess(rv);
    }

    /* Input: original input data and input padding Data */
    const size_t numBlocksOfInputData = opt->numBytesOfInputData / NumBytesOfInputBlock;
    const size_t numBlocksOfInputPadding = computeNumBlocksOfInputPadding(opt);
    const size_t numBlocksOfInput = numBlocksOfInputData + numBlocksOfInputPadding;
    const size_t numBlocksOfInputPerDevice = numBlocksOfInput / opt->numDevices;
    if (xDO) {
        assert(opt->numBytesOfInputData % NumBytesOfInputBlock == 0);
        assert(numBlocksOfInput % opt->numDevices == 0);
        assert(numBlocksOfInput % NumInputBlocksPerThreadBlock == 0);
        dprintf("numBlocksOfInputData %zu\n", numBlocksOfInputData);
        dprintf("numBlocksOfInputPadding %zu\n", numBlocksOfInputPadding);
        dprintf("numBlocksOfInput %zu\n", numBlocksOfInput);
        dprintf("numBlocksOfInputPerDevice %zu\n", numBlocksOfInputPerDevice);
    }

    const size_t numBytesOfInputData    = opt->numBytesOfInputData;
    const size_t numBytesOfInputPadding = NumBytesOfInputBlock * numBlocksOfInputPadding;
    const size_t numBytesOfInput = numBytesOfInputData + numBytesOfInputPadding;
    const size_t numBytesOfInputPerDevice = numBytesOfInput / opt->numDevices;
    if (xDO) {
        dprintf("numBytesOfInputData %zu\n", numBytesOfInputData);
        dprintf("numBytesOfInputPadding %zu\n", numBytesOfInputPadding);
        dprintf("numBytesOfInput %zu\n", numBytesOfInput);
        dprintf("numBytesOfInputPerDevice %zu\n", numBytesOfInputPerDevice);
    }

    const uint8_t* const inputData_h = (uint8_t*)inputData;
    uint8_t* const inputPadding_h = (uint8_t*)malloc(numBytesOfInputPadding);
    MustBe(inputPadding_h != NULL);
    memset(inputPadding_h, 0x00, numBytesOfInputPadding);
    inputPadding_h[0] = 0x80; // Stop bit
    uint32_t* input_g[opt->numDevices]; // Not uint8_t *
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);
        rv = cudaMalloc((void **)&input_g[gpuIdx], numBytesOfInputPerDevice);
        MustBeCudaSuccess(rv);
    }

    /* Output data */
    const size_t numBlocksOfOutputData    = numBlocksOfInputData;
    const size_t numBlocksOfOutputPadding = numBlocksOfInputPadding;
    const size_t numBlocksOfOutput = numBlocksOfInputData + numBlocksOfInputPadding;
    const size_t numBlocksOfOutputPerDevice = numBlocksOfOutput / opt->numDevices;
    if (xDO) {
        assert(numBlocksOfOutput % opt->numDevices == 0);
        dprintf("numBlocksOfOutputData %zu\n", numBlocksOfOutputData);
        dprintf("numBlocksOfOutputPadding %zu\n", numBlocksOfOutputPadding);
        dprintf("numBlocksOfOutput %zu\n", numBlocksOfOutput);
        dprintf("numBlocksOfOutputPerDevice %zu\n", numBlocksOfOutputPerDevice);
    }

    const size_t numBytesOfOutputData    = NumBytesOfOutputBlock * numBlocksOfOutputData;
    const size_t numBytesOfOutputPadding = NumBytesOfOutputBlock * numBlocksOfOutputPadding;
    const size_t numBytesOfOutput = numBytesOfOutputData + numBytesOfOutputPadding;
    const size_t numBytesOfOutputPerDevice = numBytesOfOutput / opt->numDevices;
    if (xDO) {
        assert(numBytesOfOutputPerDevice == numBlocksOfOutputPerDevice * NumBytesOfOutputBlock);
        dprintf("numBytesOfOutputData %zu\n", numBytesOfOutputData);
        dprintf("numBytesOfOutputPadding %zu\n", numBytesOfOutputPadding);
        dprintf("numBytesOfOutput %zu\n", numBytesOfOutput);
        dprintf("numBytesOfOutputPerDevice %zu\n", numBytesOfOutputPerDevice);
    }

    uint8_t* const output_h = (uint8_t*)malloc(numBytesOfOutput);
    MustBe(output_h != NULL);
    uint32_t* output_g[opt->numDevices]; // Not uint8_t *
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);
        rv = cudaMalloc((void **)&output_g[gpuIdx], numBytesOfOutputPerDevice);
        MustBeCudaSuccess(rv);
    }

    /* Kernel-function parameteres */
    /* GridDimX depends on the size of data. */
    const size_t gridDimX = numBlocksOfInputPerDevice / NumInputBlocksPerThreadBlock;
    const dim3 dG = dim3(gridDimX, 1, 1);
    const dim3 dB = dim3(BlockDimX, BlockDimY, 1);
    if (xDO) {
        assert(numBlocksOfInputPerDevice % NumInputBlocksPerThreadBlock == 0);
        dprintf("gridDimX %zu\n", gridDimX);
    }

    /* Perform the preprocessing using GPUs. If multi GPUs are
     * avaiable, then streams and cudaMemcpyAsync() can be used.
     * However, our benchmark shows it is slow if they are used. */
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        if (xDO) {
            dprintf("gpuIdx %d\n", gpuIdx);
        }

        /* Allocate GPU memory for a matrix, and transfer the matrix
         * there. */
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);

        /* Matrix: host -> GPU */
        rv = cudaMemcpy((void*)mat_g[gpuIdx],
                        (const void*)mat_h,
                        sizeof(mat_h),
                        cudaMemcpyHostToDevice);
        MustBeCudaSuccess(rv);

        /* Input data: host -> GPU */
        if (gpuIdx < opt->numDevices - 1) {
            rv = cudaMemcpy((void*)input_g[gpuIdx],
                            (const void*)(inputData_h + numBytesOfInputPerDevice * gpuIdx),
                            numBytesOfInputPerDevice,
                            cudaMemcpyHostToDevice);
            MustBeCudaSuccess(rv);
        } else {
            /* Last GPU: copy data in both of inputData and inputPadding. */
            const size_t numBytesOfRemainingInputData
                = numBytesOfInputPerDevice - numBytesOfInputPadding;
            const size_t numWordsOfRemainingInputData
                = numBytesOfRemainingInputData / NumBytesOfWord;
            if (xDO) {
                assert(numBytesOfInputPerDevice >= numBytesOfInputPadding);
                assert(numBytesOfRemainingInputData % NumBytesOfWord == 0);
                dprintf("numBytesOfRemainingInputData %zu\n", numBytesOfRemainingInputData);
                dprintf("numWordsOfRemainingInputData %zu\n", numWordsOfRemainingInputData);
            }

            rv = cudaMemcpy((void*)input_g[gpuIdx],
                            (const void*)(inputData_h + numBytesOfInputPerDevice * gpuIdx), // inputData_h is uint8_h*.
                            numBytesOfRemainingInputData,
                            cudaMemcpyHostToDevice);
            MustBeCudaSuccess(rv);

            rv = cudaMemcpy((void*)(input_g[gpuIdx] + numWordsOfRemainingInputData), // input_g[] is uint32_t*.
                            (const void*)inputPadding_h, // inputPadding_h[] is uint8_t*.
                            numBytesOfInputPadding,
                            cudaMemcpyHostToDevice);
            MustBeCudaSuccess(rv);
        }
        if (xDO) {
            dprintf("Data transfer completed: %d\n", gpuIdx);
        }

        /* Start to perform the message preprocessing on GPU. */
        computeOutputData<<< dG, dB >>>(output_g[gpuIdx], input_g[gpuIdx], mat_g[gpuIdx]);
    }

    /* Output data: GPU -> host */
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);
        rv = cudaMemcpy((void*)(output_h + (numBytesOfOutputPerDevice * gpuIdx)), // output_h is uint8_t*.
                        (const void*)output_g[gpuIdx],
                        numBytesOfOutputPerDevice,
                        cudaMemcpyDeviceToHost);
        MustBeCudaSuccess(rv);
    }

    /* All the device computation is finished. */
    cudaDeviceSynchronize();
    /* Clean up */
    for (int gpuIdx = 0; gpuIdx < opt->numDevices; ++gpuIdx) {
        rv = cudaSetDevice(gpuIdx);
        MustBeCudaSuccess(rv);
        rv = cudaFree(mat_g[gpuIdx]);
        MustBeCudaSuccess(rv);
        rv = cudaFree(input_g[gpuIdx]);
        MustBeCudaSuccess(rv);
        rv = cudaFree(output_g[gpuIdx]);
        MustBeCudaSuccess(rv);
    }

    if (xDO) { // Debug
        dprintf("paddedOutputData_h: ");
        for (size_t i = 0; i < NUM_ROWS_MATRIX; ++i) {
            printf("%02"PRIx8, output_h[i]);
            if (i % NUM_ROWS_MATRIX == NUM_ROWS_MATRIX - 1) {
                putchar('\n');
            } else if (i % 4 == 3) {
                putchar(' ');
            }
        }
    } // if (DO) {
    if (xDO) { // Debug
        /* Compare GPU's result to CPU's result. */
        /* Input */
        uint8_t* const anotherInput_h = (uint8_t*)malloc(numBytesOfInput);
        MustBe(anotherInput_h != NULL);
        memcpy(anotherInput_h, inputData_h, numBytesOfInputData);
        memcpy(anotherInput_h + numBytesOfInputData, inputPadding_h, numBytesOfInputPadding);
        /* Output */
        uint8_t* const anotherOutput_h = (uint8_t*)malloc(numBytesOfOutput);
        MustBe(anotherOutput_h != NULL);
        memset(anotherOutput_h, 0xde, numBytesOfOutput);

        /* Compute the output using the CPU again. */
        emulateComputeOutput(anotherOutput_h, anotherInput_h, numBytesOfInput, mat_h);
        /* Compare two results. */
        int numNotEqual = 0;
        for (size_t i = 0; i < numBytesOfOutput; ++i) {
            if (output_h[i] != anotherOutput_h[i]) {
                if (numNotEqual < 3) {
                    /* Print the first three elements. */
                    dprintf("NG:        output_h[%zu] = %02"PRIx8"\n", i, output_h[i]);
                    dprintf("    anotherOutput_h[%zu] = %02"PRIx8"\n", i, anotherOutput_h[i]);
                    ++numNotEqual;
                } else {
                    exit(EXIT_FAILURE);
                }
            }
        }
        dprintf("OK: output_h[] = anotherOutput_h[]\n");
        free(anotherInput_h);
        free(anotherOutput_h);
    } // if (DO) {


    /* Returned values */
    *numBytes = numBytesOfOutputData + NumBytesOfOutputBlock;
    return output_h;
}



/* ------------------------------------------------------------ */

/* Determine the number of padding blocks in such a way that the total
 * number of blocks has to be a multiple of opt->numDevices *
 * NumInputBlocksPerThreadBlock. */
static size_t
computeNumBlocksOfInputPadding(
    const option_t* const opt)
{
    const size_t numBlocksOfInputData = opt->numBytesOfInputData / NumBytesOfInputBlock;
    if (xDO) {
        assert(opt->numBytesOfInputData % NumBytesOfInputBlock == 0);
    }
    const size_t u = opt->numDevices * NumInputBlocksPerThreadBlock;

    return u - (numBlocksOfInputData % u);
}



/* ------------------------------------------------------------ */
/***** GPU kernel function *****/

/* triple operartor should be replaced in the first argument. */
static __device__ __forceinline__ uint32_t
conditionalSimdAdd(
    const uint32_t v,
    const uint32_t e,
    const uint32_t bit)
{
#if (__CUDA_ARCH__ > 300)
    /* The first argument should be "bit ? e : 0x00U" because it's
     * faster. Why? */
    return __vadd4(bit ? e : 0x00U, v);
#elif (__CUDA_ARCH__ == 300)
    uint32_t eOrZero = bit ? e : 0x00U;
    uint32_t r, c = 0;
    asm ("vadd4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(v), "r"(eOrZero), "r"(c));
    return r;
#else
    uint32_t eOrZero = bit ? e : 0x00U;
    uint32_t s = (v ^ eOrZero) & 0x80808080U;
    uint32_t r = v & 0x7f7f7f7fU;
    uint32_t t = eOrZero & 0x7f7f7f7fU;
    return (r + t) ^ s;
#endif
}


/* Compute output data using a GPU. */
__global__ void
computeOutputData(
    uint32_t* const output_g,
    const uint32_t* const input_g,
    const uint32_t* const mat_g)
{
    if (xDO) {
        assert(WarpSize == warpSize);
        assert(NumRows == WarpSize);
        assert(NumColumns % NumBitsOfWord == 0);
        assert(NumRows * NumBitsOfByte == NumThreadsPerThreadBlock);
    }

    /* Thread index in the block */
    const uint32_t threadId = threadIdx.x + BlockDimX * threadIdx.y;
    /* Lane index in the warp */
    const uint32_t laneId = threadId & 0x1fU; // threadId % warpSize;
    /* Warp index in the block */
    const uint32_t warpId = threadId >> 5; // threadId / warpSize;
    /* Thread block index */
    const uint32_t threadBlockId = blockIdx.x + gridDim.x * blockIdx.y;

    const uint32_t *thisInput_g = input_g
        + (((NumWordsOfInputBlock * NumInputBlocksPerThreadBlock) * threadBlockId)
           + NumWordsOfInputBlock * warpId);
    const uint32_t *pMat_g = mat_g;
    uint32_t v0 = 0, v1 = 0, v2 = 0, v3 = 0; // the result (i.e., the sum)
    for (uint32_t col = 0; col < NumColumns; col += NumBitsOfWord) {

        /* Read data words. */
        uint32_t w0 = *(thisInput_g + (NumWordsOfInputBlock * 8 * 0));
        uint32_t w1 = *(thisInput_g + (NumWordsOfInputBlock * 8 * 1));
        uint32_t w2 = *(thisInput_g + (NumWordsOfInputBlock * 8 * 2));
        uint32_t w3 = *(thisInput_g + (NumWordsOfInputBlock * 8 * 3));
        ++thisInput_g;

        /* Prepaer two submatrixes for saving parts of the matrix. */
        __shared__ uint32_t subMatrix0_s[NumRows * NumBitsOfByte]; // [NumThreadsPerThreadBlock]
        __shared__ uint32_t subMatrix1_s[NumRows * NumBitsOfByte]; // [NumThreadsPerThreadBlock]

        /* Read elements of the first half of submatrix using all the
         * threads. */
        subMatrix0_s[threadId] = *(pMat_g + threadId);
        pMat_g += NumThreadsPerThreadBlock;

        /* Synchronize the first half of the subMatrix. */
        __syncthreads();
        /* Read elements of the last half of submatrix using all the
         * threads. Do not synchronize the last half of the submatrix
         * here. */
        subMatrix1_s[threadId] = *(pMat_g + threadId);
        pMat_g += NumThreadsPerThreadBlock;
        /* The following loop uses only the first half of the
         * submatrix (i.e., *thisSubMatrix0_s). */
        uint32_t *thisElement = subMatrix0_s + (laneId);
        uint32_t c = NumBitsOfByte;
        while (c--) {
            const uint32_t e = *thisElement;
            v0 = conditionalSimdAdd(v0, e, w0 & 0x01U);
            v1 = conditionalSimdAdd(v1, e, w1 & 0x01U);
            v2 = conditionalSimdAdd(v2, e, w2 & 0x01U);
            v3 = conditionalSimdAdd(v3, e, w3 & 0x01U);
            w0 >>= 1;
            w1 >>= 1;
            w2 >>= 1;
            w3 >>= 1;
            thisElement += NumRows;
        } // while (c--) {

        /* Synchronize the last half of the subMatrix (i.e.,
         * *thisSubMatrix1_s). */
        __syncthreads();
        /* Read elements of the first half of submatrix using all the
         * threads. Do not synchronize the first half of the submatrix
         * here. */
        subMatrix0_s[threadId] = *(pMat_g + threadId);
        pMat_g += NumThreadsPerThreadBlock;
        /* The following loop uses only the last half of the submatrix
         * (i.e., *thisSubMatrix1_s). */
        thisElement = subMatrix1_s + (laneId);
        c = NumBitsOfByte;
        while (c--) {
            const uint32_t e = *thisElement;
            v0 = conditionalSimdAdd(v0, e, w0 & 0x01U);
            v1 = conditionalSimdAdd(v1, e, w1 & 0x01U);
            v2 = conditionalSimdAdd(v2, e, w2 & 0x01U);
            v3 = conditionalSimdAdd(v3, e, w3 & 0x01U);
            w0 >>= 1;
            w1 >>= 1;
            w2 >>= 1;
            w3 >>= 1;
            thisElement += NumRows;
        } // while (c--) {

        /* Synchronize the first half of the subMatrix (i.e.,
         * *thisSubMatrix0_s). */
        __syncthreads();
        /* Read elements of the last half of submatrix using all the
         * threads. Do not synchronize the first half of the submatrix
         * here. */
        subMatrix1_s[threadId] = *(pMat_g + threadId);
        pMat_g += NumThreadsPerThreadBlock;
        /* The following loop uses only the first half of the submatrix
         * (i.e., *thisSubMatrix0_s). */
        thisElement = subMatrix0_s + (laneId);
        c = NumBitsOfByte;
        while (c--) {
            const uint32_t e = *thisElement;
            v0 = conditionalSimdAdd(v0, e, w0 & 0x01U);
            v1 = conditionalSimdAdd(v1, e, w1 & 0x01U);
            v2 = conditionalSimdAdd(v2, e, w2 & 0x01U);
            v3 = conditionalSimdAdd(v3, e, w3 & 0x01U);
            w0 >>= 1;
            w1 >>= 1;
            w2 >>= 1;
            w3 >>= 1;
            thisElement += NumRows;
        } // while (c--) {

        /* Synchronize the last half of the subMatrix (i.e.,
         * *thisSubMatrix1_s). */
        __syncthreads();
        /* The following loop uses only the last half of the submatrix
         * (i.e., *thisSubMatrix1_s). */
        thisElement = subMatrix1_s + (laneId);
        c = NumBitsOfByte;
        while (c--) {
            const uint32_t e = *thisElement;
            v0 = conditionalSimdAdd(v0, e, w0 & 0x01U);
            v1 = conditionalSimdAdd(v1, e, w1 & 0x01U);
            v2 = conditionalSimdAdd(v2, e, w2 & 0x01U);
            v3 = conditionalSimdAdd(v3, e, w3 & 0x01U);
            w0 >>= 1;
            w1 >>= 1;
            w2 >>= 1;
            w3 >>= 1;
            thisElement += NumRows;
        } // while (c--) {

        if (xDO) {
            assert(w0 == 0x00U);
            assert(w1 == 0x00U);
            assert(w2 == 0x00U);
            assert(w3 == 0x00U);
        }
    } // for (uint32_t col = 0; col < NumColumns; col += NumBitsOfWord) {

    /* Write all the sums in global memory by coalescing access. */
    uint32_t *thisOutput_g = output_g + ((NumInputBlocksPerThreadBlock * NumRows) * threadBlockId + threadId);
    *thisOutput_g = v0; thisOutput_g += NumRows * 8;
    *thisOutput_g = v1; thisOutput_g += NumRows * 8;
    *thisOutput_g = v2; thisOutput_g += NumRows * 8;
    *thisOutput_g = v3;
}


/* ------------------------------------------------------------ */
/* Emulate the GPU kernel function for debug */
static void
emulateComputeOutput(
    uint8_t* const output,
    const uint8_t* const input,
    const size_t numBytesOfInput,
    const uint8_t* const mat)
{
    size_t idx = 0;

    /* mat (i.e., byteMatrix.h) gives elements in order of column. It
     * is converted to a 2-dimensional array in order of row. */
    uint8_t matrix[NUM_ROWS_MATRIX][NUM_COLS_MATRIX];
    for (size_t c = 0 ; c < NUM_COLS_MATRIX; ++c) {
        for (size_t r = 0 ; r < NUM_ROWS_MATRIX; ++r) {
            matrix[r][c] = mat[idx];
            ++idx;
        }
    }

    idx = 0;
    for (size_t i = 0; i < numBytesOfInput; i += NumBytesOfInputBlock) {
        /* 1 bit -> 1 byte: not cool... */
        uint8_t buf[NumBitsOfByte * NumBytesOfInputBlock];
        for (size_t j = 0, c = 0; j < NumBytesOfInputBlock; ++j) {
            uint8_t v = input[i + j];
            for (size_t k = 0; k < NumBitsOfByte; ++k) {
                buf[c] = (v & 0x01U) ? 1 : 0;
                ++c;
                v >>= 1;
            }
        }

        /* Multiply the matrix by the binary vector. */
        for (uint32_t r = 0; r < NUM_ROWS_MATRIX; ++r) {
            uint8_t sum = 0;
            for (uint32_t c = 0; c < NUM_COLS_MATRIX; ++c) {
                sum += matrix[r][c] * buf[c];
           }
            *(output + idx) = sum;
            ++idx;
        }
    } // for (size_t i = 0; i < numBytesOfInput; i += NumBytesOfInputBlock) {
}

/* end of file */
