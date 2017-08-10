/* Time-stamp: <commonParameter.h 2017-06-20 08:12:58 Hidenori Kuwakado> */

#ifndef COMMONPARAMETER_H
#define COMMONPARAMETER_H

/* NUM_COMLUMNS is given with the compiling option. */
#define NUM_ROWS_MATRIX (128)
#if NUM_COLUMNS == 4096 || NUM_COLUMNS == 8192 || NUM_COLUMNS == 16384 || NUM_COLUMNS == 32768
#define NUM_COLS_MATRIX (NUM_COLUMNS)
#else
#error "Not given: NUM_CMLUMNS"
#endif

/* The number of bits in a byte/word/double word */
#define NumBitsOfByte   (8)
#define NumBitsOfWord  (32)
#define NumBitsOfDword (64)
#define NumBytesOfWord  (4)
#define NumBytesOfDWord (8)


/* Constants of the SHA-512 compression function */
/* Input */
#define NumBitsOfSHA512InputBlock  (1024)
#define NumBytesOfSHA512InputBlock ((NumBitsOfSHA512InputBlock) / (NumBitsOfByte))
#define NumWordsOfSHA512InputBlock ((NumBitsOfSHA512InputBlock) / (NumBitsOfWord))
#define NumDwordsOfSHA512InBlock   ((NumBitsOfSHA512InputBlock) / (NumBitsOfDword))
/* Output */
#define NumBitsOfSHA512OutputBlock   (512)
#define NumBytesOfSHA512OutputBlock  ((NumBitsOfSHA512OutputBlock) / (NumBitsOfByte))
#define NumWordsOfSHA512OutputBlock  ((NumBitsOfSHA512OutputBlock) / (NumBitsOfWord))
#define NumDwordsOfSHA512OutputBlock ((NumBitsOfSHA512OutputBlock) / (NumBitsOfDword))


/* Constants that are related with message preprocessing */
/* Input */
#define NumBitsOfInputBlock   (NUM_COLS_MATRIX)
#define NumBytesOfInputBlock  ((NumBitsOfInputBlock) / (NumBitsOfByte))
#define NumWordsOfInputBlock  ((NumBitsOfInputBlock) / (NumBitsOfWord))
#define NumDwordsOfInputBlock ((NumBitsOfInputBlock) / (NumBitsOfDword))
/* Output */
#define NumBitsOfOutputBlock    (NumBitsOfSHA512InputBlock)
#define NumBytesOfOutputBlock   ((NumBitsOfOutputBlock) / (NumBitsOfByte))
#define NumWordsOfOutputBlock   ((NumBitsOfOutputBlock) / (NumBitsOfWord))
#define NumDwordsOfOutputBlock  ((NumBitsOfOutputBlock) / (NumBitsOfDword))


/* Constants of the GPU kernel function */
/* The word number of rows of the matrix. Four bytes are packed into
 * one word. This value is assumed to be equal to warpSize. */
#define NumRows ((NUM_ROWS_MATRIX) / (NumBitsOfWord / NumBitsOfByte))
/* The number of columns of the matrix. */
#define NumColumns (NUM_COLS_MATRIX)
/* warpSize is effective only in the kernel function. */
#define WarpSize (32)
/* The number of input-data blocks handled by one thread */
#define NumInputBlocksPerThread (4)
/* The number of input-data blocks handled by one thread block. The
 * sum of one row is computed with one thread, but one thread computes
 * sums of several rows (i.e., NumInputDataBlocksPerThread). */
#define NumInputBlocksPerThreadBlock ((NumThreadsPerThreadBlock / NumRows) * NumInputBlocksPerThread)
/* The number of threads in a thread block. Other parameters depends on this value. */
#define NumThreadsPerThreadBlock (256)
#define BlockDimX (32)
#define BlockDimY ((NumThreadsPerThreadBlock) / (BlockDimX))

#endif // #ifndef COMMONPARAMETER_H

/* end of file */
