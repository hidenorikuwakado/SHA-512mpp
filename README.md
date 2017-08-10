# SHA-512mpp
This is Source codes for SHA-512 with a message preprocessing. The technical description will be given later.

## Requirements
CUDA 7.5 or 8.0

## How to build
1. Modify variables in makefile. 
2. make
3. make testRun
4. Focus on 'Digest' of the output. If they are the same as the following, the buid is success. 

SHA-512:  
The number of columns of the matrix: 4096  
The number of data: 2  
The size of data: 4194304 [Bytes] = 4 [MiB]  
The total number of data: 502  
Elapsed: 3.005969 [s]  
Throughput: 668.004270 [MiB/s]  
**Digest: 3369283651ab114caf4152b5bb5513a6d8e027d57f1559b26fb77aae820813ade6f7e9e8339fbae29274d682b70b569260526785717b19b38c933a12ad96a7d6**  
SHA512mp 4096:  
The number of columns of the matrix: 4096  
The number of data: 2  
The size of data: 4194304 [Bytes] = 4 [MiB]  
The number of devices: 1  
The total number of data: 478  
Elapsed: 3.078606 [s]  
Throughput: 621.060333 [MiB/s]  
Elapsed2: 3.003162 [s]  
Throughput2: 636.662311 [MiB/s]  
**Digest: ab47ae80fa6c51685885165da6a28c4f3ac85d739ba3034f94d717632aa5f834d76ddc7bbc737faa247859a3a38dcab89d9df1f791c137e6bfc01d7da0b5bf2b**  
SHA512mp 8192:  
The number of columns of the matrix: 8192  
The number of data: 2  
The size of data: 4194304 [Bytes] = 4 [MiB]  
The number of devices: 1  
The total number of data: 564  
Elapsed: 3.077045 [s]  
Throughput: 733.170957 [MiB/s]  
Elapsed2: 3.001201 [s]  
Throughput2: 751.699091 [MiB/s]  
**Digest: 2f61dc7de76a566711d82e90f21fae7a2118f7ecc4af17487f7db89c3d3aecb4023af92b259e5260c2cf2073f994221ed66a6a4e038e77338c2ae6c7e975ab8a**  
SHA512mp 16384:  
The number of columns of the matrix: 16384  
The number of data: 2  
The size of data: 4194304 [Bytes] = 4 [MiB]  
The number of devices: 1  
The total number of data: 582  
Elapsed: 3.079117 [s]  
Throughput: 756.060954 [MiB/s]  
Elapsed2: 3.002739 [s]  
Throughput2: 775.292170 [MiB/s]  
**Digest: abddda2e576060ec683d6f57995efd20f3fd804c1a506a761177b10fc13e7feffa1e3ef7b05f0dd967b7a61260162c6fe539297bf3c7e8301bde3b6047175561**  
SHA512mp 32768:  
The number of columns of the matrix: 32768  
The number of data: 2  
The size of data: 4194304 [Bytes] = 4 [MiB]  
The number of devices: 1  
The total number of data: 466  
Elapsed: 3.081803 [s]  
Throughput: 604.840768 [MiB/s]  
Elapsed2: 3.006782 [s]  
Throughput2: 619.931912 [MiB/s]  
**Digest: 96fc8e2b33ef2357426e1be44a4cf0f2fe01a931a4e2926f9cc2dfd8197b124b26b174a2f020789a66ff185cfab1a74bf2f26a18e69a48459ef4f4a919e92f1d**  

