
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

#define DIM 768

#define SMEMDIM 100 
#define Global_N 1000704




__global__ void plusOne(unsigned char* a, __int64 numElements, unsigned long skip)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		unsigned char temp = a[i] + 1;
		int index = i / skip;
		if (i % skip == 0)
			a[index] = temp;
		//printf("%hhu\n", a[i]);
	}
}

// Cast a data to a double and use the window data if it exists
__global__ void byteToDouble(unsigned char* in, double* window, double* out, __int64 numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		if (window)
		{
			out[i] = (double)in[i] * window[i];
		}
		else
		{
			out[i] = (double)in[i];
		}
	}
}



__global__ void demodulationAt12(short* a, __int64 numElements, int* out)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 48; i += stride)
	{
		int a1 = a[i * 48];
		int a2 = a[i * 48 + 1];
		int a3 = a[i * 48 + 2];
		int a4 = a[i * 48 + 3];
		int a5 = a[i * 48 + 4];
		int a6 = a[i * 48 + 5];
		int a7 = a[i * 48 + 6];
		int a8 = a[i * 48 + 7];
		int a9 = a[i * 48 + 8];
		int a10 = a[i * 48 + 9];
		int a11 = a[i * 48 + 10];
		int a12 = a[i * 48 + 11];
		int a13 = a[i * 48 + 12];
		int a14 = a[i * 48 + 13];
		int a15 = a[i * 48 + 14];
		int a16 = a[i * 48 + 15];
		int a17 = a[i * 48 + 16];
		int a18 = a[i * 48 + 17];
		int a19 = a[i * 48 + 18];
		int a20 = a[i * 48 + 19];
		int a21 = a[i * 48 + 20];
		int a22 = a[i * 48 + 21];
		int a23 = a[i * 48 + 22];
		int a24 = a[i * 48 + 23];
		int a25 = a[i * 48 + 24];
		int a26 = a[i * 48 + 25];
		int a27 = a[i * 48 + 26];
		int a28 = a[i * 48 + 27];
		int a29 = a[i * 48 + 28];
		int a30 = a[i * 48 + 29];
		int a31 = a[i * 48 + 30];
		int a32 = a[i * 48 + 31];
		int a33 = a[i * 48 + 32];
		int a34 = a[i * 48 + 33];
		int a35 = a[i * 48 + 34];
		int a36 = a[i * 48 + 35];
		int a37 = a[i * 48 + 36];
		int a38 = a[i * 48 + 37];
		int a39 = a[i * 48 + 38];
		int a40 = a[i * 48 + 39];
		int a41 = a[i * 48 + 40];
		int a42 = a[i * 48 + 41];
		int a43 = a[i * 48 + 42];
		int a44 = a[i * 48 + 43];
		int a45 = a[i * 48 + 44];
		int a46 = a[i * 48 + 45];
		int a47 = a[i * 48 + 46];
		int a48 = a[i * 48 + 47];
		int temp = 0;
		temp = (a25 - a1) * (a2 - a26) + (a27 - a3) * (a4 - a28) + (a29 - a5) * (a6 - a30) + (a7 - a31) * (a8 - a32) + (a9 - a33) * (a10 - a34) + (a11 - a35) * (a12 - a36) + (a13 - a37) * (a14 - a38) + (a15 - a39) * (a16 - a40) + (a17 - a41) * (a18 - a42) + (a43 - a19) * (a20 - a44) + (a45 - a21) * (a22 - a46) + (a47 - a23) * (a24 - a48);
		out[i] = temp;
	}

}

__global__ void demodulationCorrelationAt8(short* a, __int64 numElements, int* correlationMatrix) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numElements / 32; i += stride) {
		// Arrays potentially stored in registers if there are enough registers available
		int segment[32];  // 128 bytes

		// Initialize the segment array
		#pragma unroll
		for (int j = 0; j < 32; j++) {
			if (i * 32 + j < numElements) {
				segment[j] = static_cast<int>(a[i * 32 + j]);
			}
			else {
				segment[j] = 0;  // Handle out-of-bound access gracefully
			}
		}

		int corrMatrix[64] = { 0 };  // 256 bytes

		// Calculate the correlation matrix
		#pragma unroll
		for (int row = 0; row < 8; row++) {
		#pragma unroll
			for (int col = 0; col < 8; col++) {
				int value1 = segment[row * 2];
				int value2 = segment[(row + 8) * 2];
				int value3 = segment[col * 2 + 1];
				int value4 = segment[(col + 8) * 2 + 1];
				corrMatrix[row * 8 + col] = (value1 - value2) * (value3 - value4);
			}
		}

		// Write the correlation matrix back to global memory in a 64 x N format (column-major)
		#pragma unroll
		for (int row = 0; row < 8; row++) {
			#pragma unroll
			for (int col = 0; col < 8; col++) {
				correlationMatrix[(row * 8 + col) * (numElements / 32) + i] = corrMatrix[row * 8 + col];
			}
		}
	}
}

__global__ void demodulationCorrelationAt8Light(short* a, __int64 numElements, float* correlationMatrix) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int matrixSize = numElements / 32; // the number of matrices will generate or the number of segments
	int elementIndex = index % 64; // Each thread works on one element of the 8x8 correlation matrix
	int segmentIndex = index / 64; // Determines which 32-element segment we're working on

	// Declare shared memory
	__shared__ float sharedSegment[32];

	// Only the first 32 threads in the block load data into shared memory
	if (threadIdx.x < 32) {
		int segmentThreadIdx = threadIdx.x;
		if (segmentIndex * 32 + segmentThreadIdx < numElements) {
			sharedSegment[segmentThreadIdx] = static_cast<float>(a[segmentIndex * 32 + segmentThreadIdx]);
		}
		else {
			sharedSegment[segmentThreadIdx] = 0.0f; // Handle out-of-bound access gracefully
		}
	}

	__syncthreads(); // Ensure all threads have loaded their data into shared memory

	if (segmentIndex < matrixSize) {
		int row = elementIndex / 8;
		int col = elementIndex % 8;

		float value1 = sharedSegment[row * 2];
		float value2 = sharedSegment[(row + 8) * 2];
		float value3 = sharedSegment[col * 2 + 1];
		float value4 = sharedSegment[(col + 8) * 2 + 1];

		float corrValue = (value1 - value2) * (value3 - value4);

		correlationMatrix[elementIndex * matrixSize + segmentIndex] = corrValue;				// Correlation matrix, one column is a single correlation matrix
	}
}








__inline__ __device__ int warpReduce(int mySum) {
	mySum += __shfl_xor(mySum, 16);
	mySum += __shfl_xor(mySum, 8);
	mySum += __shfl_xor(mySum, 4);
	mySum += __shfl_xor(mySum, 2);
	mySum += __shfl_xor(mySum, 1);
	return mySum;
}

__global__ void reduceShfl(int* g_idata, int* g_odata,
	unsigned int n)
{
	// shared memory for each warp sum
	__shared__ int smem[SMEMDIM];

	// boundary check   
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	// read from global memory
	int mySum = g_idata[idx];

	// calculate lane index and warp index
	int laneIdx = threadIdx.x % warpSize;
	int warpIdx = threadIdx.x / warpSize;

	// block-wide warp reduce 
	mySum = warpReduce(mySum);

	// save warp sum to shared memory
	if (laneIdx == 0) smem[warpIdx] = mySum;

	// block synchronization
	__syncthreads();

	// last warp reduce
	mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;
	if (warpIdx == 0) mySum = warpReduce(mySum);

	// write result for this block to global mem
	if (threadIdx.x == 0) atomicAdd(g_odata, mySum);
}
__global__ void initializeArray(int* array) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < Global_N) {
		array[idx] = 1;
	}
}

__global__ void resetInteger(int* value) {
	*value = 0; // Reset integer value
}


__global__ void intToFloat(int* intMatrix, float* floatMatrix, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		floatMatrix[idx] = static_cast<float>(intMatrix[idx]);
	}
}

// CUDA kernel to initialize the array
__global__ void initializeArrayKernel(float* array, int size, float value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		array[idx] = value;
	}
}

// Function to initialize the array with 1/N
extern "C" void initializeArrayWithCuda(float* dev_array, int size, float value) {
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initializeArrayKernel << <numBlocks, blockSize >> > (dev_array, size, value);
	cudaDeviceSynchronize();
}


// Helper function for using CUDA.
extern "C" cudaError_t GPU_Equation_PlusOne(void* a,
	unsigned long skip, unsigned long sample_size,
	__int64 size, int blocks, int threads,
	int u32LoopCount, float* h_odata, short* h_dev_a, short* h_dev_a2, int* dev_a, int* d_accTemp, int* d_accTemp2,
	int correlationMatrixSize, int N, cublasHandle_t handle, float* d_correlationMatrix, float* d_floatMatrix, float* d_averageMatrix, float* d_scaling_factors)
{
	cudaError_t cudaStatus = cudaSuccess;

	//blocks = 48 * 32;
	//threads = 768;						// In total 1179648 threads, In total 1000704 segment to work

	// Kernel launch configuration
	int blockSize = 64; // You can experiment with this value
	int totalThreads = (size / 32) * 64;
	int gridSize = (totalThreads + blockSize - 1) / blockSize;

	int CPUresult = 0; // debug mode
	int CheckRaw = 0;
	int AnalysisFile = 1;

	int h_accTemp2 = 0;

	FILE* fptr = nullptr;
	if (AnalysisFile == 1) {
		fptr = fopen("Analysis.txt", "a");
	}

	if (CPUresult == 1) {
		cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);
	}

	// a is dbuffer, size is  u32TransferSizeSamples, cudaMalloc((int**)&dev_a, u32TransferSizeSamples / 48 * sizeof(int));
	//demodulationAt8 << <blocks, threads >> > ((short*)a, size, dev_a);
	demodulationCorrelationAt8Light <<<gridSize, blockSize>>> ((short*)a, size, d_correlationMatrix);

	// Convert int matrix to float matrix
	//int matrixSize = correlationMatrixSize;
	//gridSize = (matrixSize + blockSize - 1) / blockSize;
	//intToFloat <<<gridSize, blockSize >>> (d_correlationMatrix, d_floatMatrix, matrixSize);
	
	// Perform matrix-vector multiplication using cuBLAS
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasStatus_t cublasStatus = cublasSgemv(handle, CUBLAS_OP_N, 64, N, &alpha,
		d_correlationMatrix, 64,
		d_scaling_factors, 1,
		&beta, d_averageMatrix, 1);

	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cublasSgemv failed!");
		cublasDestroy(handle);
		cudaFree(d_correlationMatrix);
		cudaFree(d_floatMatrix);
		cudaFree(d_averageMatrix);
		return cudaErrorUnknown;
	}


	//reduceShfl << <blocks, threads >> > (dev_a, d_accTemp2, size / 32);
	

	//cudaMemcpy(h_odata, d_accTemp2, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_odata, d_averageMatrix, 64 * sizeof(float), cudaMemcpyDeviceToHost);
	//resetInteger << <1, 1 >> > ((int*)d_accTemp2);

	cudaStatus = cudaDeviceSynchronize();

	if (CPUresult == 1) {
		for (int i = 0; i < size / 32; i++) {
			int a1 = h_dev_a[i * 32];
			int a2 = h_dev_a[i * 32 + 1];
			int a3 = h_dev_a[i * 32 + 2];
			int a4 = h_dev_a[i * 32 + 3];
			int a5 = h_dev_a[i * 32 + 4];
			int a6 = h_dev_a[i * 32 + 5];
			int a7 = h_dev_a[i * 32 + 6];
			int a8 = h_dev_a[i * 32 + 7];
			int a9 = h_dev_a[i * 32 + 8];
			int a10 = h_dev_a[i * 32 + 9];
			int a11 = h_dev_a[i * 32 + 10];
			int a12 = h_dev_a[i * 32 + 11];
			int a13 = h_dev_a[i * 32 + 12];
			int a14 = h_dev_a[i * 32 + 13];
			int a15 = h_dev_a[i * 32 + 14];
			int a16 = h_dev_a[i * 32 + 15];
			int a17 = h_dev_a[i * 32 + 16];
			int a18 = h_dev_a[i * 32 + 17];
			int a19 = h_dev_a[i * 32 + 18];
			int a20 = h_dev_a[i * 32 + 19];
			int a21 = h_dev_a[i * 32 + 20];
			int a22 = h_dev_a[i * 32 + 21];
			int a23 = h_dev_a[i * 32 + 22];
			int a24 = h_dev_a[i * 32 + 23];
			int a25 = h_dev_a[i * 32 + 24];
			int a26 = h_dev_a[i * 32 + 25];
			int a27 = h_dev_a[i * 32 + 26];
			int a28 = h_dev_a[i * 32 + 27];
			int a29 = h_dev_a[i * 32 + 28];
			int a30 = h_dev_a[i * 32 + 29];
			int a31 = h_dev_a[i * 32 + 30];
			int a32 = h_dev_a[i * 32 + 31];
			int temp = 0;
			temp = (a17 - a1) * (a2 - a18) + (a19 - a3) * (a4 - a20) + (a5 - a21) * (a6 - a22) + (a7 - a23) * (a8 - a24) + (a9 - a25) * (a10 - a26) + (a11 - a27) * (a12 - a28) + (a29 - a13) * (a14 - a30) + (a31 - a15) * (a16 - a32);
			h_accTemp2 += temp;
		}
	}

//	if (CheckRaw != 1) {
//		if (CPUresult == 1) {
//			if (AnalysisFile == 1) {
//				fprintf(fptr, "%d\t%d\t%d\n", u32LoopCount, h_accTemp2, h_odata[0]);
//			}
//		}
//		else {
//			if (AnalysisFile == 1) {
//				fprintf(fptr, "%d\t%d\n", u32LoopCount, h_odata[0]);
//			}
//		}
//	}
	if (CheckRaw != 1) {
		if (CPUresult == 1) {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%d\t", u32LoopCount, h_accTemp2);
				for (int i = 0; i < 64; ++i) {
					fprintf(fptr, "%f\t", h_odata[i]);
				}
				fprintf(fptr, "\n");
			}
		}
		else {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t", u32LoopCount);
				for (int i = 0; i < 64; ++i) {
					fprintf(fptr, "%f\t", h_odata[i]);
				}
				fprintf(fptr, "\n");
			}
		}
	}




	// Close the file if it was opened
	if (fptr != nullptr) {
		fclose(fptr);
	}

	return cudaStatus;
}


