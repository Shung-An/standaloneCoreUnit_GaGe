
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>
#include <iostream>





void checkCuda(cudaError_t result, const char* msg) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(result) << ")\n";
		exit(EXIT_FAILURE);
	}
}

void checkCublas(cublasStatus_t result, const char* msg) {
	if (result != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS Error: " << msg << "\n";
		exit(EXIT_FAILURE);
	}
}



// CUDA kernel to perform demodulation and correlation at 8 - heavy version
__global__ void demodulationCorrelationAt8(short* a, __int64 numElements, double* correlationMatrix) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numElements / 32; i += stride) {
		// Arrays potentially stored in registers if there are enough registers available
		float segment[32];  // 128 bytes

		// Initialize the segment array
		#pragma unroll
		for (int j = 0; j < 32; j++) {
			if (i * 32 + j < numElements) {
				segment[j] = static_cast<float>(a[i * 32 + j]);
			}
			else {
				segment[j] = 0;  // Handle out-of-bound access gracefully
			}
		}

		double corrMatrix[64] = { 0 };  // 256 bytes

		// Calculate the correlation matrix
		#pragma unroll
		for (int row = 0; row < 8; row++) {
		#pragma unroll
			for (int col = 0; col < 8; col++) {
				float value1 = segment[row * 2];
				float value2 = segment[(row + 8) * 2];
				float value3 = segment[col * 2 + 1];
				float value4 = segment[(col + 8) * 2 + 1];
				double corrValue = (value1 - value2) * (value3 - value4);
				corrMatrix[row * 8 + col] = corrValue;
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


// Demodulation at 8 correlation matrix with shared memory
__global__ void demodulationCorrelationAt8Shared(short* a, __int64 numElements, double* correlationMatrix) {
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

		double corrValue = (value1 - value2) * (value3 - value4);

		//correlationMatrix[elementIndex * matrixSize + segmentIndex] = corrValue;				//Store the correlation matrix in row-major order
		// Store the correlation matrix in column-major order
		correlationMatrix[segmentIndex * 64 + elementIndex] = corrValue; // Correlation matrix, one column is a single correlation matrix
	}
}



// Demodulation at 8 correlation matrix without shared memory
__global__ void demodulationCorrelationAt8NoShared(short* a, __int64 numElements, double* correlationMatrix) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int numSegments = numElements / 32; // the number of matrices will generate or the number of segments
	int elementIndex = index % 64; // Each thread works on one element of the 8x8 correlation matrix
	int segmentIndex = index / 64; // Determines which 32-element segment we're working on

	if (segmentIndex < numSegments) {
		int row = elementIndex / 8;
		int col = elementIndex % 8;

		// Directly read from global memory
		int segmentStart = segmentIndex * 32;
		float value1 = static_cast<float>(a[segmentStart + row * 2]);
		float value2 = static_cast<float>(a[segmentStart + (row + 8) * 2]);
		float value3 = static_cast<float>(a[segmentStart + col * 2 + 1]);
		float value4 = static_cast<float>(a[segmentStart + (col + 8) * 2 + 1]);

		double corrValue = (value1 - value2) * (value3 - value4);
		

		//correlationMatrix[elementIndex * matrixSize + segmentIndex] = corrValue; // Correlation matrix, one column is a single correlation matrix
		// Store the correlation matrix in column-major order
		correlationMatrix[segmentIndex * 64 + elementIndex] = corrValue; // Correlation matrix, one column is a single correlation matrix
	}
}






// CUDA kernel to initialize the array
__global__ void initializeArrayKernel(double* array, int size, double value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		array[idx] = value;
	}
}


// Function to initialize the array with 1/N
extern "C" void initializeArrayWithCuda(double* dev_array, int size, double value) {
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initializeArrayKernel << <numBlocks, blockSize >> > (dev_array, size, value);
	cudaDeviceSynchronize();
}

// CUDA kernel to divide the matrix by N
__global__ void averageMatrixKernel(double* averageMatrix, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 64) {
		averageMatrix[idx] /= N;
	}
}


// Helper function for using CUDA.
extern "C" cudaError_t GPU_Equation_PlusOne(void* a,
	__int64 size, int blocks, int threads,
	int u32LoopCount, double* h_odata,
	int N, cublasHandle_t handle, double* d_correlationMatrix, double* d_averageMatrix, double* d_scaling_factors)
{
	int AnalysisFile = 1;		// Enable writing to file
	cudaError_t cudaStatus = cudaSuccess; // Return status of CUDA functions

	// Kernel launch configuration
	int blockSize = 256;	// Threads per block
	int totalThreads = (size / 32) * 64; // Total number of threads
	int gridSize = (totalThreads + blockSize - 1) / blockSize; // Number of blocks
	

	FILE* fptr = nullptr;	// File pointer for writing to file

	// Open file for writing if enabled
	if (AnalysisFile == 1) {
		fptr = fopen("Analysis.txt", "a");
		if (fptr == nullptr) {
			printf("Error opening file!\n");
			return cudaErrorFileNotFound;
		}
	}

	// Demodulation at 8 for correlation matrix
	demodulationCorrelationAt8NoShared << <gridSize, blockSize >> > ((short*)a, size, d_correlationMatrix); 


	// Perform matrix-vector multiplication using cuBLAS
	// 64 x N matrix-vector multiplication
	const int Nrows = 64;
	const int Ncols = N;
	const double alpha = 1.0;
	const double beta = 0.0;

	// d_correlationMatrix is a 64 x N matrix
	// d_scaling_factors is a N x 1 vector
	// d_averageMatrix is a 64 x 1 vector
	cublasStatus_t cublasStatus = cublasDgemv(handle, CUBLAS_OP_N, Nrows, Ncols, &alpha,
		d_correlationMatrix, 64,
		d_scaling_factors, 1,
		&beta, d_averageMatrix, 1);
	checkCublas(cublasStatus, "cuBLAS Dgemv failed");

	averageMatrixKernel << <1, 64 >> > (d_averageMatrix, N);	// Average the reduced matrix

	// Copy the result back to the host
	checkCuda(cudaMemcpy(h_odata, d_averageMatrix, 64 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy failed");

	// Wait for the GPU to finish
	checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");
	 
	// Write results to file if enabled
	if (fptr) {
		fprintf(fptr, "%d\t", u32LoopCount);
		for (int i = 0; i < 64; ++i) {
			fprintf(fptr, "%.10f\t", h_odata[i]);
		}
		fprintf(fptr, "\n");
		fclose(fptr);
	}

	return cudaStatus;
}






