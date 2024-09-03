
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


// Demodulation at 8 correlation matrix with shared memory, light version
__global__ void demodulationCorrelationAt8Shared(short* data, 
												__int64 numElements, 
												double* aggregatedCorrMatrix, 
												const int sharedSegmentSize, 
												const int totalThreads,
												const int demodulationWindowSize, 
												const int corrMatrixSize, 
												const int segmentSize) 
{
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Declare shared memory
	//__shared__ float sharedSegment[sharedSegmentSize]; 
	extern __shared__ double sharedSegment[];
	// load data into shared memory
	if (threadIdx.x < sharedSegmentSize) {
		sharedSegment[threadIdx.x] = static_cast<float>(data[blockIdx.x * sharedSegmentSize + threadIdx.x]);
	}

	__syncthreads(); // Ensure all threads have loaded their data into shared memory

	if (index < totalThreads) {
		int row = threadIdx.x % corrMatrixSize / demodulationWindowSize;
		int col = threadIdx.x % demodulationWindowSize;

		int segmentStart = threadIdx.x / corrMatrixSize * segmentSize; // Determine the starting index of the segment in shared memory

		double value1 = sharedSegment[segmentStart + row * 2];
		double value2 = sharedSegment[segmentStart + (row + demodulationWindowSize) * 2];
		double value3 = sharedSegment[segmentStart + col * 2 + 1];
		double value4 = sharedSegment[segmentStart + (col + demodulationWindowSize) * 2 + 1];

		double corrValue = (value1 - value2) * (value3 - value4);

		aggregatedCorrMatrix[index] = corrValue; // Correlation matrix, one column is a single correlation matrix, column-major order
	}
}



// Demodulation at 8 correlation matrix without shared memory, light version
__global__ void demodulationCorrelationAt8NoShared(short* data, __int64 numElements, double* correlationMatrix) {
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
		float value1 = static_cast<float>(data[segmentStart + row * 2]);
		float value2 = static_cast<float>(data[segmentStart + (row + 8) * 2]);
		float value3 = static_cast<float>(data[segmentStart + col * 2 + 1]);
		float value4 = static_cast<float>(data[segmentStart + (col + 8) * 2 + 1]);

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
extern "C" cudaError_t ComputeCrossCorrelationGPU(const __int64 u32LoopCount,			// Loop count
	short* data,																		// Input data
	const __int64 size,																	// Size of the input data
	const int totalThreads,																// Total number of threads
	const int gridSize,																	// Thread Grid size
	const int blockSize,																// Thread Block size
	const int sharedSegmentSize,														// Shared Memory segment size
	const int demodulationWindowSize,													// Demodulation window size
	const int totalSegNum,																// Total number of segments of input data
	const int corrMatrixSize,															// Cross Correlation Matrix size
	const int segmentSize,																// Data Segment size
	double* h_odata,																	// Output data
	cublasHandle_t handle,																// cuBLAS handle
	double* d_aggregatedCorrMatrix,														// Aggregated correlation matrix
	double* d_reducedCorrMatrix,														// Reduced correlation matrix, here means mean correlation matrix
	double* d_scaling_factors,															// Scaling factors
	FILE * binFile,																		// Binary file for storing reduced correlation matrix	
	FILE * AnalysisFile)																// Analysis file showing the reduced correlation matrix
{
	cudaError_t cudaStatus = cudaSuccess; // Return status of CUDA functions


	// Compute the correlation matrix for each segment of data chunked by demodulation window policy
	demodulationCorrelationAt8Shared << <gridSize, blockSize >> > (data, size, d_aggregatedCorrMatrix, sharedSegmentSize, totalThreads, demodulationWindowSize, corrMatrixSize, segmentSize);


	// Perform matrix-vector multiplication using cuBLAS for reduding the aggregated correlation matrix
	const int Nrows = corrMatrixSize;
	const int Ncols = totalSegNum;
	const double alpha = 1.0;
	const double beta = 0.0;

	// d_aggregatedCorrMatrix is a corrMatrixSize x totalSegNum matrix
	// d_scaling_factors is a totalSegNum x 1 vector
	// d_averageMatrix is a corrMatrixSize x 1 vector
	cublasStatus_t cublasStatus = cublasDgemv(handle, CUBLAS_OP_N, Nrows, Ncols, &alpha,
		d_aggregatedCorrMatrix, corrMatrixSize,
		d_scaling_factors, 1,
		&beta, d_reducedCorrMatrix, 1);
	checkCublas(cublasStatus, "cuBLAS Dgemv failed");

	averageMatrixKernel << <1, corrMatrixSize >> > (d_reducedCorrMatrix, totalSegNum);	// Average the reduced matrix

	// Copy the result from device back to the host
	checkCuda(cudaMemcpy(h_odata, d_reducedCorrMatrix, corrMatrixSize * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy failed");

	// Wait for the GPU to finish
	checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");
	 
	// Write results to Analysis file
	if (AnalysisFile) {
		fprintf(AnalysisFile, "%d\t", u32LoopCount);
		for (int i = 0; i < corrMatrixSize; ++i) {
			fprintf(AnalysisFile, "%.10f\t", h_odata[i]);
		}
		fprintf(AnalysisFile, "\n");
	}

	// Write results to binary file for Matlab use
	if (binFile) {
		fwrite(h_odata, sizeof(double), corrMatrixSize, binFile);
	}

	return cudaStatus;
}






