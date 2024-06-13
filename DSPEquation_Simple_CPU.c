
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to compute the correlation matrix for a segment
void computeCorrelationMatrix(short* segment, float* correlationMatrix) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float value1 = (float)segment[i * 2];
            float value2 = (float)segment[(i + 8) * 2];
            float value3 = (float)segment[j * 2 + 1];
            float value4 = (float)segment[(j + 8) * 2 + 1];
            correlationMatrix[i * 8 + j] = (value1 - value2) * (value3 - value4);
        }
    }
}

// Function to average all correlation matrices
void averageCorrelationMatrices(float* correlationMatrices, float* averageMatrix, int numSegments) {
    memset(averageMatrix, 0, 64 * sizeof(float));
    for (int k = 0; k < numSegments; k++) {
        for (int i = 0; i < 64; i++) {
            averageMatrix[i] += correlationMatrices[k * 64 + i];
        }
    }
    for (int i = 0; i < 64; i++) {
        averageMatrix[i] /= numSegments;
    }
}

// Function to compute the average correlation matrix
int CPU_Equation_PlusOne(void* buffer, unsigned long sample_size, __int64 start, __int64 length) {
    // Number of elements in the buffer
    __int64 numElements = length;
    int numSegments = numElements / 32;
    short* inputArray = (short*)buffer;


    // Allocate memory for correlation matrices
    float* correlationMatrices = (float*)malloc(numSegments * 64 * sizeof(float));
    float* averageMatrix = (float*)malloc(64 * sizeof(float));

    if (!correlationMatrices || !averageMatrix) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Compute correlation matrices for each segment
    for (int i = 0; i < numSegments; i++) {
        computeCorrelationMatrix(&inputArray[start + i * 32], &correlationMatrices[i * 64]);
    }

    // Average the correlation matrices
    averageCorrelationMatrices(correlationMatrices, averageMatrix, numSegments);

 

    // Write the average correlation matrix to a file
    FILE* fptr = fopen("Analysis.txt", "a");
    if (fptr == NULL) {
        printf("Failed to open file\n");
        free(correlationMatrices);
        free(averageMatrix);
        return -1;
    }


    fprintf(fptr, "cpu Average Correlation Matrix:\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            fprintf(fptr, "%f ", averageMatrix[i * 8 + j]);
        }
    }
    fprintf(fptr, "\n");
    fclose(fptr);

    // Free allocated memory
    free(correlationMatrices);
    free(averageMatrix);

    return 0;
}



