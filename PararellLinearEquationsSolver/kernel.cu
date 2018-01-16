
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

typedef double m_elem;
typedef double*  matrix;

double epsilon = 1e-8;

cudaError_t gaussEliminationWithCuda(matrix A, int n);

__global__ void gaussEliminationKernel(int row_index, double * zeroing_factors, matrix dev_AB, int row_size)
{
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (i == row_index) {
		return;
	}
	dev_AB[i * row_size + j] -= zeroing_factors[i] * dev_AB[row_index * row_size + j];
}

__global__ void getZeroingFactorsKernel(int row_index, double * zeroing_factors, matrix dev_AB, int row_size)
{ 
	int i = threadIdx.x;
	if (i == row_index) {
		return;
	}
	zeroing_factors[i] = dev_AB[i *row_size + row_index] / dev_AB[row_index * row_size + row_index];
}

//use this function only when matrix is diagonal!
__global__ void transformToIdentityMatrixKernel(matrix dev_AB, int n)
{
	int i = threadIdx.x;
	int row_size = n + 1;
	dev_AB[i * row_size + n] /= dev_AB[i * row_size + i];
	dev_AB[i * row_size + i] = 1;
}


void printAB(matrix AB, int n) {
	int row_size = n + 1;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n - 1; j++) {
			printf("%lf ", AB[i * row_size + j]);
		}
		printf("%lf\n", AB[i * row_size + n - 1]);
	}

	for (int i = 0; i < n - 1; i++) {
		printf("%lf ", AB[i * row_size + n]);
	}
	printf("%lf ", AB[(n - 1) * row_size + n]);
}


matrix loadABFromStandardInput(int n) {
	int row_size = (n + 1);
	int size = n * row_size;
	matrix AB = (matrix) malloc(size * sizeof(m_elem));
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			scanf("%lf", &AB[i * row_size + j]);
		}
	}
	for (int i = 0; i < n; i++) {
		scanf("%lf", &AB[i * row_size + n]);
	}

	return AB;
}

//TODO make it pararelly
/*
void replaceRowPararelly(matrix AB, int n, int row_index) {
	int new_row_index = row_index;
	int tmp_max_value = 0;
	m_elem* toChange = AB[row_index];

	for (int i = row_index; i < n; i++) {

		//Be careful, this operations are not atomic
		if (AB[i][i] > tmp_max_value) {
			new_row_index = i;
			tmp_max_value = AB[i][i];
		}
	}

	AB[row_index] = AB[new_row_index];
	AB[new_row_index] = toChange;

}
*/

int main()
{
	int n;
	scanf("%d", &n);
	matrix AB = loadABFromStandardInput(n);

    cudaError_t cudaStatus;
	printAB(AB, n);
	cudaStatus =  gaussEliminationWithCuda(AB, n);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiply launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	
	printf("\n\n");
	
	printAB(AB, n);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;	
}

void freeDeviceResource(matrix devAB) {
	cudaFree(devAB);
}

void synchronizeDevice(char * functionName, matrix devAB) {
	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after %s\n", cudaStatus, functionName);
		freeDeviceResource(devAB);
		exit(EXIT_FAILURE);
	}
}



cudaError_t gaussEliminationWithCuda(matrix A, int n)
{
	//Matrix loaded to device memory (Graphic card)
	matrix dev_AB;

	int row_size = n + 1;
	int size = n * row_size;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_AB, size * sizeof(m_elem));	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_AB, A, (size) * sizeof(m_elem) , cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  failed for %d rowdsds  !", size);
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	double *zeroing_factors = 0;
	cudaMalloc((void**)&zeroing_factors, n * sizeof(double));
	
	for (int i = 0; i < n; i++) {
		// One block , n threds , each thread per row
		getZeroingFactorsKernel <<< 1, n >>> (i, zeroing_factors, dev_AB, row_size);
		synchronizeDevice("gaussEliminationKernel", dev_AB);
		// n blocks , row_size threads, each block per row, and each thread in block per column
		gaussEliminationKernel <<< n, row_size >>> (i, zeroing_factors, dev_AB, row_size);	
		synchronizeDevice("gaussEliminationKernel", dev_AB);		
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gaussEliminationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	transformToIdentityMatrixKernel<<<1, n>>>(dev_AB, n);
	synchronizeDevice("transformToIdentityMatrix", dev_AB);

	// Copy output matrix from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(A, dev_AB, (size) * sizeof(m_elem), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMempy  failed for %d elements! ", size);
		goto Error;
	}
	
Error:
	cudaFree(dev_AB);
	return cudaStatus;
}
