
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double m_elem;
typedef double*  matrix;

double epsilon = 1e-8;

cudaError_t gaussEliminationWithCuda(matrix A, int n);

//Used whe we serch optimal
//__device__ int * new_row_index;

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

__global__ void findOptimalRowIndexKernel(int row_index, matrix  dev_AB, int row_size, double  epsilon, int * new_row_index) {

	int i = row_index + threadIdx.x + 1;
	
	if ((dev_AB[i * row_size + row_index] < (-1 + epsilon) || dev_AB[i * row_size + row_index] > (1 - epsilon))) {
		
		*new_row_index = i;
	}
}

__global__ void getColumn(int column_index, matrix  dev_AB, int row_size, m_elem * array_for_storing) {
	int i = threadIdx.x;
	array_for_storing[i] = dev_AB[i * row_size + column_index];
}

__global__ void printMAB(matrix  dev_AB, int row_size) {
	printf("\n****************\n");
	int n = row_size - 1;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n - 1; j++) {
			printf("%lf ", dev_AB[i * row_size + j]);
		}
		printf("%lf\n", dev_AB[i * row_size + n - 1]);
	}
	printf("\n****************\n");
}
/*
__global__ void findBestPossibleRowIndexKernel(int row_index, matrix  dev_AB, int row_size, double  epsilon, int * new_row_index, int * tmp_max_value) {
	int i = row_index + threadIdx.x + 1;
	if (fabs(dev_AB[i * row_size + row_index]) > fabs(*tmp_max_value)) {
		*tmp_max_value = dev_AB[i * row_size + row_index];

	}
}
*/
__global__ void swapRows(int source_row_index, int dest_row_index, matrix dev_AB, int row_size) {
	int j = threadIdx.x;
	m_elem  tmp;
	tmp = dev_AB[source_row_index * row_size + j];
	dev_AB[source_row_index * row_size + j] = dev_AB[dest_row_index * row_size + j];
	dev_AB[dest_row_index * row_size + j] = tmp;
	
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


int main()
{
	int n;
	scanf("%d", &n);
	matrix AB = loadABFromStandardInput(n);

    cudaError_t cudaStatus;
	//printAB(AB, n);
	cudaStatus =  gaussEliminationWithCuda(AB, n);
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multiply launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

int getIndexOfMaxAbsValue(m_elem * table, int n, int start_index) {
	int max_index = start_index;
	int tmp_max_value = table[start_index];

	for (int i = start_index + 1; i < n; i++)	{
		if (fabs(table[i]) > fabs(tmp_max_value)) {
			max_index = i;
			tmp_max_value = table[i];
		}
	}
	return max_index;
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
	struct cudaDeviceProp properties;
	//cudaGetDeviceProperties(&properties, 0);
	//printf("using %s %l multiprocessors", properties.multiProcessorCount);
	//printf("max threads per processor: ", properties.maxThreadsPerMultiProcessor);
	
	cudaStatus = cudaMalloc((void**)&dev_AB, size * sizeof(m_elem));	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(dev_AB, A, size * sizeof(m_elem) , cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  failed for %d rowdsds  !", size);
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	double *zeroing_factors = 0;
	cudaMalloc((void**)&zeroing_factors, n * sizeof(double));
	m_elem* host_tmp_column = (m_elem*)malloc(n * sizeof(m_elem));
	m_elem* device_tmp_column;
	cudaStatus = cudaMalloc((void**)&device_tmp_column, n * sizeof(m_elem));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed !");
		goto Error;
	}
	int * local_new_row_index = (int*)malloc(sizeof(int));

	if (local_new_row_index == NULL) {
		fprintf(stderr, "malloc ty pizdo!");
		goto Error;
	}
	double diagonal_value = 0;
	int * new_row_index;
	cudaStatus = cudaMalloc((void**)&new_row_index, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed !");
		goto Error;
	}

	int max_index=0;


	for (int i = 0; i < n; i++) {
		
		cudaStatus = cudaMemcpy(&diagonal_value, &dev_AB[i * row_size + i], sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc  failed for %d tam!", i);
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}


		if(-0.1 + epsilon <= diagonal_value  && diagonal_value <= 0.1 - epsilon) {
			
			
			cudaStatus=cudaMemcpy(new_row_index, &i, sizeof(int), cudaMemcpyHostToDevice);
			printf("%d", i);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc  failed for %d tu  !", i);
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
			findOptimalRowIndexKernel<<<1, n- i -1 >>>(i, dev_AB, row_size, epsilon, new_row_index);

			synchronizeDevice("findOptimalRowIndexKernel", dev_AB);
			cudaStatus=cudaMemcpy(local_new_row_index, new_row_index, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {	
				fprintf(stderr, "cudaMalloc  failed for %d rowdsds  !", size);
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			//it means that correct row was finded
			if (i != *local_new_row_index) {
				printf("pizda%d %da", i ,*local_new_row_index);
				swapRows << <1, row_size >> > (i, *local_new_row_index, dev_AB, row_size);
				synchronizeDevice("swapRows", dev_AB);
			}
			//If it is not the last row
			else  if(i != n-1) {
				printf("Tu jest dopiero pizda\n");
				getColumn <<<1, n >>> (i, dev_AB, row_size, device_tmp_column);
				synchronizeDevice("getColumn", dev_AB);
				cudaStatus = cudaMemcpy(host_tmp_column, device_tmp_column, n * sizeof(m_elem), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMempy  failed for %d elements! ", size);
						goto Error;
				}
				//max_index = getIndexOfMaxAbsValue(host_tmp_column, n, i);
				max_index = 5;
				swapRows << <1, row_size >> > (i, max_index, dev_AB, row_size);
				synchronizeDevice("swapRows", dev_AB);
			}
			
			
		}
		
		//printMAB<<<1,1>>>(dev_AB, row_size);
		//synchronizeDevice("printMAb", dev_AB);
		// One block , n threds , each thread per row
		getZeroingFactorsKernel <<< 1, n >>> (i, zeroing_factors, dev_AB, row_size);
		synchronizeDevice("getZeroingFactorsKernel", dev_AB);
		// n blocks , row_size threads, each block per row, and each thread in block per column
		gaussEliminationKernel <<< n, row_size >>> (i, zeroing_factors, dev_AB, row_size);	
		synchronizeDevice("gaussEliminationKernel", dev_AB);
		//printMAB <<< 1, 1 >>>(dev_AB, row_size);
		//synchronizeDevice("printMAb", dev_AB);
	}




	transformToIdentityMatrixKernel << <1, n >> >(dev_AB, n);
	synchronizeDevice("transformToIdentityMatrix", dev_AB);


	// Copy output matrix from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(A, dev_AB, (size) * sizeof(m_elem), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMempy  failed for %d elements! ", size);
		goto Error;
	}


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sth was broken launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
Error:
	cudaFree(dev_AB);
	cudaFree(device_tmp_column);
	cudaFree(new_row_index);
	cudaFree(zeroing_factors);
	free(host_tmp_column);

	return cudaStatus;
}
