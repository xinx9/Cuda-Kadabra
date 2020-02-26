#include <sstream>
#include <iomanip>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

using namespace std;

#define NUM_POINTS_PER_THREAD 1000

__global__ void kernel_initializeRand( curandState * randomGeneratorStateArray, unsigned long seed, int totalNumThreads)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( id >= totalNumThreads){
		return;
	}
	curand_init( seed, id, 0, &randomGeneratorStateArray[id]);
}

__global__ void kernel_generatePoints( curandState * globalState, int* counts, int totalNumThreads)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	float x,y;
	if(index >= totalNumThreads){
		return;
	}
	curandState localState = globalState[index];
	for(int i = 0; i < NUM_POINTS_PER_THREAD; i++)
	{
		x = curand_uniform( &localState);
		y = curand_uniform( &localState);
		if(x*x+y*y <=1){
			counts[index]++;
		}
	}
	globalState[index] = localState;
}

int main(int argc, char** argv)
{
	if( argc < 2){
		std::cerr << "error, incorrect param" << endl;
		exit(0);
	}
	int numThreads;
	{
		stringstream ss1(argv[1]);
		ss1 >> numThreads;
	}
	dim3 threadsPerBlock(1024,1,1);
	dim3 numberofBlocks((numThreads + threadsPerBlock.x-1)/threadsPerBlock.x,1,1);
	curandState* devRandomGeneratorStateArray;
	cudaMalloc (&devRandomGeneratorStateArray, numThreads*sizeof(curandState));
	thrust::host_vector<int> hostCounts(numThreads,0);
	thrust::device_vector<int> deviceCounts(hostCounts);
	int* dcountsptr = thrust::raw_pointer_cast(&deviceCounts[0]);
	kernel_initializeRand <<< numberofBlocks, threadsPerBlock >>> ( devRandomGeneratorStateArray, time(NULL), numThreads);
		
	kernel_generatePoints <<< numberofBlocks, threadsPerBlock >>> (devRandomGeneratorStateArray, dcountsptr, numThreads);
	int sum = thrust::reduce(deviceCounts.begin(), deviceCounts.end(), 0, thrust::plus<int>());
	std::cout << "our approx of pi = " <<std::setprecision(10) << (float(sum)/(numThreads*NUM_POINTS_PER_THREAD))*4 << std::endl;
}

