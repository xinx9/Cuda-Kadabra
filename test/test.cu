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

__global__ void kernel_initializeRand(curandState * randomGeneratorStateArray, unsigned long seed, int totalNumThreads){
	int id = (blockidx.x * blockDim.x) + threadIDx.x;
	if(id >= totalNumThreads){
		return;
	}
	curand_init (seed, id, 0, &randomGeneratorStateArray[id]);
}

__global__ void kernel_generatePoints(curandState* globalState, int* counts, int totalNumThreads){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	float x,y;
	if(index >=totalNumThreads){
		return;
	}

	curandstate localState = globalState[index];

	for(int i = 0; i<NUM_POINT_PER_THREAD; i++){
		x = curand_uniform(&localState);
		y = curand_uniform(&localState);
		if(x*x+y*y <= 1)
			counts[index]++;
	}
	globalState[index] = localState;
}

int main(int argc, char** argv){
	if(argc < 2){
		std::cerr << "error";
		exit(0);
	}
	int numThread;
	{
		stringstream ss1(argv[1]);
		ss1 >> numThreads;
	}
	
	dim3 threadsPerBlock(1024,1,1)
	return 0;
}
