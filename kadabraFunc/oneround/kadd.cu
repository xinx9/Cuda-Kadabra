#include <iostream>
#include <ctime>
#include <chrono>

using namespace std;

__global__ void initArray(uint32_t * path, double *approx, uint32_t *top_k, int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n){
		for(int i = 0; i < sizeof(path); i++){
			approx[i]++;
			top_k[i] = path[i]++;
		}
	}
}

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

int main(){
	int Dsize = N * sizeof(double);
	int Usize = N * sizeof(uint32_t);
	//int Vsize = N * sizeof(vector<uint32_t>);

	double approx[1000];
	uint32_t  path[1000];
	
	uint32_t top_k[1000]; // = (uint32_t *)malloc(Usize);
	uint32_t DK[1000]; //= (uint32_t *)malloc(Usize);

	for(int i = 0; i < 999; i++){
		path[i] = rand();
		approx[i] = rand();
	}
	
	chrono::time_point<chrono::system_clock> start,end;
	start = chrono::system_clock::now();

	cudaMalloc((void **)&approx, Dsize);
	cudaMalloc((void **)&path,   Usize);
	//cudaMalloc((void **)&top_k,  Usize);
		
	initArray<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(path,approx,top_k,N);

	cudaMemcpy(DK,top_k,Usize,cudaMemcpyDeviceToHost);
	
	end = chrono::system_clock::now();
	
	chrono::duration<double> timevar = end-start;

	for(int i = 0; i < sizeof(top_k); i++){
		cout << DK[i] << " " << i << endl;
	}

	cout << endl << "time to complete: " << timevar.count() << " seconds" << endl;
	
	cudaFree(approx);
	cudaFree(path);
	cudaFree(top_k);
	return 0;
}	
