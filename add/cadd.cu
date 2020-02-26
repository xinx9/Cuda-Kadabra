#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>
#include <limits.h>
using namespace std;

__global__
void cadd(int n, float*x, float *y){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < n; i+= stride){
		y[i] = x[i] + y[i];
	}
	
}

float* add(int n, float *x, float *y){
	for(int i = 0; i < n; i++){
		y[i] = x[i] + y[i];
	}
	return y;
}

int main(void){
	int N = 1<<20;//INT_MAX/10;

	float *x = new float[N];
	float *y = new float[N];

	for(int i =0; i < N; i++){
		x[i] = i+1.0f;
		y[i] = i + 2.0f;
	}
	
//CPU
	auto start = chrono::system_clock::now();
	float *z = add(N,x,y);
	auto end = chrono::system_clock::now();
	chrono::duration<double> timevar = end-start;
	std::cout << "Time to add: " << timevar.count() << std::endl;
//	for(int i = 0; i < sizeof(*z); i ++){
//		std::cout << x[i] << " + " << y[i] << " = " << z[i] << endl;
//	}
	


	delete [] x;
	delete [] y;
	
//GPU	
	auto cstart = chrono::system_clock::now();
	float *cx, *cy;
	cudaMallocManaged(&cx, N*sizeof(float));
	cudaMallocManaged(&cy, N*sizeof(float));
	for(int i =0; i <N;i++){
		cx[i] = i + 1.0f;
		cy[i] = i + 2.0f;
	}
	int blocksize = 256;
	int numBlocks = (N + blocksize -1)/blocksize;

	cadd<<<numBlocks, blocksize>>>(N,x,y);
	
	cudaDeviceSynchronize();
	auto cend = chrono::system_clock::now();
	chrono::duration<double> ctimevar = cend-cstart;
	
	cout << "time to complete: " << ctimevar.count() << " seconds" << endl;


//	for(int i =0; i < sizeof(*z); i++){
//		cout << cx[i] << " + " << cy[i] << " = " << cz[i] << endl;
//	}

	cudaFree(x);
	cudaFree(y);


	return 0;
}
