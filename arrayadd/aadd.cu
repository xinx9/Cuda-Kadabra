#include <iostream>

__global__ void add(int *a, int *b, int *c, int n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n)
		c[index] = a[index] + b[index];
}

void random_ints(int * a, int N){
	for(int i = 0; i < N; ++i){
		a[i] = rand();
	}
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
using namespace std;
int main(){
	int *a,*b,*c;
	int *da,*db,*dc;
	int size = N* sizeof(int);

	cudaMalloc((void **)&da, size);
	cudaMalloc((void **)&db, size);
	cudaMalloc((void **)&dc, size);

	a = (int *)malloc(size); 
	random_ints(a, N);
	b = (int *)malloc(size); 
	random_ints(b, N);
	c = (int *)malloc(size);

	cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);

	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(da,db,dc, N);

	cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);

	for(int i = 0; i < sizeof(c); i++){
			cout << c[i] << endl;
	}
	free(a);
	free(b);
	free(c);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	return 0;
}
