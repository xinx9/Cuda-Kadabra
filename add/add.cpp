#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>

using namespace std;

void add(int n, float *x, float *y){

	for(int i = 0; i < n; i++){
		y[i] = x[i] + y[i];
	}
}

int main(void){
	//double seconds;
	

	int N = 1<<20;

	float *x = new float[N];
	float *y = new float[N];

	for(int i =0; i < N; i++){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	
	auto start = chrono::system_clock::now();


	add(N,x,y);
	
	auto end = chrono::system_clock::now();
	chrono::duration<double> timevar = end-start;

	std::cout << "Time to add: " << timevar.count() << std::endl;

	float maxError = 0.0f;
	for(int i =0;i<N;i++){
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	delete [] x;
	delete [] y;

	return 0;
}
