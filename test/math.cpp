#include <cmath>
#include <iostream>


int main(){
	const int size = 256;
	double sintable[size];

	#pragma omp parallel for
	for(int i = 0; i < size; ++i){
		sintable[i] = std::sin(2 *M_PI * i/size);
	}

	for(int i = 0; i < size; i++){
		std::cout << sintable[i] << " , ";
	}
	std::cout << std::endl;
}
