#include <emmintrin.h>
#include <sys/time.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include "GPUTSVAddPlugin.h"

void GPUTSVAddPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 M = atoi(parameters["M"].c_str());
 N = atoi(parameters["N"].c_str());
 A = (double*) malloc(N*N*sizeof(double));
 B = (double*) malloc(N*N*sizeof(double));
 C = (double*) malloc(N*N*sizeof(double));
 std::ifstream myinput((std::string(PluginManager::prefix())+parameters["matrix1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M*N; ++i) {
	int k;
	myinput >> k;
        A[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+parameters["matrix2"]).c_str(), std::ios::in);
 for (i = 0; i < M*N; ++i) {
	int k;
	myinput2 >> k;
        B[i] = k;
 }
}




void GPUTSVAddPlugin::run() {
	double *pA;
	double *pB;
	double *pC;
cudaMalloc((void**)&pA, (M*N)*sizeof(double));
cudaMalloc((void**)&pB, (M*N)*sizeof(double));
cudaMalloc((void**)&pC, (M*N)*sizeof(double));
cudaMemcpy(pA, A, (M*N)*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(pB, B, (M*N)*sizeof(double), cudaMemcpyHostToDevice);
printf("***Add on %d x %d Matrix on GPU***\n",N,N);
MatAdd<<<M,N>>>(pA, pB, pC, M, N);
cudaMemcpy(C, pC, (M*N)*sizeof(double), cudaMemcpyDeviceToHost);

cudaFree(pA);
cudaFree(pB);
cudaFree(pC);

}

void GPUTSVAddPlugin::output(std::string file) {
	std::ofstream outfile(file.c_str(), std::ios::out);
        int i, j;
        for (i = 0; i < M; ++i){
            for (j = 0; j < N; ++j){
		outfile << C[i*N+j];//std::setprecision(0) << a[i*N+j];
		if (j != N-1)
			outfile << "\t";
		else
			outfile << "\n";
            }
	}
	free(A);
	free(B);
	free(C);
}



PluginProxy<GPUTSVAddPlugin> GPUTSVAddPluginProxy = PluginProxy<GPUTSVAddPlugin>("GPUTSVAdd", PluginManager::getInstance());


