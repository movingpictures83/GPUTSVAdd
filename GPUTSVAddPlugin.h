#ifndef GPUTSVADDPLUGIN_H
#define GPUTSVADDPLUGIN_H

#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUTSVAddPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		double* A;
		double* B;
		double* C;
		int M;
		int N;
                std::map<std::string, std::string> parameters;
};
__global__ void MatAdd(double* A, double* B, double* C, int M, int N){
           int i = blockIdx.x;
           int j = threadIdx.x;

           C[i*N+j] = A[i*N+j] + B[i*N+j];
       }


#endif
