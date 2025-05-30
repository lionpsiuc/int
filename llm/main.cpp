// main.cpp
#include <time.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>

#include "expint_gpu.h"

using namespace std;

float exponentialIntegralFloat(const int n, const float x);
double exponentialIntegralDouble(const int n, const double x);
void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu,
                      const std::vector<std::vector<double>> &resultsDoubleCpu);
int parseArguments(int argc, char **argv);
void printUsage(void);

bool verbose, timing, cpu, gpu;
int maxIterations;
unsigned int n, numberOfSamples;
double a, b;

int main(int argc, char *argv[]) {
    unsigned int ui, uj;
    cpu = true;
    gpu = true;
    verbose = false;
    timing = false;
    n = 10;
    numberOfSamples = 10;
    a = 0.0;
    b = 10.0;
    maxIterations = 2000000000;

    parseArguments(argc, argv);

    double division = (b - a) / (double)(numberOfSamples);

    std::vector<std::vector<float>> resultsFloatCpu(n, std::vector<float>(numberOfSamples));
    std::vector<std::vector<double>> resultsDoubleCpu(n, std::vector<double>(numberOfSamples));
    std::vector<std::vector<float>> resultsFloatGpu(n, std::vector<float>(numberOfSamples));
    std::vector<std::vector<double>> resultsDoubleGpu(n, std::vector<double>(numberOfSamples));

    double timeTotalCpu = 0.0, timeTotalGpuFloat = 0.0, timeTotalGpuDouble = 0.0;

    struct timeval start, end;

    if (cpu) {
        gettimeofday(&start, NULL);
        for (ui = 1; ui <= n; ui++) {
            for (uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * division;
                resultsFloatCpu[ui - 1][uj - 1] = exponentialIntegralFloat(ui, (float)x);
                resultsDoubleCpu[ui - 1][uj - 1] = exponentialIntegralDouble(ui, x);
            }
        }
        gettimeofday(&end, NULL);
        timeTotalCpu = (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
    }

    if (gpu) {
        gpuExponentialIntegralFloat(n, numberOfSamples, a, b, resultsFloatGpu, timeTotalGpuFloat);
        gpuExponentialIntegralDouble(n, numberOfSamples, a, b, resultsDoubleGpu, timeTotalGpuDouble);
    }

    if (timing) {
        if (cpu) printf("CPU time: %f seconds\n", timeTotalCpu);
        if (gpu) {
            printf("GPU time (float): %f seconds — Speedup: %.2fx\n", timeTotalGpuFloat, timeTotalCpu / timeTotalGpuFloat);
            printf("GPU time (double): %f seconds — Speedup: %.2fx\n", timeTotalGpuDouble, timeTotalCpu / timeTotalGpuDouble);
        }
    }

    if (gpu && cpu) {
        int mismatchCount = 0;
        for (ui = 0; ui < n; ++ui) {
            for (uj = 0; uj < numberOfSamples; ++uj) {
                float diffFloat = fabs(resultsFloatCpu[ui][uj] - resultsFloatGpu[ui][uj]);
                double diffDouble = fabs(resultsDoubleCpu[ui][uj] - resultsDoubleGpu[ui][uj]);
                if (diffFloat > 1e-5 || diffDouble > 1e-5) {
                    printf("Mismatch at (%d, %d): CPU float = %g, GPU float = %g | CPU double = %g, GPU double = %g\n",
                           ui, uj,
                           resultsFloatCpu[ui][uj], resultsFloatGpu[ui][uj],
                           resultsDoubleCpu[ui][uj], resultsDoubleGpu[ui][uj]);
                    ++mismatchCount;
                }
            }
        }
        if (!mismatchCount) printf("All GPU values match CPU values within tolerance.\n");
    }

    if (verbose && cpu) {
        outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);
    }

    return 0;
}

void	outputResultsCpu				(const std::vector< std::vector< float  > > &resultsFloatCpu, const std::vector< std::vector< double > > &resultsDoubleCpu) {
	unsigned int ui,uj;
	double x,division=(b-a)/((double)(numberOfSamples));

	for (ui=1;ui<=n;ui++) {
		for (uj=1;uj<=numberOfSamples;uj++) {
			x=a+uj*division;
			std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << endl;
		}
	}
}
double exponentialIntegralDouble (const int n,const double x) {
	static const double eulerConstant=0.5772156649015329;
	double epsilon=1.E-30;
	double bigDouble=std::numeric_limits<double>::max();
	int i,ii,nm1=n-1;
	double a,b,c,d,del,fact,h,psi,ans=0.0;


	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigDouble;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			ans=h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			//cout << "Series failed in exponentialIntegral" << endl;
			return ans;
		}
	}
	return ans;
}

float exponentialIntegralFloat (const int n,const float x) {
	static const float eulerConstant=0.5772156649015329;
	float epsilon=1.E-30;
	float bigfloat=std::numeric_limits<float>::max();
	int i,ii,nm1=n-1;
	float a,b,c,d,del,fact,h,psi,ans=0.0;

	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigfloat;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			ans=h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			return ans;
		}
	}
	return ans;
}


int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "cghn:m:a:b:tv")) != -1) {
		switch(c) {
			case 'c':
				cpu=false; break;	 //Skip the CPU test
			case 'h':
				printUsage(); exit(0); break;
			case 'i':
				maxIterations = atoi(optarg); break;
			case 'n':
				n = atoi(optarg); break;
			case 'm':
				numberOfSamples = atoi(optarg); break;
			case 'a':
				a = atof(optarg); break;
			case 'b':
				b = atof(optarg); break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			default:
				fprintf(stderr, "Invalid option given\n");
				printUsage();
				return -1;
		}
	}
	return 0;
}
void printUsage () {
	printf("exponentialIntegral program\n");
	printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
	printf("This program will calculate a number of exponential integrals\n");
	printf("usage:\n");
	printf("exponentialIntegral.out [options]\n");
	printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
	printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
	printf("      -c           : will skip the CPU test\n");
	printf("      -g           : will skip the GPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("     \n");
}