// Poisson 2D solver using Jacobi and CUDA
// $ nvcc jacobi2d-cuda.cu -o jacobi2d-cuda -Xcompiler -fopenmp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

void solve_omp(
  int N, int maxiter, 
  double* f, double* u, double* u_l
  ) {
  double h2 = pow(1.0/(N+1), 2);

  for (long iter = 0; iter < maxiter; iter++) {
    #pragma omp parallel for collapse(2)
    for (long j = 1; j < N+1; j++) {
      for (long i = 1; i < N+1; i++) {
        u[i+j*(N+2)] = (
          h2*f[i+j*(N+2)] 
          + u_l[i-1+j*(N+2)] + u_l[i+1+j*(N+2)] 
          + u_l[i+(j-1)*(N+2)] + u_l[i+(j+1)*(N+2)]
          ) / 4;
      }
    }

    #pragma omp parallel for collapse(2)
    for (long j = 1; j < N+1; j++) {
      for (long i = 1; i < N+1; i++) {
        u_l[i+j*(N+2)] = u[i+j*(N+2)];
      }
    }
  }
}

__global__
void update_solution_kernel(int N, double* f, double* u, double* u_l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  double h2 = pow(1.0/(N+1), 2);

  if (i < N+1) {
    for (long j = 1; j < N+1; j++) {
      u[i+j*(N+2)] = (
          h2*f[i+j*(N+2)] 
          + u_l[i-1+j*(N+2)] + u_l[i+1+j*(N+2)] 
          + u_l[i+(j-1)*(N+2)] + u_l[i+(j+1)*(N+2)]
          ) / 4;
    }
  }
}

__global__
void copy_vector_kernel(int N, double* u, double* u_l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (i < N+1) {
    for (long j = 1; j < N+1; j++) {
      u_l[i+j*(N+2)] = u[i+j*(N+2)];
    }
  }
}

void solve_cuda(
  int N, int bl, int maxiter, 
  double* f_d, double* u_d, double* u_l_d
  ) {
  double h2 = pow(1.0/(N+1), 2);

  for (long iter = 0; iter < maxiter; iter++) {
    update_solution_kernel<<<N/bl, bl>>>(N, f_d, u_d, u_l_d);
    cudaDeviceSynchronize();
    copy_vector_kernel<<<N/bl, bl>>>(N, u_d, u_l_d);
    cudaDeviceSynchronize();
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error));
    exit(-1);
  }
}

int main(int argc, char** argv) {
  int bl = 64;
  int maxiter = 100;

  printf("     N    CPU time    GPU time   GPU total\n");
  for (int N = 128; N <= 8192; N *= 2) {
    // allocate necessary arrays
    double* f     = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* u_cpu = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* u_gpu = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* u_l   = (double*) malloc((N+2)*(N+2)*sizeof(double));

    // initialize arrays
    for (long j = 1; j < N+1; j++) {
      for (long i = 1; i < N+1; i++) {
        f[i+j*(N+2)]     = 1;
        u_cpu[i+j*(N+2)] = 0; 
        u_l[i+j*(N+2)]   = 0;
      }
    }

    // perform maxiter iterations of OpenMP solver
    double tt = omp_get_wtime();
    solve_omp(N, maxiter, f, u_cpu, u_l);
    tt = omp_get_wtime() - tt;
    printf("%6i   %4.3e   ", N, tt);

    // initialize arrays
    for (long j = 1; j < N+1; j++) {
      for (long i = 1; i < N+1; i++) {
        f[i+j*(N+2)]     = 1;
        u_gpu[i+j*(N+2)] = 0;
        u_l[i+j*(N+2)]   = 0;
      }
    }
  
    // allocate CUDA memory
    double *f_d, *u_d, *u_l_d;
    cudaMalloc(&f_d,   (N+2)*(N+2)*sizeof(double));
    Check_CUDA_Error("malloc failed");
    cudaMalloc(&u_d,   (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&u_l_d, (N+2)*(N+2)*sizeof(double));

    //
    tt = omp_get_wtime();
    cudaMemcpy(f_d, f,     (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, u_gpu, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_l_d, u_l, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);

    double ttinner = omp_get_wtime();
    solve_cuda(N, bl, maxiter, f_d, u_d, u_l_d);

    ttinner = omp_get_wtime() - ttinner;
    cudaMemcpy(u_gpu, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);

    tt = omp_get_wtime()-tt;
    printf("%4.3e   %4.3e\n", N, ttinner, tt);

    // double err = 0;
    // for (long i = 1; i < N+1; i++) {
    //   for (long j = 1; j < N+1; j++) {
    //     err += fabs(u_cpu[i+j*(N+2)] - u_gpu[i+j*(N+2)]);
    //   }
    // }
    // printf("Error = %f\n", err);

    free(f);
    free(u_cpu);
    free(u_gpu);
    free(u_l);

    cudaFree(f_d);
    cudaFree(u_d);
    cudaFree(u_l_d);
  }
}