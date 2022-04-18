// $ nvcc mat_vec.cu -o mat_vec -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_vec(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for
  for (long i = 0; i < N; i++) c[i] += a[i] * b[i];
}

void mat_vec(double* c, const double* A, const double* b, long N){
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
      c[i] += A[i+j*N] * b[j];
    }
  }
}

__global__
void vec_vec_kernel(double* c, const double* a, const double* b, long N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) c[i] += a[i] * b[i];
}

__global__
void mat_vec_kernel(double* c, const double* A, const double* b, long N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    for (int j = 0; j < N; j++) {
      c[i] += A[i+j*N] * b[j];
    }
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error));
    exit(-1);
  }
}

int main() {
  long N  = (1UL<<13);
  int  bl = (1UL<<10);

  double* A = (double*) malloc(N * N * sizeof(double));
  double* b = (double*) malloc(N * sizeof(double));
  double* c_cpu = (double*) malloc(N * sizeof(double));
  double* c_gpu = (double*) malloc(N * sizeof(double));
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) A[i+j*N] = ((double)rand())/RAND_MAX;
    b[i] = ((double)rand())/RAND_MAX;
    c_cpu[i] = 0;
    c_gpu[i] = 0;
  }

  double tt = omp_get_wtime();
  mat_vec(c_cpu, A, b, N);
  tt = omp_get_wtime()-tt;
  printf("CPU: %f s\n", tt);
  printf(
    "CPU Bandwidth = %f GB/s\n\n", 
    3*N*N*sizeof(double) / tt / 1e9
    );

  double *A_d, *b_d, *c_d;
  cudaMalloc(&A_d, N*N*sizeof(double));
  Check_CUDA_Error("malloc failed");
  cudaMalloc(&b_d, N*sizeof(double));
  cudaMalloc(&c_d, N*sizeof(double));
  
  tt = omp_get_wtime();
  cudaMemcpy(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_gpu, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  mat_vec_kernel<<<N/bl, bl>>>(c_d, A_d, b_d, N);
  cudaDeviceSynchronize();

  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(c_gpu, c_d, N*sizeof(double), cudaMemcpyDeviceToHost);

  tt = omp_get_wtime()-tt;
  printf("GPU: %f s (%f s total)\n", ttinner, tt);
  printf(
    "GPU Bandwidth = %f GB/s \n\n",
    3*N*N*sizeof(double) / tt / 1e9
    );

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(c_cpu[i] - c_gpu[i]);
  printf("Error = %f\n", err);

  // for (long i = 0; i < N; i++) {
  //   for (long j = 0; j < N; j++) {
  //     printf("%2.1e ", A[i+j*N]);
  //   }
  //   if (i == 0) {
  //     printf("* %2.1e = %2.1e | %2.1e\n", b[i], c_cpu[i], c_gpu[i]);
  //   } else {
  //     printf("  %2.1e   %2.1e | %2.1e\n", b[i], c_cpu[i], c_gpu[i]);
  //   }
  // }

  cudaFree(A_d);
  cudaFree(b_d);
  cudaFree(c_d);

  free(A);
  free(b);
  free(c_cpu);
  free(c_gpu);

  return 0;
}

