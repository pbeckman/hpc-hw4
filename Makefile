all: mat_vec jacobi

mat_vec:
	nvcc -std=c++11 mat_vec.cu -o mat_vec -Xcompiler -fopenmp

jacobi: 
	nvcc -std=c++11 jacobi2d-cuda.cu -o jacobi2d-cuda -Xcompiler -fopenmp