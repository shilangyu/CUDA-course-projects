main: main.cu data.cu data.cuh
	nvcc -o $@ main.cu data.cu -std=c++14
