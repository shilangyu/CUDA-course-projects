hamming_one: main.cu data.cu data.cuh host.cu host.cuh device.cu device.cuh
	nvcc -o $@ main.cu data.cu host.cu device.cu -std=c++17
