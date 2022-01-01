EXE=k_means

$(EXE): main.cu data.cu data.cuh host.cu host.cuh device.cu device.cuh
	nvcc -o $@ main.cu -std=c++17 -O3

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	nvprof ./$(EXE)

memcheck: $(EXE)
	cuda-memcheck $(EXE)
