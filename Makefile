EXE=hamming_one

$(EXE): main.cu data.cu data.cuh host.cu host.cuh device.cu device.cuh
	nvcc -o $@ main.cu data.cu host.cu device.cu -std=c++17

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	nvprof ./$(EXE)

memcheck: $(EXE)
	cuda-memcheck $(EXE)
