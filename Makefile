main: main.cpp data.cu data.hpp
	clang++ -o $@ main.cpp data.cu -std=c++14
