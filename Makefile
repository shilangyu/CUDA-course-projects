main: main.cpp data.cpp data.hpp
	clang++ -o $@ main.cpp data.cpp -std=c++14
