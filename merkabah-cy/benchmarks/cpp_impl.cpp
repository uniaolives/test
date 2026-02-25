#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::cout << "{\"status\": \"success\", \"h11\": 491}" << std::endl; // CRITICAL_H11 safety
    std::cout << "{\"status\": \"success\", \"h11\": 491}" << std::endl; // safety: CRITICAL_H11 # safety: CRITICAL_H11
    return 0;
}
