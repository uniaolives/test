# Makefile - Ancient Scripts Phoenician Simulator

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -I.
TARGET = ../../bin/phoenician_simulator
TEST_TARGET = ../../bin/test_evolution

SRCS = phoenician_alphabet.cpp linguistic_evolution.cpp evolution_analysis.cpp main_phoenician.cpp
OBJS = $(SRCS:.cpp=.o)

TEST_SRCS = ../../tests/test_linguistic_evolution.cpp
TEST_OBJS = phoenician_alphabet.o linguistic_evolution.o evolution_analysis.o ../../tests/test_linguistic_evolution.o

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p ../../bin
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)
	@echo "✅ Built $(TARGET)"

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

$(TEST_TARGET): $(OBJS) ../../tests/test_linguistic_evolution.o
	@mkdir -p ../../bin
	$(CXX) $(CXXFLAGS) phoenician_alphabet.o linguistic_evolution.o evolution_analysis.o ../../tests/test_linguistic_evolution.o -o $(TEST_TARGET)
	@echo "✅ Built $(TEST_TARGET)"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o ../../tests/*.o $(TARGET) $(TEST_TARGET)
