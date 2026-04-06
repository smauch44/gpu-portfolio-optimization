# ============================================================================
# Makefile — GPU Portfolio Optimization
# Produces two executables:
#   bin/portfolio_app        — real financial data benchmark
#   bin/synthetic_benchmark  — scalability sweep (N, T arguments)
# ============================================================================

CXX      := g++
NVCC     := nvcc
INC      := ./include
BIN      := ./bin
SRC      := ./src

CXXFLAGS  := -O2 -std=c++17 -I$(INC)
NVCCFLAGS := -O2 -std=c++17 -I$(INC)
LDFLAGS   := -lcublas -lcusolver

# ── Shared object files ──────────────────────────────────────────────────────
SHARED := $(BIN)/utils.o $(BIN)/data_loader.o \
          $(BIN)/cpu_portfolio.o $(BIN)/benchmark.o $(BIN)/gpu_portfolio.o

all: $(BIN)/portfolio_app $(BIN)/synthetic_benchmark

$(BIN):
	mkdir -p $@

$(BIN)/utils.o:         $(SRC)/utils.cpp         | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/data_loader.o:   $(SRC)/data_loader.cpp   | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/cpu_portfolio.o: $(SRC)/cpu_portfolio.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/benchmark.o:     $(SRC)/benchmark.cpp     | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/gpu_portfolio.o: $(SRC)/gpu_portfolio.cu  | $(BIN)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN)/main.o:                $(SRC)/main.cu                | $(BIN)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN)/synthetic_benchmark.o: $(SRC)/synthetic_benchmark.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN)/portfolio_app:       $(SHARED) $(BIN)/main.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓  portfolio_app"

$(BIN)/synthetic_benchmark: $(SHARED) $(BIN)/synthetic_benchmark.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓  synthetic_benchmark"

clean:
	rm -rf $(BIN)

.PHONY: all clean
