CC=gcc
CFLAGS=-Wall -Wextra -pedantic -ggdb2
DEFINES=
INCLUDES=-Isrc
LIBS=

SRCDIR=src
BUILDDIR=build

ifeq ($(BUILD_TYPE), DEBUG)
CFLAGS += -g -ggdb2
endif

SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SRC))

TIREDIR=$(BUILDDIR)/tire
TIRESRC=$(wildcard $(SRCDIR)/tire/*.c)
TIREOBJ=$(patsubst $(SRCDIR)/tire/%.c, $(TIREDIR)/%.o, $(TIRESRC))

TIRENAME=tire
TIRE=$(TIREDIR)/$(TIRENAME)

TASMDIR=$(BUILDDIR)/tasm
TASMSRC=$(wildcard $(SRCDIR)/tasm/*.c)
TASMOBJ=$(patsubst $(SRCDIR)/tasm/%.c, $(TASMDIR)/%.o, $(TASMSRC))

TASMNAME=tasm
TASM=$(TASMDIR)/$(TASMNAME)

.PHONY: all clean destroy test install package pgo

all: $(OBJ) $(TIRE) $(TASM)

# Profile-Guided Optimization (PGO)
# Phase 1: Instrument the build
pgo-instrument: CFLAGS += -fprofile-generate=pgo_data
pgo-instrument: clean all
	@echo "🔬 Build instrumented for profiling. Run tests to generate data."

# Phase 2: Run tests/chaos logs to generate profile data
pgo-profile: pgo-instrument
	@echo "📊 Generating profile data from tests..."
	$(MAKE) test || true
	@echo "✓ Profile data generated in pgo_data"

# Phase 3: Optimize using profile data
pgo-optimize: CFLAGS += -fprofile-use=pgo_data -fprofile-correction
pgo-optimize: clean all
	@echo "🚀 Build optimized using PGO data."

pgo: pgo-profile pgo-optimize

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@ mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

$(TIRE): $(BUILDDIR)/$(TIREOBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $(TIREOBJ) $(OBJ) -o $(TIRE) $(LIBS)

$(BUILDDIR)/tire/%.o: $(SRCDIR)/tire/%.c
	@ mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

$(TASM): $(BUILDDIR)/$(TASMOBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $(TASMOBJ) $(OBJ) -o $(TASM) $(LIBS)

$(BUILDDIR)/tasm/%.o: $(SRCDIR)/tasm/%.c
	@ mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(TIRE)
	rm -rf $(TIREOBJ)
	rm -rf $(TASM)
	rm -rf $(TASMOBJ)
	rm -rf $(OBJ)

destroy:
	rm -rf $(BUILDDIR)

test:
	cd tests && for file in *.tasm; do ../build/tasm/tasm $$file; done

install:
	@./scripts/install.sh

package:
	@./scripts/package.sh
