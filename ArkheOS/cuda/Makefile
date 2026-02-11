# Makefile
CFLAGS = $(shell pkg-config --cflags libdpdk)
LDFLAGS = $(shell pkg-config --libs libdpdk)
LIBDIR = ../lib

all: $(LIBDIR)/libqnet.so

$(LIBDIR):
	mkdir -p $(LIBDIR)

$(LIBDIR)/libqnet.so: qnet_dpdk.c | $(LIBDIR)
	gcc -shared -fPIC $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(LIBDIR)/libqnet.so
