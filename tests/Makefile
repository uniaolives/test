# tests/Makefile
obj-m += tim_stress_trigger.o

# Diretório do Kernel Headers
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
