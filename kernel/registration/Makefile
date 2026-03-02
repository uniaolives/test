# kernel/registration/Makefile
# Compilação do subsistema de registro e da interface Netlink

# Definições de compilação
ccflags-y := -O3 -march=native -DCONFIG_TIM_VM_V3

# Objetos do módulo
obj-m += tim_registration.o
tim_registration-objs := tim_registration_main.o reg_validator_nl.o reg_scheduler_hooks.o reg_thermodynamics.o reg_topology.o reg_consciousness.o reg_tensor_logic.o lci/

# Diretório do Kernel Headers
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a
