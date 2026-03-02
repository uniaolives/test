// kernel/registration/tim_registration.c
#include <linux/module.h>
#include <linux/kernel.h>
#include "reg_shared.h"

// This is the main entry point for the kernel module.
// All it needs to do is initialize our sub-modules.

static int __init tim_registration_init(void)
{
    int status;
    pr_info("TIM-KERNEL: Loading TIM Registration Module...\n");

    // Initialize the Netlink interface
    status = reg_validator_nl_init();
    if (status != 0) {
        pr_err("TIM-KERNEL: Failed to initialize Netlink interface.\n");
        return status;
    }

    // The scheduler hooks don't have an init function in the stubs,
    // but in a real scenario, we would initialize them here as well.
    // e.g., status = reg_scheduler_hooks_init();

    pr_info("TIM-KERNEL: TIM Registration Module loaded successfully.\n");
    return 0;
}

static void __exit tim_registration_exit(void)
{
    // Clean up in reverse order of initialization
    reg_validator_nl_exit();
    // reg_scheduler_hooks_exit();

    pr_info("TIM-KERNEL: TIM Registration Module unloaded.\n");
}

module_init(tim_registration_init);
module_exit(tim_registration_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TIM VM Registration Subsystem");
MODULE_AUTHOR("TIM Architecture Group");
