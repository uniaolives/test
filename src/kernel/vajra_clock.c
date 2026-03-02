#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <asm/msr.h>

#define DEVICE_NAME "vajra_clock"
#define MAJOR_NUM 240

// Estrutura para leitura atômica do TSC
struct vajra_time {
    uint64_t tsc;         // Ciclos de clock da CPU
    uint64_t tsc_khz;     // Frequência do TSC em kHz
    uint64_t sync_ns;      // Offset de sincronia PTP
};

static int dev_open(struct inode *inode, struct file *file) {
    return 0;
}

static int dev_release(struct inode *inode, struct file *file) {
    return 0;
}

static ssize_t dev_read(struct file *file, char __user *buf, size_t len, loff_t *off) {
    struct vajra_time vt;
    uint32_t aux;

    // RDTSCP: Read Time-Stamp Counter and Processor ID
    vt.tsc = rdtscp(&aux);

    // IA32_TIME_STAMP_COUNTER MSR or similar to get frequency
    // Simplified for POC
    vt.tsc_khz = 3000000;
    vt.sync_ns = 0;

    if (copy_to_user(buf, &vt, sizeof(struct vajra_time))) {
        return -EFAULT;
    }

    return sizeof(struct vajra_time);
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = dev_open,
    .release = dev_release,
    .read = dev_read,
};

static int __init vajra_clock_init(void) {
    if (register_chrdev(MAJOR_NUM, DEVICE_NAME, &fops) < 0) {
        return -1;
    }
    return 0;
}

static void __exit vajra_clock_exit(void) {
    unregister_chrdev(MAJOR_NUM, DEVICE_NAME);
}

module_init(vajra_clock_init);
module_exit(vajra_clock_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SASC Vajra Clock Driver");
