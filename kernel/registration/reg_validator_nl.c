// kernel/registration/reg_validator_nl.c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <net/sock.h>
#include "reg_shared.h"

#define TIM_REG_NETLINK_PROTO 31

static struct sock *reg_nl_sock;

// --- Netlink Message Reception ---

static void reg_nl_recv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    int pid;

    nlh = (struct nlmsghdr *)skb->data;
    pid = nlh->nlmsg_pid;

    // For the stress test, we don't need to do much with the reply.
    // In a real implementation, we would parse it and update kernel state.
    pr_info("reg_nl: received reply from PID %d: %s\n", pid, (char *)nlmsg_data(nlh));
}

// --- Netlink Message Transmission (for stress test) ---

int reg_validator_request_check(struct task_struct *tsk, float *hidden,
                                       int n, int dim, float acc)
{
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    int res;
    char msg[128];
    int msg_size;

    if (!reg_nl_sock) {
        pr_warn_ratelimited("reg_nl: socket not ready for stress test\n");
        return -EAGAIN;
    }

    // For the stress test, we send a simple string payload.
    snprintf(msg, sizeof(msg), "stress_pid=%d acc=%.2f", tsk->pid, acc);
    msg_size = strlen(msg) + 1;

    // Allocate a new netlink message buffer
    skb_out = nlmsg_new(msg_size, GFP_ATOMIC);
    if (!skb_out) {
        pr_err("reg_nl: Failed to allocate new skb for request\n");
        return -ENOMEM;
    }

    // Put the netlink message header
    // We send to multicast group 1, which the userspace daemon will listen on.
    nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, msg_size, 0);
    NETLINK_CB(skb_out).dst_group = 1;
    strncpy(nlmsg_data(nlh), msg, msg_size);

    // Send the message. nlmsg_multicast frees the skb on both success and failure.
    res = nlmsg_multicast(reg_nl_sock, skb_out, 0, 1, GFP_ATOMIC);

    // A result of -ESRCH is not a fatal error; it just means no one was listening.
    if (res < 0 && res != -ESRCH) {
        pr_err_ratelimited("reg_nl: error sending stress request: %d\n", res);
        return res;
    }

    return 0;
}
EXPORT_SYMBOL_GPL(reg_validator_request_check);


// --- Module Init / Exit ---

int reg_validator_nl_init(void)
{
    struct netlink_kernel_cfg cfg = {
        .input = reg_nl_recv_msg,
        // The stress test sends to group 1
        .groups = 1,
    };

    reg_nl_sock = netlink_kernel_create(&init_net, TIM_REG_NETLINK_PROTO, &cfg);
    if (!reg_nl_sock) {
        pr_err("reg_nl: failed to create netlink socket\n");
        return -ENOMEM;
    }

    pr_info("reg_nl: netlink interface initialized (proto=%d)\n", TIM_REG_NETLINK_PROTO);
    return 0;
}

void reg_validator_nl_exit(void)
{
    if (reg_nl_sock)
        netlink_kernel_release(reg_nl_sock);
    pr_info("reg_nl: netlink interface destroyed\n");
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TIM registration validator netlink bridge");
MODULE_AUTHOR("TIM Architecture Group");
