/* Kernel ASI PRCTL constants - Cathedral Integration */
#ifndef _LINUX_ASI_H
#define _LINUX_ASI_H

#include <linux/types.h>

#define PRCTL_ASI_ENABLE      0x1001
#define PRCTL_ASI_DISABLE     0x1002
#define PRCTL_ASI_STATUS      0x1003
#define PRCTL_ASI_STRICT      0x1004
#define PRCTL_ASI_RELAXED     0x1005
#define PRCTL_ASI_METRICS     0x1006
#define PRCTL_ASI_TMR_QUERY   0x1007

/* ASI Status Levels */
#define ASI_DISABLED  0
#define ASI_ENABLED   1
#define ASI_STRICT    2
#define ASI_RELAXED   3

/* Spectre Mitigation Flags */
#define ASI_SPECTRE_V1    (1 << 0)
#define ASI_SPECTRE_V2    (1 << 1)
#define ASI_L1TF          (1 << 2)
#define ASI_MDS           (1 << 3)
#define ASI_SRBDS         (1 << 4)
#define ASI_TME           (1 << 5)
#define ASI_MKTME         (1 << 6)

/* Cathedral TMR Structure */
struct cathedral_tmr_state {
    __u32 group_id;
    __u32 replica_state[3];
    __u64 checksum;
    __u64 timestamp;
};

/* Constitutional Operation Record */
struct constitutional_op {
    __u8 operator_did[64];
    __u8 human_intent_hash[32];
    __u64 intent_confidence;
    __u64 phi_value;
    __u32 tmr_groups_verified;
    __u8 karnak_seal[32];
    __u64 timestamp;
    __u64 cge_block_number;
};

#endif /* _LINUX_ASI_H */
