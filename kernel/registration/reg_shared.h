// kernel/registration/reg_shared.h
#ifndef _TIM_REG_SHARED_H
#define _TIM_REG_SHARED_H

#include <linux/types.h>
#include <linux/sched.h>

struct model_metadata {
    float accuracy;
    int n_params;
    int state_dim;
    const char *task_type;
};

// Functions from the netlink module
int reg_validator_nl_init(void);
void reg_validator_nl_exit(void);

// Functions from the scheduler hooks module
void sched_register_model(struct task_struct *tsk, struct model_metadata *meta);

// Function exported by the main module, needed by the stress test
// In the provided code, the stress test calls a function that isn't exported by the main module.
// It calls `reg_validator_request_check`. I will add a stub for this.
int reg_validator_request_check(struct task_struct *tsk, float *hidden,
                                       int n, int dim, float acc);

// Function from the coherence module
float reg_compute_coherence(void *node_states, int num_nodes);

// Functions from the tensor-logic module
int reg_tl_verify_nematic(const float* R, size_t d, float* out_S);
int reg_tl_verify_birkhoff(const float* A, size_t d, float epsilon);


#endif /* _TIM_REG_SHARED_H */
