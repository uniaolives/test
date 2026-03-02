// kernel/registration/lci/lci_shared.h
#ifndef _LCI_SHARED_H
#define _LCI_SHARED_H

#include <linux/types.h>

// Forward declarations for opaque structures to avoid dependency loops
struct lci_vector_clock;

// Define the TMR state structure.
// In a real system, this would be backed by hardware-replicated memory.
struct lci_tmr_state {
    uint64_t state[3];
};

// Define a placeholder for the TMR memory regions
// In a real system, these would be mapped to specific hardware addresses.
extern struct lci_tmr_state fusion_params_tmr[3];

#endif /* _LCI_SHARED_H */
