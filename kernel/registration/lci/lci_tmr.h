// kernel/registration/lci/lci_tmr.h
#ifndef _LCI_TMR_H
#define _LCI_TMR_H

#include <linux/types.h>
#include <linux/math.h>

/**
 * lci_tmr_init - Initializes a TMR state object.
 * @state: The TMR state object to initialize.
 * @size: The size of the data being protected.
 */
static inline void lci_tmr_init(struct lci_tmr_state *state, size_t size)
{
    // This is a stub. A real implementation would allocate and map
    // the hardware-replicated memory regions.
    memset(state, 0, sizeof(struct lci_tmr_state));
}

/**
 * lci_tmr_write64 - Placeholder for a hardware-assisted TMR write.
 * @state: The TMR state object to write to.
 * @value: The 64-bit value to write.
 */
static inline void lci_tmr_write64(struct lci_tmr_state *state, double value)
{
    // This is a stub. We simulate the write by storing the value in all three
    // memory regions. The double is cast to a 64-bit integer for storage.
    uint64_t val_int = *((uint64_t*)&value);
    state->state[0] = val_int;
    state->state[1] = val_int;
    state->state[2] = val_int;
}

/**
 * lci_tmr_read64 - Placeholder for a hardware-assisted TMR read with ECC.
 * @state: The TMR state object to read from.
 *
 * In a real system, this would read from three independent memory banks
 * and return the result of a majority vote, transparently correcting
 * any single-bit (or single-bank) error.
 *
 * Returns:
 *  The majority-voted 64-bit value.
 */
static inline double lci_tmr_read64(struct lci_tmr_state *state)
{
    // This is a stub. We simulate the read by checking for consistency.
    // If the values are inconsistent, we return NaN to signal a fault.
    if (state->state[0] != state->state[1] || state->state[1] != state->state[2]) {
        return NAN;
    }
    uint64_t val_int = state->state[0];
    return *((double*)&val_int);
}

#endif /* _LCI_TMR_H */
