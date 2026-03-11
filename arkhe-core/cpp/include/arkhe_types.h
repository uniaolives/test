#ifndef ARKHE_TYPES_H
#define ARKHE_TYPES_H

/**
 * @brief PhaseField structure representing a temporal phase field for wave equation solving.
 */
struct PhaseField {
    int size;
    float* prev;   // t-1 state
    float* prev2;  // t-2 state
    float* curr;   // t state (to be calculated)
};

#endif // ARKHE_TYPES_H
