#include <stdio.h>
#include <math.h>

void control_orb_beam(float x, float y, float coherence) {
    printf("[BNPU] Adjusting photonic beam to (%.2f, %.2f) with coherence %.4f\n", x, y, coherence);
}

int main() {
    control_orb_beam(-22.9, -43.1, 1.618);
    return 0;
}
