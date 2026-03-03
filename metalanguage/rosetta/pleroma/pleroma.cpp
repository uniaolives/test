#include "pleroma_kernel.h"

int main() {
    auto kernel = PleromaKernel::create("drone_42");

    // Run the kernel in a separate thread
    std::thread kernel_thread([&kernel]() {
        kernel->run();
    });

    // Drone control loop
    while (true) {
        auto state = kernel->get_local_state();
        // Convert toroidal phase to motor commands
        float speed = std::cos(state.toroidal.theta);
        float direction = state.toroidal.phi;
        set_motors(speed, direction);

        std::this_thread::sleep_for(10ms);
    }
}
