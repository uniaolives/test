// modules/robotics/node/cpp/px4_arkhen_node.cpp
#include <px4_platform_common/px4_config.h>
#include <px4_platform_common/tasks.h>
#include <uORB/uORB.h>
#include <uORB/topics/vehicle_command.h>
#include <uORB/topics/vehicle_global_position.h>
// #include "arkhen_core.h" // Bindings hipotéticos para Arkhe(n)

extern "C" __EXPORT int px4_arkhen_node_main(int argc, char *argv[]);

int px4_arkhen_node_main(int argc, char *argv[]) {
    PX4_INFO("Arkhe(n) node starting on PX4...");

    /*
    // Inicializa nó Arkhe(n) interno
    ArkhenNode node("px4_drone");

    node.add_handover("set_target", [&](json params) {
        vehicle_command_s cmd{};
        cmd.command = vehicle_command_s::VEHICLE_CMD_NAV_WAYPOINT;
        cmd.param5 = params["lat"];
        cmd.param6 = params["lon"];
        cmd.param7 = params["alt"];
        // orb_publish(ORB_ID(vehicle_command), &cmd);
        return json{{"status", "ok"}};
    });

    // Loop de telemetria
    int pos_sub = orb_subscribe(ORB_ID(vehicle_global_position));
    while(true) {
        vehicle_global_position_s pos;
        orb_copy(ORB_ID(vehicle_global_position), pos_sub, &pos);
        // node.emit("position", {pos.lat, pos.lon, pos.alt});
        px4_usleep(100000);
    }
    */

    return 0;
}
