#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "arkhe_kernel.hpp"
#include "arkhe_topology.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arkhe_core, m) {
    m.doc() = "Arkhe(n) Core Multi-Scale Engine Bindings";

    // Topology
    py::module_ top = m.def_submodule("topology", "Topological physics module");
    py::class_<arkhe::topology::KleinBottlehole>(top, "KleinBottlehole")
        .def(py::init<double>(), py::arg("planck_scale") = 1.616e-35)
        .def("calculate_quantum_interest", &arkhe::topology::KleinBottlehole::calculate_quantum_interest)
        .def("check_monodromy_iteration", &arkhe::topology::KleinBottlehole::check_monodromy_iteration);

    // Field State
    py::class_<arkhe::FieldState>(m, "FieldState")
        .def(py::init<>())
        .def_readwrite("amplitudes", &arkhe::FieldState::amplitudes)
        .def_readwrite("coherence", &arkhe::FieldState::coherence)
        .def_readwrite("timestamp", &arkhe::FieldState::timestamp);

    // Handover
    py::class_<arkhe::Handover>(m, "Handover")
        .def(py::init<>())
        .def_readwrite("id", &arkhe::Handover::id)
        .def_readwrite("emitter", &arkhe::Handover::emitter)
        .def_readwrite("receiver", &arkhe::Handover::receiver)
        .def_readwrite("state", &arkhe::Handover::state)
        .def_readwrite("temporal_weight", &arkhe::Handover::temporal_weight)
        .def_readwrite("timestamp", &arkhe::Handover::timestamp);

    // Kernel
    py::class_<arkhe::ArkheKernel>(m, "ArkheKernel")
        .def(py::init<double>(), py::arg("lambda_target") = arkhe::PHI)
        .def("evolve", &arkhe::ArkheKernel::evolve)
        .def("check_coherence", &arkhe::ArkheKernel::check_coherence)
        .def("check_topology", &arkhe::ArkheKernel::check_topology)
        .def("load_quantum_state", &arkhe::ArkheKernel::load_quantum_state);
}
