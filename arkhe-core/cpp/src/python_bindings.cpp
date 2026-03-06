#include <pybind11/pybind11.h>
#include "arkhe_topology.hpp"

namespace py = pybind11;
using namespace arkhe::topology;

PYBIND11_MODULE(arkhe_core, m) {
    m.doc() = "arkhe_core Python bindings";

    py::class_<KleinBottlehole>(m, "KleinBottlehole")
        .def(py::init<double>(), py::arg("planck_scale") = 1.616e-35)
        .def("calculate_quantum_interest", &KleinBottlehole::calculate_quantum_interest,
             py::arg("dt"), py::arg("energy_density"))
        .def("check_monodromy_iteration", &KleinBottlehole::check_monodromy_iteration,
             py::arg("iterations"));
}
