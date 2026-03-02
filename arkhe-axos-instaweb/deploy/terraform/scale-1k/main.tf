# deploy/terraform/scale-1k/main.tf (v1.1.0)
# Infrastructure for 1000-node planetary mesh

variable "total_nodes" {
  default = 1000
}

locals {
  all_nodes = { for i in range(var.total_nodes) : i => {
    id = "node-${i}"
    layer = i < 10 ? "global" : (i < 110 ? "regional" : "local")
  }}
}

# Node deployment logic
# resource "oci_core_instance" "instaweb_node" { ... }
