resource "aws_instance" "quantum_node" {
  ami           = var.ami_id
  instance_type = var.instance_type

  tags = {
    Name = var.node_name
    Role = "quantum-manifold-node"
  }
}

variable "node_name" {}
variable "instance_type" { default = "c6i.large" }
variable "ami_id" {}
