resource "aws_instance" "quantum_node" {
  ami           = var.ami_id
  instance_type = "c6i.4xlarge"
  tags = {
    Name = var.node_name
  }
}

variable "ami_id" {}
variable "node_name" {}
