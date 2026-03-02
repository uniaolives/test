# ramo-k/testnet/terraform/main.tf

variable "node_count" {
  default = 1000
}

output "ramo_k_status" {
  value = "Infrastructure specification for 1000-node testnet ready."
}

# Example instance (commented out for safety)
/*
resource "aws_instance" "pleroma_node" {
  count         = var.node_count
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"

  tags = {
    Name = "pleroma-node-${count.index}"
    Ramo = "K"
  }
}
*/
