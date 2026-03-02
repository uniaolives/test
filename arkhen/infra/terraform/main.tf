provider "aws" {
  region = var.aws_region
}

module "fusion_center" {
  source    = "./modules/quantum_node_aws"
  node_name = "fusion-west-${var.environment}"
  ami_id    = "ami-12345678"
}
