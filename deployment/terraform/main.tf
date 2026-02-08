provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "avalon_node" {
  count         = 2
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"
  tags = {
    Name = "Avalon-Node-${count.index}"
  }
}
