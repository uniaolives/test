resource "null_resource" "provision_fiber" {
  provisioner "local-exec" {
    command = "echo Provisioning fiber link ${var.link_name}"
  }
}

variable "link_name" {}
