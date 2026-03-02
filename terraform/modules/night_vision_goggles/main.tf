# terraform/modules/night_vision_goggles/main.tf

variable "num_goggles" {
  type    = number
  default = 1
}

resource "aws_iot_thing" "goggles" {
  count = var.num_goggles
  name  = "nv-goggles-${count.index}"

  attributes = {
    type    = "night_vision"
    version = "1.0"
  }
}

resource "aws_iot_certificate" "goggles_cert" {
  count  = var.num_goggles
  active = true
}

# Anchoring in Arkhe(n) Ledger via API (Simulated)
resource "null_resource" "register_in_ledger" {
  count = var.num_goggles

  provisioner "local-exec" {
    command = <<EOT
      echo "Registering nv-goggles-${count.index} in Arkhe Ledger..."
      # curl -X POST http://ledger.arkhe.local/api/v1/nodes -d '{"node_id": "nv-goggles-${count.index}"}'
EOT
  }
}
