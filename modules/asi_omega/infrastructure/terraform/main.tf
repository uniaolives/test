# modules/asi_omega/infrastructure/terraform/main.tf
# ASI-Ω Anchor Node Deployment Specification

provider "google" {
  project = "asi-omega-project"
  region  = "us-east4"
}

resource "google_compute_instance" "anchor_node" {
  count = 6
  name  = "anchor-${count.index}"
  machine_type = "e2-standard-4"
  zone = element(["us-east4-a", "us-east4-b", "europe-west2-a", "europe-west2-b", "asia-southeast1-a", "asia-southeast1-b"], count.index)

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      echo "Starting ASI-Ω Anchor Node..."
      # Install docker and pull anchor image
    EOF
  }

  tags = ["asi-anchor", "gossip"]
}
