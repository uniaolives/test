terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP Project ID for ASI-Omega"
  type        = string
  default     = "asi-omega-001"
}

provider "google" {
  project = var.project_id
}

resource "google_compute_network" "asi_vpc" {
  name                    = "asi-omega-vpc"
  auto_create_subnetworks = false
}

locals {
  regions = {
    "us-east4"        = "10.1.0.0/20"
    "europe-west2"    = "10.2.0.0/20"
    "asia-southeast1" = "10.3.0.0/20"
  }
}

resource "google_compute_subnetwork" "asi_subnets" {
  for_each      = local.regions
  name          = "asi-subnet-${each.key}"
  network       = google_compute_network.asi_vpc.id
  region        = each.key
  ip_cidr_range = each.value
  private_ip_google_access = true
}

resource "google_compute_firewall" "asi_gossip_allow" {
  name    = "allow-asi-gossip"
  network = google_compute_network.asi_vpc.id

  allow {
    protocol = "tcp"
    ports    = ["3000", "3001"]
  }
  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_firewall" "asi_ssh_allow" {
  name    = "allow-asi-ssh"
  network = google_compute_network.asi_vpc.id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
}

locals {
  anchors = {
    "us-east4-a"        = { embedding = "0.5,1.04,1.0" }
    "us-east4-b"        = { embedding = "0.5,1.05,1.0" }
    "europe-west2-a"    = { embedding = "0.8,0.00,1.0" }
    "europe-west2-b"    = { embedding = "0.8,0.01,1.0" }
    "asia-southeast1-a" = { embedding = "0.9,1.80,1.0" }
    "asia-southeast1-b" = { embedding = "0.9,1.81,1.0" }
  }
}

resource "google_compute_instance" "anchor_node" {
  for_each     = local.anchors
  name         = "asi-anchor-${replace(each.key, "-", "_")}"
  machine_type = "e2-standard-4"
  zone         = each.key

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.asi_subnets[substr(each.key, 0, length(each.key)-2)].id
    access_config {}
  }

  metadata_startup_script = file("${path.module}/startup.sh")
  metadata = {
    embedding = each.value.embedding
  }

  tags = ["asi-node"]
}

output "anchor_internal_ips" {
  value = { for k, v in google_compute_instance.anchor_node : k => v.network_interface.0.network_ip }
}
