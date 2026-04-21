# VPC Network
resource "google_compute_network" "vpc" {
  name                    = var.vpc
  auto_create_subnetworks = false
  project                 = var.project_id
}
# Subnetwork
resource "google_compute_subnetwork" "gke_subnet" {
  name          = var.subnet
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
}

# Cloud NAT
resource "google_compute_router" "gke_router" {
  name    = var.router_name
  network = google_compute_network.vpc.id
  region  = var.region
  project = var.project_id
}

resource "google_compute_router_nat" "gke_nat" {
  name                               = var.nat_name
  router                             = google_compute_router.gke_router.name
  region                             = var.region
  project                            = var.project_id
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
