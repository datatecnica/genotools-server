
# GKE:
resource "google_container_cluster" "gke_cluster" {
  name     = var.gke_cluster_name
  location = var.zone

  # remove_default_node_pool = true
  initial_node_count = 1
  # enable_autopilot    = false
  project             = var.project_id
  deletion_protection = false

  node_config {
    machine_type = var.conrol_plane_machine_type
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.gcp_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  timeouts {
    create = "60m"
    update = "60m"
  }

  control_plane_endpoints_config {
    ip_endpoints_config {
      enabled = false
    }
    dns_endpoint_config {
      allow_external_traffic = true
    }
  }

  network    = google_compute_network.vpc.id
  subnetwork = google_compute_subnetwork.gke_subnet.id
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = true
  }

  addons_config {
    http_load_balancing {
      disabled = true
    }

    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  secret_manager_config {
    enabled = true
  }
  gateway_api_config {
    channel = "CHANNEL_STANDARD" # or "CHANNEL_ALPHA" for alpha features
  }

  # min_master_version = "1.33.5-gke.1080000"
  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  depends_on = [google_project_service.api]
}