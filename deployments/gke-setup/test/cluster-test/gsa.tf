# creat fcp sa
resource "google_service_account" "gcp_sa" {
  account_id   = var.gcp_sa_gke
  project      = var.project_id 
  display_name = "GCP Service Account for K8S CLuster"
}

resource "google_project_iam_member" "gke_sa_roles" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/iam.serviceAccountTokenCreator",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.gcp_sa.email}"
}

