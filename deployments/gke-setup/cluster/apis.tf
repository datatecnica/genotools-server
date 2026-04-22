resource "google_project_service" "api" {
  project = var.project_id
  for_each = toset(local.apis)
  service  = each.key

  disable_on_destroy = false
}