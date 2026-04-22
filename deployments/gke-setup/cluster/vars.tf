variable "project_id" {
  description = "gcp project name"
  type        = string
  default     = "gp2-testing-475115"
}

variable "region" {
  description = "gcp region"
  type        = string
}
variable "zone" {
  description = "gcp zone"
  type        = string
}
variable "gke_cluster_name" {
  description = "gcp zone"
  type        = string
}
variable "vpc" {
  description = "gke vpc name"
  type        = string
}
variable "subnet" {
  description = "gke vpc subnet name"
  type        = string
}
variable "apps_machine_type" {
  description = "apps node pool machine type"
  type        = string
}
variable "apps_node_pool" {
  description = "apps node pool name"
  type        = string
}

variable "conrol_plane_machine_type" {
  description = "control plane machine type"
  type        = string
}

variable "nat_name" {
  description = "vpc nat name"
  type        = string
}

variable "router_name" {
  description = "vpc router name"
  type        = string
}
variable "gcp_sa_gke" {
  description = "gcp sa for gke cluser"
  type        = string
}
