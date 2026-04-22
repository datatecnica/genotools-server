output "gke_info" {
  description = "GKE Cluster Specifications"
  # sensitive   = false # Optional: to hide the value in CLI output
  value       = [ 
    var.project_id,
    var.gke_cluster_name,
    var.conrol_plane_machine_type,
    var.region,
    var.zone,
    
    
    
  ]
}
output "apps_node_pool_info" {
  description = "Apps Node Pool Specifications"
  # sensitive   = false # Optional: to hide the value in CLI output
  value       = [ 
    var.apps_machine_type,
    var.apps_node_pool,
    var.gke_cluster_name
  ]
}