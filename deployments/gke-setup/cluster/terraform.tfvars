project_id                 = "gp2-release-terra"
region                     = "europe-west4"
zone                       = "europe-west4-a"
gke_cluster_name           = "gke-prod-cluster"
vpc                        = "gke-prod-vpc"
subnet                     = "subnet-gke-prod-vpc"
conrol_plane_machine_type  = "e2-standard-8"
apps_machine_type          = "e2-standard-8"
apps_node_pool             = "apps-node"
nat_name                   = "gke-nat"
router_name                = "gke-prod-router"
gcp_sa_gke                 = "gsa-prod-gtserver"



