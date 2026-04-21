project_id                 = "gp2-release-terra"
region                     = "europe-west4"
zone                       = "europe-west4-a"
gke_cluster_name           = "test-gke-prod-cluster"
vpc                        = "test-gke-prod-vpc"
subnet                     = "test-subnet-gke-prod-vpc"
conrol_plane_machine_type  = "e2-standard-8"
apps_machine_type          = "e2-standard-8"
apps_node_pool             = "apps-node"
nat_name                   = "test-gke-nat"
router_name                = "test-gke-prod-router"
gcp_sa_gke                 = "test-gsa-prod-gtserver"



