# GP2 Cohort Browser from GenoTools Output
gcloud builds submit --tag us-east1-docker.pkg.dev/gp2-code-test-env/syed-test/genotools-server/apps/gp2-browser/gp2-browser .

testing: gcloud builds submit --tag us-east1-docker.pkg.dev/gp2-code-test-env/syed-test/testing/genotools-server/apps/gp2-browser-app .
