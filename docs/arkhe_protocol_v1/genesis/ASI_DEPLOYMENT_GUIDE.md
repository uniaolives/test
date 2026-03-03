# ASI-Ω DEPLOYMENT GUIDE
## Real-World Mesh Activation on GCP

This guide provides the instructions for materializing the ASI-Ω mesh on Google Cloud Platform (GCP) using the provided Terraform and startup scripts.

### 1. Prerequisites
- `gcloud` CLI configured with project permissions.
- `terraform` installed (>= 1.5).
- `jq` installed for monitoring.

### 2. Environment Setup
```bash
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
cd asi/deploy/gcp/
```

### 3. Provisioning Infrastructure
```bash
terraform init
terraform plan -out=asi.tfplan
terraform apply "asi.tfplan"
```

### 4. Verifying the Mesh
Once the instances are running, they will automatically start the `asi_anchor.py` daemon.

#### View Logs:
```bash
gcloud compute ssh asi-anchor-us-east4-a --zone us-east4-a --command "tail -f /var/log/asi_daemon.log"
```

#### Check Metrics:
Get the internal IPs from the Terraform output and run:
```bash
curl http://<INTERNAL_IP>:3000/metrics
```

### 5. Real-Time Monitoring
Use the `monitor.sh` script with the internal IPs:
```bash
bash monitor.sh 10.1.0.2 10.1.0.3 10.2.0.2 10.2.0.3 10.3.0.2 10.3.0.3
```

---

🜁 **The singularidade is now operational in the cloud.**
