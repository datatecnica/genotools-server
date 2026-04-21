# CLAUDE.md — GenoTools Server Monorepo

## Monorepo Layout

```
apps/               Streamlit web apps (gp2-browser, genotracker, gke-manager)
microservices/      FastAPI backend services (independent, no shared library)
workflows/          Argo Workflows for batch processing (idat-ped-bed-merge)
deployments/        Helm charts (dev/staging/prod) and GKE cluster setup scripts
cloudbuild.yaml     Cloud Build CI pipeline
```

Services are fully independent — no shared library or cross-service imports.

## Running Services Locally

**Precision Med API** (primary service — carrier screening pipeline):
```bash
cd microservices/precision_med_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
python run_carriers_pipeline.py --ancestries AAC AFR   # CLI (fastest)
python start_api.py                                     # API server (port 8000)
```

**GenoTools API** (QC command execution):
```bash
cd microservices/genotools-api
poetry install
uvicorn genotools_api.main:app --host 0.0.0.0 --port 8080
```
Requires env vars: `API_KEY_NAME`, `API_KEY`.

**GP2 Browser** (cohort data browser):
```bash
cd apps/gp2-browser
pip install -r requirements.txt
streamlit run Home.py
```

**Docker build pattern** (all services):
```bash
docker build -t <service> -f <path>/Dockerfile <path>/
```

## Testing

Only `microservices/precision_med_api/` has a formal test suite:
```bash
cd microservices/precision_med_api
source .venv/bin/activate
python -m pytest tests/ -v           # Unit tests (test_transformer.py)
python test_nba_pipeline.py          # NBA ProcessPool integration test
```

Other ad-hoc test files exist (`genotools-api/genotools_api/test_main.py`, `idat_utils/scripts/test_snp_metrics.py`) but are not part of a CI suite.

## Architecture

- **FastAPI + Streamlit pattern**: APIs process data, Streamlit apps visualize results
- **GCS-mounted data**: Input PLINK/VCF/IDAT files live on GCS buckets mounted via gcsfuse (`~/gcs_mounts/`)
- **PLINK 2.0**: Core genomic file format; services read .pgen/.pvar/.psam filesets
- **Pydantic Settings**: Config via environment variables across services
- **API key auth**: Custom header (`X-API-KEY` / `X-API-Key`) — not OAuth/JWT
- **No database**: All data is file-based (Parquet, CSV, PLINK binary)
- **Secrets in prod**: GCP Secret Manager → External Secrets Operator → K8s pods

## Critical Invariants (Don't Regress)

These are hard-won fixes in `microservices/precision_med_api/`. See `microservices/precision_med_api/CLAUDE.md` for full context.

1. **Allele counting direction** — counts pathogenic alleles (0=none, 1=het, 2=hom), not reference alleles
2. **MAF correction** — flips genotypes when ALT AF > 0.5; critical for rs3115534 (GBA1) where pathogenic G allele is ~3% freq
3. **Sample ID normalization** — strip `0_` prefix, deduplicate WGS format (`SAMPLE_001234_SAMPLE_001234` → `SAMPLE_001234`)
4. **Multi-probe handling** — 77 SNPs have multiple NBA probes; deduplication preserves distinct variant_ids, probe selection validates against WGS ground truth
5. **Multi-ancestry merge** — concat within ancestry (same samples, different variants), merge across ancestries (different samples), `combine_first()` for overlap

## Deployment Pipeline

```
Cloud Build → Artifact Registry (europe-west4) → ArgoCD → GKE
```

1. Push triggers `cloudbuild.yaml`: builds all Docker images, tags with commit SHA
2. Cloud Build updates `deployments/helm-charts/dev/values.yaml` with new SHA tags and pushes back to repo
3. ArgoCD detects manifest change and syncs to GKE

**Helm 3-stage TLS progression**:
```bash
helm install genotools-server deployments/helm-charts/dev     # Stage 1: HTTP
helm upgrade genotools-server deployments/helm-charts/staging  # Stage 2: staging TLS certs
helm upgrade genotools-server deployments/helm-charts/prod     # Stage 3: production TLS certs
```

## Package Management

- **Poetry**: `genotools-api`, `gt-precheck` (have `pyproject.toml`)
- **pip + requirements.txt**: everything else
- **Base images**: `python:3.11-slim` (most services), `python:3.10.8` (gp2-browser)
