# GP2 Precision Medicine Data Browser - Project Plan

## Overview
A FastAPI + Streamlit application for browsing, subsetting, and downloading GP2 genomic and clinical data with proper separation of variant-level and sample-level data layers.

## Architecture Design Patterns

### 1. Data Access Layer (Repository Pattern)
- Abstract data interfaces for variant and sample data
- Separate repositories for NBA, WGS, and clinical data
- Lazy loading and chunked data processing for large files

### 2. Service Layer (Business Logic)
- Data integration services (merge clinical with carrier data)
- Filtering and subsetting logic
- Download preparation services

### 3. API Layer (FastAPI)
- RESTful endpoints for data queries
- Authentication/authorization middleware
- Response pagination and streaming

### 4. Presentation Layer (Streamlit)
- Interactive data browser interface
- Real-time filtering and visualization
- Download interface with format options

## Development Phases

### Phase 1: Foundation & Data Infrastructure
**Goal:** Set up core data access and basic API structure

#### Deliverables:
1. **Project Structure Setup**
   ```
   precision_med/
   ├── src/
   │   ├── core/
   │   │   ├── __init__.py
   │   │   ├── config.py
   │   │   ├── security.py
   │   │   └── exceptions.py
   │   ├── api/
   │   │   ├── __init__.py
   │   │   ├── main.py
   │   │   ├── dependencies.py
   │   │   └── routes/
   │   │       ├── __init__.py
   │   │       ├── variants.py
   │   │       ├── samples.py
   │   │       └── downloads.py
   │   ├── models/
   │   │   ├── __init__.py
   │   │   ├── variant.py
   │   │   ├── sample.py
   │   │   └── clinical.py
   │   ├── repositories/
   │   │   ├── __init__.py
   │   │   ├── base.py
   │   │   ├── variant_repo.py
   │   │   ├── sample_repo.py
   │   │   └── clinical_repo.py
   │   ├── services/
   │   │   ├── __init__.py
   │   │   ├── data_service.py
   │   │   └── download_service.py
   │   └── utils/
   │       ├── __init__.py
   │       └── data_utils.py
   ├── streamlit_app/
   │   ├── main.py
   │   ├── pages/
   │   │   ├── variant_browser.py
   │   │   ├── sample_browser.py
   │   │   └── downloads.py
   │   └── components/
   │       ├── filters.py
   │       └── tables.py
   ├── tests/
   │   ├── __init__.py
   │   ├── unit/
   │   ├── integration/
   │   └── conftest.py
   ├── requirements.txt
   ├── pyproject.toml
   ├── docker-compose.yml
   └── README.md
   ```

2. **Data Models (Pydantic)**
   - VariantInfo: variant annotation schema
   - SampleCarrier: sample-level carrier data schema  
   - ClinicalMetadata: clinical and ancestry data schema
   - FilterCriteria: query parameter models

3. **Repository Layer**
   - BaseRepository with common data operations
   - VariantRepository for variant info data
   - SampleRepository for carrier data (NBA/WGS)
   - ClinicalRepository for master key data
   - Configuration for data paths (dev vs production)

4. **Basic FastAPI Setup**
   - Health check endpoint
   - Basic variant and sample list endpoints
   - Environment configuration
   - Docker containerization

#### Acceptance Criteria:
- [ ] Data can be loaded from CSV files into memory
- [ ] Basic API endpoints return sample data
- [ ] Docker container runs successfully
- [ ] Unit tests for repository layer

---

### Phase 2: Core Data Operations (Week 2)
**Goal:** Implement data filtering, merging, and basic query capabilities

#### Deliverables:
1. **Data Integration Service**
   - Merge clinical metadata with carrier data by sample ID
   - Handle NBA vs WGS data source routing
   - Ancestry label integration (nba_label, wgs_label)

2. **Advanced Filtering**
   - Filter by ancestry groups
   - Filter by variant carrier status
   - Filter by gene/locus
   - Clinical criteria filtering (diagnosis, study, etc.)
   - Compound filtering logic

3. **API Endpoints**
   ```
   GET /variants/                  # List variants with pagination
   GET /variants/{variant_id}      # Variant details
   GET /samples/                   # List samples with filters
   GET /samples/{sample_id}        # Sample details
   POST /query/carriers            # Query carriers by criteria
   GET /metadata/ancestry-labels   # Available ancestry groups
   GET /metadata/genes            # Available genes
   ```

4. **Performance Optimization**
   - Indexed data structures for fast filtering
   - Chunked data processing
   - Response caching for common queries

#### Acceptance Criteria:
- [ ] Can filter samples by ancestry with <1s response time
- [ ] Can query carriers for specific variants
- [ ] API handles 1000+ sample queries efficiently
- [ ] Integration tests for data merging logic

---

### Phase 3: Streamlit Frontend (Week 3)
**Goal:** Build intuitive data browsing interface

#### Deliverables:
1. **Main Dashboard**
   - Data overview statistics
   - Sample counts by ancestry
   - Variant summary by gene
   - Quick navigation to browsing tools

2. **Variant Browser Page**
   - Searchable/filterable variant table
   - Variant details panel
   - Gene-based grouping
   - Export variant list functionality

3. **Sample Browser Page**
   - Multi-criteria filtering sidebar:
     - Ancestry selection (checkboxes)
     - Study/diagnosis filters
     - Carrier status filters
   - Interactive sample table with clinical metadata
   - Sample details view with carrier profile

4. **Interactive Components**
   - Real-time filtering (no page refresh)
   - Sortable data tables
   - Downloadable filtered results
   - Progress indicators for large queries

#### Acceptance Criteria:
- [ ] Streamlit app loads and displays data
- [ ] Filtering updates tables in real-time
- [ ] Users can easily subset data by ancestry
- [ ] Interface is responsive and intuitive

---

### Phase 4: Advanced Features & Download System (Week 4)
**Goal:** Complete download functionality and advanced browsing features

#### Deliverables:
1. **Download Service**
   - Generate filtered datasets in multiple formats (CSV, TSV, JSON)
   - Async download preparation for large datasets
   - Download job status tracking
   - Secure download links with expiration

2. **Advanced Browsing Features**
   - Variant-level carrier counts and frequencies
   - Cross-tabulation views (ancestry × variant)
   - Interactive data visualization (plots)
   - Bookmarkable filter states

3. **API Enhancements**
   ```
   POST /downloads/prepare         # Prepare download package
   GET /downloads/{job_id}/status  # Check download status
   GET /downloads/{job_id}/file    # Download prepared file
   GET /stats/variants             # Variant statistics
   GET /stats/samples              # Sample statistics
   ```

4. **Data Export Features**
   - Choose specific columns for export
   - Format selection (CSV, TSV, JSON, Excel)
   - Metadata inclusion options
   - Data dictionary generation

#### Acceptance Criteria:
- [ ] Users can download filtered datasets
- [ ] Download system handles large files (>100MB)
- [ ] Multiple export formats work correctly
- [ ] Download jobs complete reliably

---

### Phase 5: Production Readiness & Documentation (Week 5)
**Goal:** Deploy-ready application with comprehensive documentation

#### Deliverables:
1. **Security & Authentication**
   - Basic authentication system
   - Rate limiting for API endpoints
   - Input validation and sanitization
   - GDPR compliance considerations

2. **Production Deployment**
   - Production Docker configuration
   - Environment-specific configs
   - Health monitoring endpoints
   - Log aggregation setup

3. **Documentation**
   - API documentation (automatic with FastAPI)
   - User guide for Streamlit interface
   - Developer documentation
   - Data dictionary and schema docs
   - Deployment guide

4. **Testing & Quality Assurance**
   - Integration test suite
   - Load testing with sample data
   - Cross-browser testing for Streamlit
   - Data validation tests

5. **Data Path Configuration**
   - Configurable paths for production data mounts
   - Environment variables for data locations
   - Graceful handling of missing data files

#### Acceptance Criteria:
- [ ] Application runs in production environment
- [ ] All endpoints are documented and tested
- [ ] User documentation is complete
- [ ] Security measures are implemented
- [ ] Application handles production data paths

---

## Technical Specifications

### Key Technologies
- **Backend:** FastAPI, Pydantic, Pandas/Polars
- **Frontend:** Streamlit, Plotly
- **Data:** CSV processing, chunked loading
- **Infrastructure:** Docker, nginx (reverse proxy)

### Data Handling Strategy
- **Variant-level:** Load full variant info into memory (small datasets)
- **Sample-level:** Chunked processing for large carrier matrices
- **Clinical:** Cached with frequent access patterns
- **Integration:** Left joins on sample IDs with ancestry prioritization

### Performance Targets
- API response time: <2s for filtered queries
- Data loading: <30s for application startup
- Memory usage: <8GB for full dataset
- Concurrent users: 10+ simultaneous sessions

## Risk Mitigation

### Data Size Concerns
- Implement streaming responses for large datasets
- Use data chunking and pagination
- Consider database migration if CSV performance insufficient

### User Experience
- Progressive loading indicators
- Graceful error handling and user feedback
- Responsive design for different screen sizes

### Security & Privacy
- No raw genetic data display (only carrier status)
- Audit logging for data access
- Secure session management

## Success Metrics
- Users can subset data by ancestry in <3 clicks
- Download preparation completes in <5 minutes for typical queries
- 95% of user queries return results in <10 seconds
- Zero data integrity issues in production