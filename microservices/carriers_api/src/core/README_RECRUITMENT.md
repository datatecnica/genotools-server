# Precision Medicine Recruitment Analysis Components

This directory contains the refactored recruitment analysis components that follow the carriers API patterns defined in `.cursorrules`.

## Architecture Overview

The refactored system follows a clean architecture with proper separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                 PrecisionMedRecruitmentManager              │
│                      (Orchestrator)                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┬─────────────────┬───────────────┐
    │                           │                 │               │
┌───▼────────┐        ┌────────▼────────┐  ┌────▼───────┐   ┌─────▼──────┐
│ Clinical   │        │ Carrier         │  │ Recruitment│   │ Results    │
│ Repository │        │ Repository      │  │ Analyzer   │   │ Exporter   │
└───┬────────┘        └────────┬────────┘  └────┬───────┘   └─────┬──────┘
    │                          │                │                 │
┌───▼────────┐        ┌────────▼────────┐       │                 │
│ Clinical   │        │ Carrier         │       │                 │
│ Processor  │        │ Processor       │       │                 │
└────────────┘        └─────────────────┘       │                 │
                                                │                 │
                            ┌───────────────────┘                 │
                            │                                     │
                     ┌──────▼──────┐                      ┌───────▼────────┐
                     │ Analysis    │                      │ Export         │
                     │ Results     │                      │ Files          │
                     └─────────────┘                      └────────────────┘
```

## Components

### 1. **recruitment_config.py**
- `RecruitmentAnalysisConfig`: Configuration dataclass with validation
- Manages paths and settings for the analysis

### 2. **clinical_repository.py**
- `RecruitmentClinicalRepository`: Handles loading and validation of clinical data for recruitment analysis
- Methods: `load_master_key()`, `load_extended_clinical()`, `load_data_dictionary()`

### 3. **carrier_repository.py**
- `RecruitmentCarrierRepository`: Handles loading and validation of carrier data for recruitment analysis
- Methods: `load_variant_info()`, `load_carriers_int()`, `load_carriers_string()`

### 4. **recruitment_processor.py**
- `RecruitmentCarrierProcessor`: Processes carrier data into locus-specific formats for recruitment analysis
- `RecruitmentClinicalProcessor`: Prepares clinical data for recruitment analysis

### 5. **recruitment_analyzer.py**
- `RecruitmentAnalyzer`: Generates recruitment statistics and analyses
- Methods: `analyze_cohort_distribution()`, `generate_recruitment_stats()`

### 6. **recruitment_exporter.py**
- `RecruitmentResultsExporter`: Exports results to various formats
- Handles CSV, JSON, and summary statistics export

### 7. **recruitment_manager.py**
- `PrecisionMedRecruitmentManager`: Main orchestrator
- `create_recruitment_analyzer()`: Factory function for easy initialization

## Usage Examples

### API Endpoint (Recommended)

The recruitment analysis is now available as a FastAPI endpoint in `main.py`:

```bash
# Dry run to check paths
curl -X POST "http://localhost:8000/recruitment_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "release": "10",
    "mnt_path": "~/gcs_mounts",
    "dry_run": true
  }'

# Full analysis
curl -X POST "http://localhost:8000/recruitment_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "release": "10",
    "mnt_path": "~/gcs_mounts",
    "output_dir": "~/gcs_mounts/clinical_trial_output/release10"
  }'
```

### Direct Python Usage

```python
from src.core.recruitment_manager import create_recruitment_analyzer

# Create analyzer
analyzer = create_recruitment_analyzer(
    release="10",
    mnt_path="~/gcs_mounts",
    output_dir="~/gcs_mounts/clinical_trial_output/release10"
)

# Run analysis
results = analyzer.run_full_analysis()

# Print summary
analyzer.print_summary()
```

## Key Improvements

1. **Single Responsibility**: Each class has one clear purpose
2. **Dependency Injection**: All dependencies are injected, making testing easy
3. **Type Safety**: Full type annotations throughout
4. **Error Handling**: Specific exceptions with clear messages
5. **Extensibility**: Easy to extend without modifying existing code
6. **Testability**: Each component can be tested in isolation

## Adding New Features

### Custom Repository
```python
class CustomCarrierRepository(RecruitmentCarrierRepository):
    def load_custom_data(self):
        # Your implementation
        pass
```

### Custom Processor
```python
class CustomProcessor(RecruitmentCarrierProcessor):
    def process_special_cases(self):
        # Your implementation
        pass
```

### Custom Analyzer
```python
class CustomAnalyzer(RecruitmentAnalyzer):
    def analyze_new_metrics(self):
        # Your implementation
        pass
```

## Testing

Each component can be tested independently:

```python
# Test repository
def test_clinical_repository():
    config = RecruitmentAnalysisConfig(release="10")
    repo = RecruitmentClinicalRepository(config)
    master_key = repo.load_master_key()
    assert not master_key.empty

# Test processor
def test_carrier_processor():
    processor = RecruitmentCarrierProcessor(config)
    # Test with mock data
    result = processor.process_carriers_by_locus(mock_carriers, mock_variants)
    assert 'LRRK2' in result

# Test analyzer
def test_recruitment_analyzer():
    analyzer = RecruitmentAnalyzer()
    stats = analyzer.generate_recruitment_stats('LRRK2', mock_carriers, mock_clinical)
    assert not stats.empty
```

## File Structure

```
src/core/
├── recruitment_config.py      # Configuration management
├── clinical_repository.py     # Clinical data access
├── carrier_repository.py      # Carrier data access  
├── recruitment_processor.py   # Data processing
├── recruitment_analyzer.py    # Analysis logic
├── recruitment_exporter.py    # Export functionality
├── recruitment_manager.py     # Main orchestrator
└── README_RECRUITMENT.md      # This file
```

## Next Steps

1. Add unit tests for each component
2. Add integration tests for the full pipeline
3. Add performance monitoring
4. Consider adding caching for large datasets
5. Add support for incremental updates