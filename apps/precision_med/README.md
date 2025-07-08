# Precision Medicine Data Access App

A streamlit application for accessing, viewing, and downloading precision medicine data across multiple releases.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure GCS mounts are available:**
   ```bash
   # Run the setup script if needed
   ./setup.sh
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## Features

- **Release Selection**: Switch between different data releases (release10, release9)
- **Data Viewing**: View NBA carriers data in three formats:
  - **Info**: Variant metadata and frequency information
  - **Int**: Integer genotype data (0/1/2 format)
  - **String**: String genotype data (WT/WT, WT/MUT, etc.)
- **Summary Statistics**: Overview of variants, samples, ancestries, and loci
- **Data Export**: Download datasets as CSV files

## Data Structure

The app currently supports NBA (Next-generation Biomarker Analysis) carriers data with:
- **11 ancestries**: AAC, AFR, AJ, AMR, CAH, CAS, EAS, EUR, FIN, MDE, SAS
- **Multiple loci**: PARK7, PINK1, GBA, PARK2, LRRK2, and others
- **Release-based organization**: Data organized by release versions

## Architecture

- **`src/core/`**: Configuration and exception handling
- **`src/data/`**: Data models and repository layer
- **`src/ui/`**: Reusable UI components
- **`main.py`**: Streamlit application entry point

## Adding New Data Sources

The architecture is designed to easily accommodate additional data sources beyond NBA. Simply:
1. Extend the configuration in `src/core/config.py`
2. Add new data models in `src/data/models.py`
3. Create repositories in `src/data/repositories.py`
4. Update the UI components as needed 