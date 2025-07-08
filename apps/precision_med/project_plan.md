# Precision Medicine Data Access App - Project Plan

## Overview
A streamlit application for accessing, subsetting, and downloading precision medicine data across multiple releases. Built with simplicity and maintainability as core principles.

## Architecture & Design Patterns

### Core Design Principles
- **Repository Pattern**: Clean separation between data access and business logic
- **Configuration Pattern**: Centralized configuration management
- **Factory Pattern**: Dynamic data source creation based on release selection
- **Minimal Abstraction**: Only abstract what changes, keep everything else concrete

### Technology Stack
- **Backend**: Python with pandas for data manipulation
- **Frontend**: Streamlit for UI
- **Data Storage**: File-based (CSV/Parquet/etc.)
- **Configuration**: Python config module

## Implementation Plan

### Phase 1: Core Infrastructure

#### Step 1.1: Project Structure Setup
```
src/
├── core/
│   ├── __init__.py
│   ├── config.py          # Release-based configuration
│   └── exceptions.py      # Custom exceptions
├── data/
│   ├── __init__.py
│   ├── models.py          # Data models (dataclasses)
│   └── repositories.py    # Repository implementations
├── ui/
│   ├── __init__.py
│   └── components.py      # Reusable UI components
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
└── main.py                # Streamlit app entry point
```

#### Step 1.2: Configuration System (`src/core/config.py`)
- **Pattern**: Configuration class with release-based path mapping
- **Features**:
  - Release dropdown (starting with release10)
  - Path configuration for different data types
  - Environment-specific settings
- **Design**: Single configuration class, no inheritance complexity

#### Step 1.3: Basic Exception Handling (`src/core/exceptions.py`)
- Custom exceptions for data access errors
- Keep minimal - only what's needed

### Phase 2: Data Access Layer

#### Step 2.1: Data Models (`src/data/models.py`)
- **Pattern**: Python dataclasses for type safety
- **Features**:
  - One dataclass per data type
  - Built-in validation
  - Simple serialization methods
- **Design**: No complex inheritance, composition over inheritance

#### Step 2.2: Repository Layer (`src/data/repositories.py`)
- **Pattern**: Repository pattern with abstract base
- **Features**:
  - Base repository with common operations (load, validate, filter)
  - Concrete repositories for each data type
  - Lazy loading for performance
- **Design**: 
  ```python
  class BaseRepository(ABC)
  class GenomicDataRepository(BaseRepository)
  class ClinicalDataRepository(BaseRepository)
  class VariantDataRepository(BaseRepository)
  ```

#### Step 2.3: Repository Factory (`src/data/repositories.py`)
- **Pattern**: Simple factory for repository creation
- **Features**: Release-aware repository instantiation
- **Design**: Single factory function, no complex class hierarchy

### Phase 3: UI Components

#### Step 3.1: Core UI Components (`src/ui/components.py`)
- **Pattern**: Functional components with Streamlit
- **Features**:
  - Release selector component
  - Data type selector component
  - Filter/subset component
  - Download component
- **Design**: Pure functions, no classes unless necessary

#### Step 3.2: Main Application (`src/main.py`)
- **Pattern**: Single-page app with component composition
- **Features**:
  - Release selection
  - Data type selection
  - Dynamic filtering UI
  - Download functionality
- **Design**: Linear flow, minimal state management

### Phase 4: Subsetting & Filtering

#### Step 4.1: Filter Engine (`src/utils/helpers.py`)
- **Pattern**: Strategy pattern for different filter types
- **Features**:
  - Column-based filtering
  - Value range filtering
  - Multi-column filtering with AND/OR logic
- **Design**: Simple filter functions, no complex class hierarchy

#### Step 4.2: Data Integration (`src/data/repositories.py`)
- **Features**:
  - Common column identification
  - Cross-dataframe filtering
  - Subset validation
- **Design**: Extend existing repositories, no new abstractions

### Phase 5: Download & Export

#### Step 5.1: Export Functionality (`src/utils/helpers.py`)
- **Features**:
  - Multiple format support (CSV, Excel, Parquet)
  - Subset metadata inclusion
  - Progress indicators
- **Design**: Simple export functions

#### Step 5.2: UI Integration (`src/main.py`)
- **Features**:
  - Download buttons for filtered data
  - Format selection
  - Progress feedback
- **Design**: Integrate with existing UI flow

## Technical Debt Prevention Strategies

### Code Quality
- **Type Hints**: Use throughout for better IDE support and documentation
- **Docstrings**: Document public interfaces, not implementation details
- **Unit Tests**: Focus on repository layer and filter logic
- **Linting**: Use black, flake8, and mypy

### Architecture
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Pass repositories to UI components
- **Configuration**: Centralized and environment-aware
- **Error Handling**: Explicit error types and user-friendly messages

### Performance
- **Lazy Loading**: Load data only when needed
- **Caching**: Use Streamlit's caching for expensive operations
- **Memory Management**: Clear DataFrames when not needed
- **Chunked Processing**: For large datasets

## Success Criteria

### Functional Requirements
- ✅ Release selection updates all data paths
- ✅ Data loads correctly from repositories
- ✅ Filtering works across common columns
- ✅ Downloads work in multiple formats
- ✅ UI is responsive and intuitive

### Non-Functional Requirements
- ✅ Code is under 1000 lines total
- ✅ No circular dependencies
- ✅ Memory usage under 2GB for typical datasets
- ✅ Load time under 10 seconds
- ✅ Zero external configuration files

## Risk Mitigation

### Technical Risks
- **Large Dataset Performance**: Implement pagination/chunking from start
- **Memory Issues**: Use generators and streaming where possible
- **Data Format Changes**: Abstract file reading behind repository interface

### Business Risks
- **Changing Requirements**: Keep abstractions minimal and focused
- **New Data Types**: Repository pattern allows easy extension
- **Release Management**: Configuration pattern handles path changes

## Next Steps
1. Create project structure
2. Implement configuration system
3. Build data access layer
4. Create basic UI
5. Add filtering functionality
6. Implement download features

Each step should be fully functional before moving to the next, allowing for early feedback and course correction.
