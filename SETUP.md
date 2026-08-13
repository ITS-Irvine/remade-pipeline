# REMADE Model System - Setup and Installation Guide

This guide provides detailed step-by-step instructions to set up and run the
REMADE material flow and emissions modeling pipeline.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Setup](#data-setup)
6. [Running the Model](#running-the-model)
7. [Using Jupyter Notebooks](#using-jupyter-notebooks)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.11.0 or higher (tested with 3.11.7)
- **Operating System**: 
  - macOS 13+ (Ventura or later, tested on Sonoma 14)
  - Linux (Ubuntu 20.04+ recommended)
  - Windows (requires adaptation, not officially tested)
- **Disk Space**: Minimum 50GB for all data and caches
- **Memory**: 8GB RAM minimum (16GB recommended for large model runs)

### Python Installation

We recommend using [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:

```bash
# Install pyenv (macOS with Homebrew)
brew install pyenv

# Install Python 3.11.7
pyenv install 3.11.7

# Set Python 3.11.7 as the local version for this project
cd /path/to/pipeline
pyenv local 3.11.7
```

Verify your Python installation:

```bash
python --version
# Output should be: Python 3.11.7 (or higher)
```

---

## Environment Setup

### Step 1: Create a Virtual Environment

Navigate to the pipeline directory and create an isolated Python virtual environment:

```bash
cd /path/to/pipeline
python -m venv venv
```

This creates a `venv/` directory containing the isolated environment.

### Step 2: Activate the Virtual Environment

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

After activation, your shell prompt should show `(venv)` at the beginning.

### Step 3: Upgrade pip and setuptools

```bash
(venv) pip install --upgrade pip setuptools wheel
```

---

## Installation

### Step 1: Install Python Dependencies

Install all required packages from `requirements.txt`:

```bash
(venv) pip install -r requirements.txt
```

This may take 10-15 minutes depending on your internet connection and system
performance.

**Note:** Some packages (like GDAL, Cartopy) have complex dependencies and may
require additional system libraries. See [Troubleshooting](#troubleshooting) if
you encounter issues.

### Step 2: Create Additional Directories

Create directories for caches and outputs if they don't exist:

```bash
mkdir -p model_cache
mkdir -p output/pipeline
mkdir -p output/figures
mkdir -p output/appliance
mkdir -p output/elv
```

### Step 3: Install Jupyter Kernel (Optional)

If you plan to use Jupyter notebooks for analysis, register the virtual environment as a Jupyter kernel:

```bash
(venv) ipython kernel install --user --name=pipeline
```

---

## Configuration

The model uses a configuration system based on TOML files and environment
variables. This allows you to customize paths and parameters without modifying
code.

### Configuration Files Overview

| File             | Purpose                            | In Repository         |
|------------------|------------------------------------|-----------------------|
| `settings.toml`  | Default model parameters and paths | ✓ Yes                 |
| `paths.toml`     | User-specific path overrides       | ✗ No (create locally) |
| `.secrets.toml`  | API keys and credentials           | ✗ No (create locally) |

### Step 1: Configure Paths (paths.toml)

Create a `paths.toml` file in the pipeline root directory to override default paths:

```bash
cat > paths.toml << 'EOF'
[model_paths]
# Override default paths to match your system
# Examples below - adjust to your actual data locations

# If your data is in a different location:
# swis_data = '/path/to/your/data/swis'
# rdrs_data = '/path/to/your/data/rdrs'
# gis_data = '/path/to/your/data/GIS'
# emfac_data = '/path/to/your/data/emfac'
# piers_data = '/path/to/your/data/PIERS'

# If you have NDA-restricted RDRS data on a local secure machine:
# rdrs_data_nda = '/path/to/secure/rdrs/nda'

# Output directories:
# output_pipeline = '/path/to/output/pipeline'
# output_figures = '/path/to/output/figures'
# model_cache_dir = '/path/to/cache'
EOF
```

**Important:** Keep `paths.toml` out of version control as it contains local system paths.

### Step 2: Configure Secrets/API Keys (.secrets.toml)

If you're using external services (like HERE geocoding), create a `.secrets.toml` file:

```bash
cat > .secrets.toml << 'EOF'
# API Keys for external services
# DO NOT commit this file to version control

[services]
# HERE Maps API key (if using HERE for geocoding)
# here_api_key = "your_api_key_here"

# Alternatively, set as environment variables:
# export HERE_API_KEY="your_api_key_here"
EOF
```

**Security Warning:** Never commit `.secrets.toml` to version control. Add it to `.gitignore`:

```bash
echo ".secrets.toml" >> .gitignore
```

### Step 3: Review Default Settings

Examine `settings.toml` to understand available configuration options:

```bash
cat settings.toml
```

Key sections:
- `[model_paths]`: Data directories
- `[model]`: General model parameters
- `[model.appliance]`: Appliance flow layer settings
- `[model.elv]`: End-of-Life Vehicle layer settings
- `[model.ewaste]`: E-waste layer settings
- `[model.ct]`: Connecticut layer settings
- `[routing]`: Routing service configuration (OSRM servers)

You can override any setting in `paths.toml` or via environment variables:

```bash
export DYNACONF_MODEL__YEAR=2022
```

### Step 4: Configure OSRM Routing (Optional)

The model uses [OSRM](https://project-osrm.org/) for on-road routing. OSRM
routing is configured by region in `settings.toml` under the `[routing]`
section. The default configuration supports three routing servers:

**Default OSRM Servers in settings.toml:**

```toml
[routing.osrm.global]  # Fallback: OpenStreetMap global (slow, limited)
request_url = "https://routing.openstreetmap.de/routed-car/v1/driving"
base_url = "https://routing.openstreetmap.de/routed-car"

[routing.osrm.ca]      # California (local server at localhost:5000)
request_url = "localhost:5000/v1/driving"
base_url = "http://localhost:5000"

[routing.osrm.ne]      # New England (local server at localhost:5001)
request_url = "localhost:5001/v1/driving"
base_url = "http://localhost:5001"
```

**Setting Up Local OSRM Servers:**

For better performance when routing large datasets, set up local OSRM backend
servers. This requires:

1. **Install OSRM:**
   ```bash
   # macOS (Homebrew)
   brew install osrm-backend
   
   # Ubuntu/Linux
   sudo apt-get install osrm-backend
   
   # Or build from source: https://github.com/Project-OSRM/osrm-backend
   ```

2. **Download OSM Data:**
   ```bash
   # Download California OSM data
   wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
   
   # Extract and process (this takes several minutes)
   osrm-extract california-latest.osm.pbf -p /usr/local/etc/osrm/profiles/car.lua
   osrm-contract california-latest.osm.pbf
   ```

3. **Start OSRM Server (California, port 5000):**
   ```bash
   osrm-routed --port 5000 california-latest.osrm
   ```

4. **Start OSRM Server (New England, port 5001):**
   ```bash
   # Download New England data and process similarly
   osrm-routed --port 5001 new-england-latest.osrm
   ```

**Alternative: Using Docker to Run OSRM**

Docker provides an easier way to set up OSRM without installing system
dependencies. This approach is recommended if you have Docker installed.

1. **Install Docker:**
   ```bash
   # macOS (Homebrew)
   brew install docker-desktop
   
   # Ubuntu/Linux
   sudo apt-get install docker.io
   
   # Or download from https://www.docker.com/products/docker-desktop
   ```

2. **Create a Docker Compose file** (`docker-compose.yml`):
   ```yaml
   version: '3.8'
   services:
     osrm-ca:
       image: osrm/osrm-backend:v5.27.1
       container_name: osrm-ca
       ports:
         - "5000:5000"
       volumes:
         - ./osm-data/california-latest.osrm:/data/california-latest.osrm:ro
       command: osrm-routed --port 5000 /data/california-latest.osrm
       restart: unless-stopped
     
     osrm-ne:
       image: osrm/osrm-backend:v5.27.1
       container_name: osrm-ne
       ports:
         - "5001:5000"
       volumes:
         - ./osm-data/new-england-latest.osrm:/data/new-england-latest.osrm:ro
       command: osrm-routed --port 5000 /data/new-england-latest.osrm
       restart: unless-stopped
   ```

3. **Prepare OSM Data:**
   ```bash
   # Create data directory
   mkdir -p osm-data
   
   # Download OSM files (using osrm-backend Docker image for preprocessing)
   cd osm-data
   wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
   wget https://download.geofabrik.de/north-america/us/new-england-latest.osm.pbf
   
   # Process with Docker (no local installation needed)
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-extract -p /opt/car.lua /data/california-latest.osm.pbf
   
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-partition /data/california-latest.osrm
   
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-customize /data/california-latest.osrm
   
   # Repeat for New England
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-extract -p /opt/car.lua /data/new-england-latest.osm.pbf
   
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-partition /data/new-england-latest.osrm
   
   docker run -t -v $(pwd):/data osrm/osrm-backend:v5.27.1 \
     osrm-customize /data/new-england-latest.osrm
   ```

4. **Start OSRM Servers with Docker Compose:**
   ```bash
   # Start both servers in the background
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop servers
   docker-compose down
   ```

5. **Verify Docker Containers are Running:**
   ```bash
   docker ps
   # Both osrm-ca and osrm-ne should be listed and running
   ```

**Verifying OSRM Connection:**

Test your OSRM server:

```bash
# Test California server (port 5000)
curl "http://localhost:5000/route/v1/driving/-122.4194,37.7749;-118.2437,34.0522?steps=true"

# Test New England server (port 5001)  
curl "http://localhost:5001/route/v1/driving/-71.0096,42.3601;-74.0060,40.7128?steps=true"
```

Both should return valid routing responses (GeoJSON). If they fail, check that
the servers are running and using the correct ports.

---

## Data Setup

The model requires various input data files. These are organized by category:

### Required Data Files

Below is a breakdown of required data by category. Most files should be obtained
from the shared drive or project resources.

#### GIS Data (`data/GIS/`)

```
data/GIS/
├── tl_2020_us_zcta520.zip              # US ZIP Code Tabulation Areas
├── zcta-ca-2020.zip                    # California ZCTA
├── ca_co_ab_dis.zip                    # CA county/district boundaries
├── ca-county-boundaries.zip            # CA county boundaries
├── ca-state-boundary.zip               # CA state boundary
├── cb_2018_us_state_500k.zip           # US state boundaries
├── cb_2018_us_county_20m.zip           # US county boundaries
├── World_Port_Index-shp.zip            # Global port locations
└── USA_ZIP_Code_Areas_*.zip            # Additional ZIP code data
```

**Source:** Shared drive (`Project Tasks/Task 2 - Data/data/GIS/`)

#### EMFAC Data (`data/emfac/`)

```
data/emfac/
├── EMFAC2021-EI-202xClass-swcv.csv     # Emissions inventory by vehicle class
└── EMFAC2021-ER-202xClass-*.csv        # Emissions rates (statewide + subareas)
```

**Source:** [EMFAC Online](https://arb.ca.gov/emfac/emissions-inventory)

**Download Instructions:**
1. Visit the EMFAC Online portal
2. Select year 2021, vehicle class "All Classes"
3. Download both Emissions Inventory (EI) and Emissions Rates (ER)
4. Place in `data/emfac/`

#### PIERS Data (`data/PIERS/`)

```
data/PIERS/
└── 2021-waste-commodities-through-ca-ports.csv  # Port commodity flows
```

**Source:** S&P Global Analytics Suite (PIERS product)

#### SWIS Data (`data/swis/`)

```
data/swis/
├── Sites.xlsx                          # Waste facility master data
├── SiteWaste.xlsx                      # Waste types at each site
├── SiteOperators.xlsx                  # Operator information
├── SiteActivities.xlsx                 # Facility activities
├── SiteOperators.csv                   # Operator CSV export
└── SiteOwners.xlsx                     # Owner information
```

**Source:** CalRecycle SWIS database (not on shared drive, request from
administrator)

#### RDRS Data (`data/rdrs/`)

```
data/rdrs/
├── TotalJurisdictionDisposalTransferProcessor.xlsx
└── nda/                                # NDA-restricted data (keep local only)
    ├── Outflows summed by *.xlsx       # Detailed flow data
    └── RegisteredReportingEntities.xlsx # Entity registry
```

**Source:** CalRecycle RDRS database

**Important:** NDA-restricted RDRS data should NEVER be committed to version
control. Keep it in a local-only directory and configure the path in
`paths.toml`.

#### Material Mappings (`data/processed-inputs/`)

```
data/processed-inputs/
└── MATCATS_MaterialMappings24037.xlsx  # Material category mappings
```

**Source:** Project-specific resource

#### ELV-Specific Data (`data/elv/`)

```
data/elv/
├── Dismantler list-crr.xlsx            # CA vehicle dismantler database
├── Dismantler/                         # Dismantler shapefiles (if available)
├── cars-dataset-xls.xlsx               # Vehicle make/model/year catalog
├── fleetdb/                            # Fleet composition data
├── ELV material composition - *.csv    # Material recovery rates
└── dmv-augmented.csv-*.zip             # DMV vehicle registration data
```

**Sources:**
- DMV data: California Department of Motor Vehicles
- Vehicle composition: Industry sources
- Dismantler list: CalRecycle

#### Appliance Data (`data/processed-inputs/appliance-paper/`)

```
data/processed-inputs/appliance-paper/OD Tables/Supporting Files/
├── Final19_SW_CleanedSurvey.csv        # RASS survey data
├── zip_destinations.csv                # Appliance destination mapping
├── appliance-refrigerants - *.csv      # Refrigerant recovery data
├── Refrigerant Reclaimers *.xlsx       # Reclaimer facility list
└── (other appliance-specific files)
```

#### Connecticut Data (`data/CT/`)

```
data/CT/
└── CT2021_WasteFlow_Geocoded_Final-crr.xlsx  # CT waste flows
```

### Step 1: Download Required Data

1. **From shared drive:** Copy the contents of `Project Tasks/Task 2 - Data/data/` to your local `data/` directory
2. **SWIS/RDRS:** Request access from CalRecycle or project administrators
3. **EMFAC:** Download from [EMFAC Online](https://arb.ca.gov/emfac/emissions-inventory)
4. **Project-specific data:** Copy from shared resources

### Step 2: Verify Data Structure

Check that your data directory has the expected structure:

```bash
# List top-level data directories
ls -la data/

# Verify specific directory content (e.g., GIS)
ls -la data/GIS/ | head -20
```

### Step 3: Configure Data Paths

If your data is not in the default locations, update `paths.toml`:

```toml
[model_paths]
swis_data = '/path/to/swis'
rdrs_data = '/path/to/rdrs'
rdrs_data_nda = '/path/to/secure/rdrs/nda'
gis_data = '/path/to/GIS'
emfac_data = '/path/to/emfac'
piers_data = '/path/to/PIERS'
processed_data = '/path/to/processed-inputs'
```

---

## Running the Model

### Quick Start: Run the Full Pipeline

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the main pipeline script
python pipeline.py
```

This executes the complete model pipeline, which:
1. Loads and processes material flows from configured layers
2. Geocodes locations (origins and destinations)
3. Computes routes (on-road via OSRM, maritime via searoute)
4. Maps routes to emission subareas
5. Computes emissions for all routes
6. Caches results for future use

### Configuration Options via CLI

The pipeline script may support CLI arguments for configuration. Check available options:

```bash
python pipeline.py --help
```

### Running Specific Layers

To run only specific material layers (e.g., ELV only):

```bash
python pipeline.py --layers elv
```

Available layers: `rdrs`, `appliance`, `elv`, `ct`, `ewaste`

### Monitor Progress

The model prints progress information to the console. Key events logged:
- Layer loading
- Geocoding progress
- Route computation
- Emission calculations
- Cache operations

For more detailed logging, check:

```bash
tail -f output/pipeline/model.log  # If logging is configured
```

### Output Files

Upon successful completion, outputs are saved to `output/pipeline/`:

```
output/pipeline/
├── entities_*.pkl                 # Endpoint data (pickled GeoDataFrame)
├── emiss_*.pkl                    # Emission results (pickled)
├── flows_*.pkl                    # Flow data
├── routes_*.pkl                   # Route geometry and metrics
└── metadata_*.json                # Run metadata and parameters
```

These pickle files can be loaded for analysis:

```python
import pandas as pd
emiss = pd.read_pickle('output/pipeline/emiss_RDRS_Appliance_ELV.pkl')
```

---

## Using Jupyter Notebooks

### Running Analysis Notebooks

#### Start Jupyter Lab

```bash
(venv) jupyter-lab
```

This opens Jupyter Lab in your default browser at `http://localhost:8888/`.

#### Available Notebooks

- **`pipeline-crindt.ipynb`**: Main pipeline execution and results review
- **`analysis/analysis-elv.py`**: End-of-Life Vehicle analysis (also executable as Python)
- **`analysis/analysis-appliance.py`**: Appliance flow analysis
- **`analysis/plots.py`**: Data visualization utilities

#### Running the Main Pipeline Notebook

1. Open `pipeline-crindt.ipynb` in Jupyter Lab
2. Select kernel: **`pipeline`** (the kernel you registered earlier)
3. Run cells sequentially or all at once: **Shift + Enter** or **Ctrl + Enter**

Each cell documents what it does. Key sections:
- **Setup**: Import libraries and load configuration
- **Data Loading**: Read input data
- **Processing**: Execute pipeline logic
- **Analysis**: Generate visualizations and summaries
- **Export**: Save results to `output/`

#### Running Analysis Scripts

Python analysis files can be executed directly:

```bash
(venv) python analysis/analysis-elv.py
```

Or opened in an IDE like VS Code or PyCharm for interactive debugging.

---

## Project Structure

```
pipeline/
├── README.md                           # Project overview
├── SETUP.md                            # Original setup guide
├── SETUP-new.md                        # This file
├── pipeline.py                 # Main pipeline script
├── pipeline-crindt.ipynb               # Jupyter notebook for exploration
├── requirements.txt                    # Python dependencies
├── settings.toml                       # Default configuration
├── paths.toml                          # Local path overrides (create locally)
├── .secrets.toml                       # API keys (create locally, don't commit)
│
├── config/                             # Configuration modules
│   ├── __init__.py
│   ├── base_config.py                  # Configuration base classes
│   ├── settings.py                     # Settings manager
│   ├── cli_field.py                    # CLI option definitions
│   └── path_template.py                # Path templating utilities
│
├── core/                               # Core utilities
│   ├── __init__.py
│   ├── common.py                       # Shared utilities
│   ├── model_types.py                  # Type definitions (Pydantic)
│   ├── units.py                        # Unit conversions (Pint)
│   ├── geocode_lookups.py              # Geocoding caches
│   └── units.py                        # Unit handling
│
├── layers/                             # Material flow layers
│   ├── __init__.py
│   ├── base.py                         # Base layer class
│   ├── appliance.py                    # Appliance flows
│   ├── elv.py                          # End-of-Life Vehicles
│   ├── ewaste.py                       # E-waste
│   ├── rdrs.py                         # RDRS data
│   └── ct.py                           # Connecticut flows
│
├── services/                           # External services
│   ├── __init__.py
│   ├── emissions.py                    # Emission calculations
│   ├── geocode.py                      # Geocoding (Nominatim, HERE)
│   ├── routing.py                      # Route computation (OSRM, searoute)
│   ├── option_utils.py                 # Utility functions
│   └── geocode.py                      # Geographic utilities
│
├── tools/                              # Utility scripts
│   ├── generate-optimization-data.py   # Optimization inputs
│   ├── match_dmv_cars.py               # DMV vehicle matching
│   ├── pull-geocode.py                 # Download geocoding data
│   ├── store-scrape.py                 # Data scraping utilities
│   └── (other utility scripts)
│
├── analysis/                           # Analysis notebooks/scripts
│   ├── analysis-elv.py                 # ELV analysis
│   ├── elv-flows.py                    # ELV flow analysis
│   ├── plots.py                        # Plotting utilities
│   ├── remade-system-calcs.py          # System calculations
│   └── (other analysis files)
│
├── test/                               # Unit tests
│   ├── __init__.py
│   ├── test_pandera_*.py               # Data validation tests
│   └── test.py                         # General tests
│
├── utils/                              # Utilities
│   ├── __init__.py
│   ├── logging_config.py               # Logging configuration
│   └── pandera_checks.py               # Data schema validation
│
├── data/                               # Input data (not in version control)
│   ├── GIS/                            # Geographic data
│   ├── SWIS/                           # Waste facility data
│   ├── RDRS/                           # RDRS flows
│   ├── EMFAC/                          # Emissions factors
│   ├── PIERS/                          # Port flows
│   ├── CT/                             # Connecticut data
│   ├── elv/                            # Vehicle data
│   ├── e-waste/                        # E-waste data
│   ├── appliance/                      # Appliance data
│   ├── BLS/                            # Labor statistics
│   └── processed-inputs/               # Derived inputs
│
├── model_cache/                        # Computed caches
│   └── (pickled data, not in version control)
│
├── output/                             # Model outputs
│   ├── pipeline/                       # Pipeline results
│   ├── figures/                        # Generated plots
│   ├── appliance/                      # Appliance-specific outputs
│   ├── elv/                            # ELV-specific outputs
│   └── (other outputs)
│
└── venv/                               # Virtual environment
    └── (do not commit)
```

### Key Modules

**config/** - Configuration management using dynaconf
- Loads settings from TOML files
- Merges environment-specific overrides
- Type-validates configuration

**core/** - Core data types and utilities
- `model_types.py`: Pydantic schemas for data validation
- `units.py`: Pint-based unit handling
- `common.py`: Shared functions (geocoding, caching, etc.)

**layers/** - Material flow layers
- Each layer defines endpoints, flows, and material mappings
- `base.py`: Common layer interface
- Subclasses: RDRS, Appliance, ELV, E-waste, Connecticut

**services/** - External service integrations
- `geocode.py`: Nominatim (free) and HERE (API) geocoding
- `routing.py`: OSRM and searoute for routing
- `emissions.py`: EMFAC emissions calculations

---

## Troubleshooting

### Common Issues and Solutions

#### 1. ImportError: No module named 'gdal'

**Problem:** GDAL installation failed during `pip install`

**Solution (macOS):**

```bash
# Install GDAL system library via Homebrew
brew install gdal

# Reinstall Python package
pip install --force-reinstall GDAL
```

**Solution (Ubuntu/Linux):**

```bash
sudo apt-get install gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install --force-reinstall GDAL
```

#### 2. Cartopy Build Failure

**Problem:** Cartopy fails to compile during installation

**Solution:**

Cartopy is installed from git in `requirements.txt`. This is already pre-configured, but if it fails:

```bash
# Try installing pre-built wheel
pip install Cartopy --only-binary :all:
```

If that doesn't work, install from conda (which has pre-built binaries):

```bash
# Create new environment via conda
conda create -n pipeline python=3.11
conda activate pipeline
conda install -c conda-forge cartopy
```

Then install remaining packages:

```bash
pip install -r requirements.txt --no-build-isolation
```

#### 3. ModuleNotFoundError: No module named 'pint'

**Problem:** Unit handling module not found

**Solution:**

```bash
pip install Pint Pint-Pandas
```

#### 4. HERE API Key Issues

**Problem:** Geocoding fails with "Invalid API key"

**Solution:**

1. Verify `.secrets.toml` exists with correct API key
2. Set as environment variable instead:

```bash
export HERE_API_KEY="your_actual_key_here"
```

3. Test connection:

```python
from services.geocode import HERE
# This will fail early if key is invalid
```

#### 5. OSRM Connection Error

**Problem:** Routing fails with "Cannot connect to OSRM" or "No settings for
routing.osrm.*"

**Solution:**

OSRM routing is configured by region in `settings.toml` under `[routing.osrm.*]`
sections. The model automatically selects the appropriate OSRM server based on
the geographic location of the flows.

1. **Verify OSRM servers are configured in settings.toml:**
   ```toml
   [routing.osrm.ca]      # California
   request_url = "localhost:5000/v1/driving"
   base_url = "http://localhost:5000"
   
   [routing.osrm.ne]      # New England
   request_url = "localhost:5001/v1/driving"
   base_url = "http://localhost:5001"
   
   [routing.osrm.global]  # Fallback
   request_url = "https://routing.openstreetmap.de/routed-car/v1/driving"
   base_url = "https://routing.openstreetmap.de/routed-car"
   ```

2. **Start local OSRM servers:**
   ```bash
   # Terminal 1: Start California server
   osrm-routed --port 5000 california-latest.osrm
   
   # Terminal 2: Start New England server (if needed)
   osrm-routed --port 5001 new-england-latest.osrm
   ```

3. **Test connectivity:**
   ```bash
   curl "http://localhost:5000/route/v1/driving/-122.4194,37.7749;-118.2437,34.0522"
   ```

4. **If servers are unavailable,** the model falls back to the global OSRM
   server (slower, but works without local setup). This is configured as the
   default fallback in `settings.toml`.

5. **To override OSRM settings per run,** create a `paths.toml` override:
   ```toml
   [routing.osrm.ca]
   request_url = "http://your-server.com:5000/v1/driving"
   base_url = "http://your-server.com:5000"
   ```

#### 6. Memory Error: "Cannot allocate memory"

**Problem:** Model runs out of RAM during processing

**Solution:**

- Increase available RAM or reduce data scope
- Split processing by layer:

```bash
python pipeline.py --layers rdrs
python pipeline.py --layers appliance
```

- Reduce geographic scope (e.g., process single county at a time)

#### 7. FileNotFoundError: Data files not found

**Problem:** "data/emfac/EMFAC2021-ER-*.csv not found"

**Solution:**

1. Verify data is downloaded:

```bash
ls -la data/emfac/
```

2. Check configuration in `settings.toml` or `paths.toml`:

```bash
grep emfac_data settings.toml
```

3. Update path if necessary:

```toml
# In paths.toml
[model_paths]
emfac_data = '/correct/path/to/emfac'
```

#### 8. SQLite Database Lock

**Problem:** "database is locked" error during caching

**Solution:**

```bash
# Kill any lingering Python processes
pkill -f python

# Clear old cache (if safe to do so)
rm -rf model_cache/*.pkl
rm -rf model_cache/*.db

# Retry the model run
python pipeline.py
```

#### 9. Jupyter Kernel Not Found

**Problem:** "No module named 'jupyter'" or kernel won't start

**Solution:**

```bash
# Ensure Jupyter is installed
pip install jupyter jupyterlab ipykernel

# Re-register kernel
ipython kernel install --user --name=pipeline --display-name="Pipeline (Python 3.11)"

# Restart Jupyter Lab
jupyter-lab
```

#### 10. Git Credential Issues

**Problem:** `pip install` from git repos fails (e.g., osrm, cartopy)

**Solution:**

Ensure git is configured and you have credentials for private repos:

```bash
# Check git credentials
git config --global user.name
git config --global user.email

# For private repos, use SSH instead of HTTPS
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

### Getting Help

If you encounter issues:

1. Check the logs in `output/pipeline/`
2. Review `SETUP.md` for original setup notes
3. Check project documentation in `README.md`
4. Review Jupyter notebooks for example usage: `pipeline-crindt.ipynb`

---

## Quick Reference

### Activate Environment

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model

```bash
python pipeline.py
```

### Open Jupyter

```bash
jupyter-lab
```

### Check Configuration

```bash
python -c "from config.settings import settings; print(settings.model.to_dict())"
```

### View Recent Outputs

```bash
ls -lhrt output/pipeline/ | tail -20
```

### Clear Caches

```bash
rm -rf model_cache/*
```

---

## Additional Resources

- **Main README:** [README.md](README.md)
- **Original Setup Guide:** [SETUP.md](SETUP.md)
- **Dynaconf Documentation:** https://www.dynaconf.com/
- **EMFAC Online:** https://arb.ca.gov/emfac/emissions-inventory
- **OSRM (Open Source Routing Machine):** https://project-osrm.org/
- **Searoute (Maritime Routing):** https://pypi.org/project/searoute/
- **Pint (Units):** https://pint.readthedocs.io/
- **GeoPandas:** https://geopandas.org/

---

**Last Updated:** 2026-06-06

**Tested Environment:** Python 3.11.7, macOS 14 (Sonoma), M1 Pro

For updates or corrections, please refer to the project repository or contact
the development team.
