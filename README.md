# REMADE Material Flow & Emissions Modeling Pipeline

A comprehensive Python-based system for modeling material flows through waste and recycling networks, computing routes across transportation networks, and estimating environmental emissions from material transport and processing.

## Overview

The REMADE (Resource and Environmental Management for Advancement and Development Economics) pipeline models the flow of materials (waste, end-of-life vehicles, appliances, e-waste, etc.) through California's and regional waste management networks. The system:

1. **Loads material flows** from multiple data sources (RDRS, DMV, appliance surveys, e-waste databases)
2. **Geocodes origins and destinations** using Nominatim, HERE Maps, and GIS data
3. **Computes routes** using OSRM (on-road) and searoute (maritime routing)
4. **Maps routes to subareas** for emissions factor lookup
5. **Calculates emissions** using EMFAC-derived rates by vehicle class and subarea
6. **Aggregates results** by origin-destination pair, material type, transport mode, and pollutant

### Key Use Cases

- **Environmental Justice:** Identify which communities bear disproportionate pollution burden from waste transport
- **Infrastructure Planning:** Optimize routing and facility placement to minimize emissions
- **Policy Analysis:** Evaluate impact of regulations on material flows and emissions
- **Supply Chain Resilience:** Analyze geographic concentration of waste processing facilities
- **Material Recovery:** Track end-of-life material streams and recovery rates

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (recommended for OSRM, optional)
- 50+ GB disk space
- 8+ GB RAM (16+ recommended)

### Installation

```bash
# Clone and navigate to pipeline
cd /path/to/pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure paths (if data is in non-default locations)
cat > paths.toml << 'EOF'
[model_paths]
# Add custom paths here if needed
EOF
```

### Run the Pipeline

```bash
# Activate environment
source venv/bin/activate

# Start OSRM servers (optional, for local routing)
osrm-routed --port 5000 california-latest.osrm &
osrm-routed --port 5001 new-england-latest.osrm &

# Run full pipeline
python pipeline.py

# Or run specific layers only
python pipeline.py --layers rdrs,appliance

# View progress
tail -f output/pipeline/*.log
```

### Explore Results

```bash
# Results are saved as pickle files
python -c "
import pandas as pd
emiss = pd.read_pickle('output/pipeline/emiss_RDRS_Appliance_ELV.pkl')
print(emiss.columns)
print(emiss.head())
"

# Or open analysis notebooks
jupyter-lab analysis/analysis-elv.py
```

For detailed setup instructions, see [SETUP-new.md](SETUP-new.md).

## System Architecture

### Pipeline Execution Flow

```
Data Input → Endpoints → Flows → Routes → Emissions → Output
   ↓            ↓          ↓        ↓         ↓         ↓
Layers    Geocoding   Aggregation OSRM/   EMFAC    Pickle
(RDRS,    (Nominatim  & Grouping  Search  Lookup   Files
Appliance, HERE,GIS)   by Material Route   & Calc   & Stats
ELV, etc)
```

### Core Components

#### 1. **Material Layers** (`layers/`)

Each layer represents a waste/material stream with its own data sources, endpoints, and flows:

| Layer | Description | Data Source | Material Types |
|---------------|-------------------------------------|--------------------------------|-------------------------------|
| **RDRS**      | Hazardous/non-hazardous waste flows | CalRecycle RDRS                | 200+ waste categories         |
| **Appliance** | Refrigerated appliance recycling    | Survey data + facility lists   | Refrigerators, AC units, etc. |
| **ELV**       | End-of-life vehicle dismantling     | DMV data + dismantler database | Metal, glass, plastic, hazmat |
| **E-waste**   | Electronic waste flows              | UWED database                  | Electronics, TVs, etc.        |
| **Connecticut (CT)** | State-specific waste flows   | CT DEEP database               | Various waste types           |

Each layer implements:
- **Endpoints:** Origins (generation points) and destinations (processors/recyclers/landfills)
- **Flows:** Quantity of material moving from origin to destination
- **Geocoding:** Latitude/longitude for routing
- **Material Mapping:** Classification of materials for emissions calculation

#### 2. **Geocoding Services** (`services/geocode.py`)

Locates origins and destinations using multiple strategies:
- **Database lookups:** SWIS, port directories, facility databases
- **Address-based:** Nominatim (free, OpenStreetMap) and HERE Maps (API)
- **Region-based:** County, state, and country centroids
- **ZIP code areas:** ZCTA boundaries for aggregated sources

All geocodes are cached to avoid redundant API calls.

#### 3. **Routing Services** (`services/routing.py`)

Computes travel distance and duration:

| Mode         | Engine   | Data           | Use Case                                |
|--------------|----------|----------------|-----------------------------------------|
| **On-road**  | OSRM     | OpenStreetMap  | Truck transport to landfills, recyclers |
| **Maritime** | Searoute | Shipping lanes | Port-to-port waste transport            |

**OSRM Configuration:**
- Multiple regional servers (California, New England, global fallback)
- Configured via `settings.toml` under `[routing.osrm.*]`
- Local servers recommended for large datasets (faster, no API costs)
- Docker setup available for easy deployment

#### 4. **Emissions Calculation** (`services/emissions.py`)

Estimates air emissions using EMFAC rates:

**Emissions Tracked:**
- **Criteria pollutants:** PM2.5, PM10, NOx, CO, CO2
- **GHG:** CO2, CH4, N2O
- **Aggregated:** GHG intensity, CO2 equivalents

**Calculation Steps:**
1. Vehicle classification by material type and transport mode (T7 Class 8 trucks, etc.)
2. Fuel consumption rates from EMFAC
3. Route segment emission rates by CoAbDis subarea
4. Aggregation to origin-destination flows

#### 5. **Configuration Management** (`config/`)

Flexible configuration system using dynaconf:
- **`settings.toml`:** Default settings (versioned)
- **`paths.toml`:** Local path overrides (not versioned)
- **`.secrets.toml`:** API keys (not versioned)
- **Environment variables:** Runtime overrides

Example override:
```toml
# paths.toml
[model_paths]
emfac_data = "/custom/path/to/emfac"

[routing.osrm.ca]
request_url = "http://my-server:5000/v1/driving"
```

### Data Flow Architecture

```
Material Layer
  ├── Endpoints (origins & destinations)
  │    └── Geocoding
  │         ├── SWIS database
  │         ├── Nominatim (free)
  │         ├── HERE Maps (API)
  │         └── GIS regions
  │
  ├── Flows (material quantities)
  │    └── Material mapping
  │         └── Aggregation by type
  │
  └── Routes
       ├── OSRM (on-road)
       ├── Searoute (maritime)
       └── Subarea mapping
            └── Emissions calculation
                 ├── EMFAC rates
                 ├── Vehicle class
                 └── Trip aggregation
                      └── Output (pickle files)
```

## Project Structure

```
pipeline/
├── README-new.md                       # This file
├── SETUP-new.md                        # Installation & setup guide
├── pipeline.py                         # Main pipeline script
├── requirements.txt                    # Python dependencies
├── settings.toml                       # Default configuration
├── paths.toml                          # Local path overrides (create locally)
│
├── config/                             # Configuration management
│   ├── settings.py                     # Settings loader
│   ├── base_config.py                  # Configuration base classes
│   ├── cli_field.py                    # CLI option definitions
│   └── path_template.py                # Path templating
│
├── core/                               # Core utilities
│   ├── units.py                        # Pint unit system
│   ├── model_types.py                  # Pydantic schemas
│   ├── common.py                       # Shared utilities
│   └── geocode_lookups.py              # Geocoding caches
│
├── layers/                             # Material flow layers
│   ├── base.py                         # Base layer class & common methods
│   ├── rdrs.py                         # RDRS waste flows
│   ├── appliance.py                    # Appliance recycling
│   ├── elv.py                          # End-of-life vehicles
│   ├── ewaste.py                       # E-waste flows
│   └── ct.py                           # Connecticut flows
│
├── services/                           # External services
│   ├── geocode.py                      # Nominatim, HERE, GIS geocoding
│   ├── routing.py                      # OSRM, searoute routing
│   ├── emissions.py                    # EMFAC emissions calculations
│   └── option_utils.py                 # Utility functions
│
├── analysis/                           # Analysis scripts & notebooks
│   ├── analysis-elv.py                 # ELV-specific analysis
│   ├── analysis-appliance.py           # Appliance-specific analysis
│   ├── plots.py                        # Visualization utilities
│   └── remade-system-calcs.py          # System-wide calculations
│
├── test/                               # Unit tests
│   ├── test_pandera_*.py               # Data validation tests
│   └── test.py                         # General tests
│
├── utils/                              # Utilities
│   ├── logging_config.py               # Logging setup
│   └── pandera_checks.py               # Data schema validation
│
├── data/                               # Input data (not versioned)
│   ├── GIS/                            # Geographic boundaries
│   ├── SWIS/                           # Waste facility database
│   ├── RDRS/                           # RDRS flow data
│   ├── EMFAC/                          # Emissions factors
│   ├── PIERS/                          # Port flows
│   ├── elv/                            # Vehicle data
│   ├── appliance/                      # Appliance data
│   └── processed-inputs/               # Pre-processed inputs
│
├── model_cache/                        # Geocoding & routing caches
├── output/                             # Pipeline outputs
│   ├── pipeline/                       # Main results (pickle files)
│   ├── figures/                        # Generated plots
│   └── [layer-specific]/               # Layer outputs
│
└── venv/                               # Virtual environment (not versioned)
```

## Running the Pipeline

### Command-Line Options

```bash
python pipeline.py --help
```

Common options:
- `--layers RDRS,Appliance`: Run only specific layers (comma-separated)
- `--year 2021`: Model year (default: 2021)
- `--case best`: Appliance case (best, mid, low)
- `--refr-case min`: Refrigerant case (min, mid, max)
- `--verbose`: Enable debug logging

### Example Commands

```bash
# Run all layers
python pipeline.py

# Run only RDRS and ELV
python pipeline.py --layers RDRS,ELV

# Run appliance with specific case
python pipeline.py --layers Appliance --case best --refr-case min

# Enable verbose logging
python pipeline.py --verbose
```

### Monitoring Progress

The pipeline logs progress to console and optionally to file:

```bash
# Watch logs in real-time
tail -f output/pipeline/*.log

# Check cache operations
ls -lh model_cache/

# View outputs
ls -lh output/pipeline/*.pkl
```

## Using Analysis Notebooks

### Open Jupyter Lab

```bash
jupyter-lab
```

### Available Notebooks

1. **`analysis/analysis-elv.py`** - End-of-Life Vehicle analysis
   - Vehicle retirement and material composition
   - Dismantler network analysis
   - Transportation flows and routing
   - Emissions by trip type

2. **`analysis/plots.py`** - Visualization utilities
   - Route maps with basemaps
   - Flow sankey diagrams
   - Concentration curves (Lorenz curves)

3. **`analysis/remade-system-calcs.py`** - System-wide analysis
   - Cross-layer aggregations
   - Material recovery rates
   - Facility concentration analysis

## Input Data Requirements

### Key Data Files

| Category | Files | Source | Required |
|----------|-------|--------|----------|
| **GIS** | Boundaries, ZIP codes, ports | Geofabrik, US Census | ✓ |
| **EMFAC** | Emissions rates by vehicle/subarea | CARB | ✓ |
| **SWIS** | Waste facility database | CalRecycle | ✓ |
| **RDRS** | Flow data | CalRecycle | ✓ |
| **PIERS** | Port commodity flows | S&P Global | ✗ |
| **DMV** | Vehicle registration data | CA DMV | ✗ (ELV only) |
| **Appliance** | Survey data, recovery rates | Industry sources | ✗ (Appliance only) |

See [SETUP-new.md - Data Setup](SETUP-new.md#data-setup) for detailed download instructions.

## Output Data

### Pickle Files

Pipeline outputs are saved as pickled GeoDataFrames in `output/pipeline/`:

```python
import pandas as pd

# Emission results
emiss = pd.read_pickle('output/pipeline/emiss_RDRS_Appliance_ELV.pkl')

# Flow endpoints
endpoints = pd.read_pickle('output/pipeline/entities_RDRS_Appliance_ELV.pkl')

# Route geometries
routes = pd.read_pickle('output/pipeline/routes_RDRS_Appliance_ELV.pkl')
```

### Key Columns

**Emissions GeoDataFrame:**
- `o_id`, `d_id`: Origin and destination facility IDs
- `o_n1`, `d_n1`: Origin and destination names
- `trips`: Number of trips on this OD pair
- `wt_sent`: Material weight transported (tonnes)
- `emiss_ghg_u`: GHG emissions (kg CO2e)
- `emiss_nox_u`, `emiss_pm25_u`: Criteria pollutants
- `distance_km`, `vmt_u`: Distance and vehicle miles traveled
- `geometry_*`: Origin, destination, and route geometries

## Configuration Guide

### Common Settings

```toml
[model]
model_year = 2021                    # Analysis year
use_cachier = true                   # Cache expensive computations

[model.elv]
illegal_vehicles_fraction = 0.3      # % of vehicles to illegal dismantlers
mexico_export_fraction = 0.05        # % exported to Mexico

[routing.osrm.ca]                    # California OSRM server
request_url = "localhost:5000/v1/driving"
base_url = "http://localhost:5000"
```

See `settings.toml` for all available options.

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OSRM connection error | Start local OSRM servers or use global fallback |
| "No module named 'gdal'" | Install GDAL (see [SETUP-new.md](SETUP-new.md#troubleshooting)) |
| Pint unit warnings | Use `force=True` when redefining units |
| Memory exhausted | Reduce data scope or split by layer |
| Data files not found | Check `paths.toml` or `settings.toml` configuration |

See [SETUP-new.md - Troubleshooting](SETUP-new.md#troubleshooting) for detailed solutions.

## Key Classes & Functions

### Material Layers

```python
from layers.rdrs import RDRSLayer
from layers.appliance import ApplianceLayer
from layers.elv import ELVLayer

# Load a layer
layer = RDRSLayer()
endpoints = layer.get_endpoints()
flows = layer.get_flows()
```

### Geocoding

```python
from services.geocode import MergeGeocoder, NullGeocoder
from services.geocode import get_county_endpoints, get_ca_port_endpoints

# Combine multiple geocoders
geocoder = MergeGeocoder([swis_geocoder, nominatim_geocoder, here_geocoder])
gdf = geocoder.geocode(df)
```

### Routing

```python
from services.routing import safe_route_by_endpoints

# Get route between two points
p1 = shapely.geometry.Point(-122.4, 37.8)  # SF
p2 = shapely.geometry.Point(-118.2, 34.0)  # LA
route = safe_route_by_endpoints(p1, p2)
```

### Emissions

```python
from services.emissions import compute_emissions, compute_total_emissions

# Compute emissions for routes
emiss = compute_emissions(routes_gdf, vehicle_class='T7_Class8')
```

## Development

### Running Tests

```bash
pytest test/
```

### Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Use `black` for formatting

### Adding a New Layer

1. Create `layers/newlayer.py` inheriting from `layers.base.ModelLayer`
2. Implement required methods:
   - `get_endpoints()`: Return GeoDataFrame of origins and destinations
   - `get_flows()`: Return flows between endpoints
   - `filter_to_layer()`: Filter data to this layer only
3. Register in pipeline configuration

## References

- **OSRM:** https://project-osrm.org/
- **Searoute:** https://pypi.org/project/searoute/
- **EMFAC:** https://arb.ca.gov/emfac/
- **Pint Units:** https://pint.readthedocs.io/
- **GeoPandas:** https://geopandas.org/
- **Dynaconf Config:** https://www.dynaconf.com/

## Documentation

- [SETUP-new.md](SETUP-new.md) - Detailed installation and configuration
- [SETUP.md](SETUP.md) - Original setup guide
- [TODO.md](TODO.md) - Roadmap and known issues
- Jupyter notebooks in `analysis/` - Interactive examples
- Source code docstrings - Function-level documentation

## Support & Issues

For issues or questions:

1. Check the [SETUP-new.md Troubleshooting](SETUP-new.md#troubleshooting) section
2. Review existing Jupyter notebooks for examples
3. Check `settings.toml` for configuration options
4. Examine logs in `output/pipeline/`

## Citation

If you use this system in research, please cite:

```
REMADE Model System (2021-2026)
Center for Environmental Research & Technology
UC Riverside
```

## License

[Add license information here]

---

**Last Updated:** 2026-06-06

**Python:** 3.11.7+ | **Status:** Active Development
