# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2026-01-15

### Added
- Enhanced SETUP.md documentation with troubleshooting guide
- Extended README with detailed code summaries for each module

### Fixed
- Long-standing units warning in emissions calculations
- OSRM configuration now properly managed through settings.toml

### Changed
- Moved OSRM configuration from hardcoded values to settings.toml

---

## [0.3.0] - 2025-12-10

### Added
- Support for extended vehicle dataset (make/model/class/series/fuel/year)
- Improved routing module with distance caching
- New test suite for data validation with pandera

### Changed
- Restructured emissions module for better maintainability
- Geocoding now supports both Nominatim and HERE API backends
- Updated configuration system to use dynaconf

### Fixed
- Routing distance calculations for maritime routes
- Memory usage in model cache with proper cleanup

---

## [0.2.1] - 2025-11-01

### Added
- Basic geographic boundary files (CT, northeastern states)
- Enhanced data dictionaries for emissions

### Fixed
- Coordinate projection issues in GIS layers

---

## [0.2.0] - 2025-10-15

### Added
- Initial appliance layer implementation
- End-of-life vehicle (ELV) processing module
- RDRS data processing pipeline

### Changed
- Unified data layer architecture
- Improved geocoding performance

---

## [0.1.0] - 2025-09-01

### Added
- Initial project structure
- Core pipeline framework
- Configuration system
- Basic geocoding functionality
