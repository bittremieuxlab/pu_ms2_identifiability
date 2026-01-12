# External Tools

This directory should contain external tools required for data preprocessing. These tools are **not** included in this repository and must be downloaded/installed separately.

## Required Tools

### 1. ThermoRawFileParser

**Purpose**: Convert Thermo .raw files to .mzML format

**Installation**:

```bash
# Download latest release
wget https://github.com/compomics/ThermoRawFileParser/releases/download/v1.4.4/ThermoRawFileParser1.4.4.zip

# Extract
unzip ThermoRawFileParser1.4.4.zip -d tools/ThermoRawFileParser/

# Verify installation
mono tools/ThermoRawFileParser/ThermoRawFileParser.exe --help
```

**Links**:
- GitHub: https://github.com/compomics/ThermoRawFileParser
- Latest Release: https://github.com/compomics/ThermoRawFileParser/releases/latest

**Requirements**:
- Mono runtime (Linux/macOS) or .NET Framework (Windows)
- For HPC clusters: `module load mono`

**Expected location**:
```
tools/ThermoRawFileParser/
└── ThermoRawFileParser.exe
```

---

### 2. ScanHeadsman

**Purpose**: Extract instrument settings from Thermo .raw files

**Link**: https://bitbucket.org/caetera/scanheadsman/src/master/

**Expected location**:
```
tools/ScanHeadsman/
└── ScanHeadsman.exe
```

