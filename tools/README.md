# External Tools

This directory should contain external tools required for data preprocessing. These tools are **not** included in this repository and must be downloaded/installed separately.

## Required Tools

### 1. ThermoRawFileParser

**Purpose**: Convert Thermo .raw files to .mzML format


**Links**:
- GitHub: https://github.com/compomics/ThermoRawFileParser

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

