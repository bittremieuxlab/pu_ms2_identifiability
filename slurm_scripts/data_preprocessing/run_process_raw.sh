#!/bin/bash

#SBATCH --job-name=msv_process_py
#SBATCH --output=logs/process_py_%j.log
#SBATCH --error=logs/process_py_%j.err
#SBATCH --partition=cpucourt

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --account=YOUR_ACCOUNT

#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL

# -----------------------
# Load environment
# -----------------------
module purge
module load mono

# -----------------------
# Paths to external tools (download separately - see README)
# -----------------------
# ThermoRawFileParser: https://github.com/compomics/ThermoRawFileParser
#   - Download latest release or compile from source
#   - Place at: /path/to/your/working/directory/tools/ThermoRawFileParser/ThermoRawFileParser.exe
#
# ScanHeadsman: https://bitbucket.org/caetera/scanheadsman/src/master/
#   - See link for download instructions
#   - Place at: /path/to/your/working/directory/tools/ScanHeadsman/ScanHeadsman.exe

SCANHEADSMAN="${SCANHEADSMAN:-/path/to/your/working/directory/tools/ScanHeadsman/ScanHeadsman.exe}"
THERMORAWFILEPARSER="${THERMORAWFILEPARSER:-/path/to/your/working/directory/tools/ThermoRawFileParser/ThermoRawFileParser.exe}"

# -----------------------
# Argument: parent folder of all MSV directories
#   - optional 1st CLI arg; else env MSV_PARENT; else default
# -----------------------
MSV_PARENT_DEFAULT="/path/to/your/working/directory/new_data"
SCRIPT_DIR="/path/to/your/working/directory/scripts/data_preprocessing"
MSV_PARENT="${1:-${MSV_PARENT:-$MSV_PARENT_DEFAULT}}"

if [ -z "$MSV_PARENT" ] || [ ! -d "$MSV_PARENT" ]; then
    echo "ERROR: MSV_PARENT is not a valid directory: '$MSV_PARENT'" >&2
    echo "Usage: sbatch process_raw_python.sh /path/to/msv_parent (or export MSV_PARENT)" >&2
    exit 1
fi




echo "Processing MSV folders inside: $MSV_PARENT"
echo "ScanHeadsman: $SCANHEADSMAN"
echo "ThermoRawFileParser: $THERMORAWFILEPARSER"

srun python /path/to/your/working/directory/scripts/data_preprocessing/process_raw.py \
  --parent "$MSV_PARENT" \
  --scanheadsman "$SCANHEADSMAN" \
  --thermo "$THERMORAWFILEPARSER"

exit $?


