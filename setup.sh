#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
ARCHIVE_DIR="${DATA_ROOT}/archives"
MESSIDOR_ROOT="${DATA_ROOT}/messidor"
BASELINE_DIR="${PROJECT_ROOT}/baseline"
ARCHIVE_PATH="${ARCHIVE_DIR}/messidor.zip"
MESSIDOR_URL="https://www.kaggle.com/api/v1/datasets/download/hanhan2010/messidor"
ENV_NAME="26ml"
TEMP_DIR=""

cleanup() {
    if [[ -n "${TEMP_DIR}" && -d "${TEMP_DIR}" ]]; then
        rm -rf "${TEMP_DIR}"
    fi
}
trap cleanup EXIT

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing dependency: $1" >&2
        return 1
    fi
}

for command in unzip aria2c conda; do
    check_command "${command}"
done

mkdir -p "${ARCHIVE_DIR}" "${MESSIDOR_ROOT}" "${BASELINE_DIR}"

if [[ -f "${MESSIDOR_ROOT}/train.csv" && -f "${MESSIDOR_ROOT}/test.csv" \
      && -d "${MESSIDOR_ROOT}/train" && -d "${MESSIDOR_ROOT}/test" ]]; then
    echo "Messidor dataset already exists at ${MESSIDOR_ROOT}; skipping download."
else
    TEMP_DIR="$(mktemp -d)"
    aria2c -c -d "${ARCHIVE_DIR}" -o "$(basename "${ARCHIVE_PATH}")" "${MESSIDOR_URL}"
    unzip -q -o "${ARCHIVE_PATH}" -d "${TEMP_DIR}"

    if [[ ! -d "${TEMP_DIR}/Messidor" ]]; then
        echo "Unexpected archive layout: ${TEMP_DIR}/Messidor not found." >&2
        exit 1
    fi

    rm -rf "${MESSIDOR_ROOT}"
    mkdir -p "${MESSIDOR_ROOT}"
    cp -a "${TEMP_DIR}/Messidor/." "${MESSIDOR_ROOT}/"
    rm -f "${ARCHIVE_PATH}"
fi

if [[ -f "${MESSIDOR_ROOT}/mpiotte-standard.model" ]]; then
    mv -f "${MESSIDOR_ROOT}/mpiotte-standard.model" "${BASELINE_DIR}/mpiotte-standard.model"
fi

if conda env list | grep -Eq "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment ${ENV_NAME} already exists; skipping creation."
else
    conda env create -f "${PROJECT_ROOT}/environment.yml"
fi

echo "Setup complete."
echo "Dataset: ${MESSIDOR_ROOT}"
echo "Baseline model: ${BASELINE_DIR}/mpiotte-standard.model"
echo "Activate with: conda activate ${ENV_NAME}"