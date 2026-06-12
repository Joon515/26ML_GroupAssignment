#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing dependency: $1"
        return 1
    fi
}

check_command unzip
check_command aria2c
check_command conda

aria2c -d "${PROJECT_ROOT}/datasets" -o eyepacs-aptos-messidor-diabetic-retinopathy.zip "https://www.kaggle.com/api/v1/datasets/download/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"

aria2c -d "${PROJECT_ROOT}/datasets" -o messidor.zip "https://www.kaggle.com/api/v1/datasets/download/hanhan2010/messidor"

unzip "${PROJECT_ROOT}/datasets/eyepacs-aptos-messidor-diabetic-retinopathy.zip" -d "${PROJECT_ROOT}/datasets/" 

rm "${PROJECT_ROOT}/datasets/eyepacs-aptos-messidor-diabetic-retinopathy.zip"

unzip "${PROJECT_ROOT}/datasets/messidor.zip" -d "${PROJECT_ROOT}/datasets/"

rm "${PROJECT_ROOT}/datasets/messidor.zip"

mv "${PROJECT_ROOT}/datasets/mpiotte-standard.model" "${PROJECT_ROOT}/baseline/mpiotte-standard.model"

conda env create -f "${PROJECT_ROOT}/environment.yml"

conda activate 26ml

echo "Setup complete. You can now activate the conda environment with 'conda activate 26ml' and start working on the project."