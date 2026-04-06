#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for NaCl nlayers experiments.
#
# Interface location:
#   - Python batch runner: deepmd_input/run_nlayers_cases.py
#   - Python low-level interface: deepmd_input/nlayers_interface.py
#
# Usage:
#   ./run_nlayers_cases.sh quick-prepare
#   ./run_nlayers_cases.sh quick-train
#   ./run_nlayers_cases.sh compare-prepare
#   ./run_nlayers_cases.sh compare-train
#   ./run_nlayers_cases.sh sog-scan-prepare
#   ./run_nlayers_cases.sh sog-scan-train
#   ./run_nlayers_cases.sh les-136-prepare
#   ./run_nlayers_cases.sh les-136-train
#   ./run_nlayers_cases.sh nlayers --model les --nlayers-list 1 3 6 --run --numb-steps 500000
#   ./run_nlayers_cases.sh custom --case compare --models sog les --run --numb-steps 2000
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python
#   NUMB_STEPS=2000
#   MODELS="dpa les sog"
#   DP_CMD="dp --pt train"
#   CONTINUE_ON_ERROR=1

usage() {
  cat <<'EOF'
run_nlayers_cases.sh - one-command launcher for nlayers experiments

Subcommands:
  quick-prepare      Generate JSONs for quick case (dpa=1, les=3, sog=6)
  quick-train        Generate + train quick case
  compare-prepare    Generate JSONs for compare case (dpa/les/sog each 1,3,6)
  compare-train      Generate + train compare case
  sog-scan-prepare   Generate JSONs for sog_scan case (sog 1..6)
  sog-scan-train     Generate + train sog_scan case
  les-136-prepare    Generate LES JSONs for nlayers=1,3,6
  les-136-train      Generate + train LES nlayers=1,3,6
  nlayers ...        Pass args directly to nlayers_interface.py
  custom ...         Pass custom args directly to run_nlayers_cases.py

Examples:
  ./run_nlayers_cases.sh compare-train
  NUMB_STEPS=1000 ./run_nlayers_cases.sh compare-train
  MODELS="sog les" ./run_nlayers_cases.sh compare-prepare
  ./run_nlayers_cases.sh les-136-train --numb-steps 500000
  ./run_nlayers_cases.sh nlayers --model les --nlayers-list 1 3 6 --run --numb-steps 500000
  ./run_nlayers_cases.sh custom --case compare --models sog les --run --numb-steps 2000
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${SCRIPT_DIR}/run_nlayers_cases.py"
INTERFACE="${SCRIPT_DIR}/nlayers_interface.py"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Error: runner not found: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${INTERFACE}" ]]; then
  echo "Error: interface not found: ${INTERFACE}" >&2
  exit 1
fi

ACTION="${1:-compare-prepare}"
if [[ $# -gt 0 ]]; then
  shift
fi

CMD=("${PYTHON_BIN}" "${RUNNER}")

case "${ACTION}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  quick-prepare)
    CMD+=(--case quick)
    ;;
  quick-train)
    CMD+=(--case quick --run)
    ;;
  compare-prepare)
    CMD+=(--case compare)
    ;;
  compare-train)
    CMD+=(--case compare --run)
    ;;
  sog-scan-prepare)
    CMD+=(--case sog_scan)
    ;;
  sog-scan-train)
    CMD+=(--case sog_scan --run)
    ;;
  les-136-prepare)
    CMD=("${PYTHON_BIN}" "${INTERFACE}" --model les --nlayers-list 1 3 6)
    ;;
  les-136-train)
    CMD=("${PYTHON_BIN}" "${INTERFACE}" --model les --nlayers-list 1 3 6 --run)
    ;;
  nlayers)
    CMD=("${PYTHON_BIN}" "${INTERFACE}" "$@")
    ;;
  custom)
    CMD=("${PYTHON_BIN}" "${RUNNER}" "$@")
    ;;
  *)
    echo "Unknown subcommand: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac

# Optional environment-driven overrides.
if [[ -n "${NUMB_STEPS:-}" ]]; then
  CMD+=(--numb-steps "${NUMB_STEPS}")
fi

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_ARR=(${MODELS})
  if [[ ${#MODEL_ARR[@]} -gt 0 ]]; then
    CMD+=(--models "${MODEL_ARR[@]}")
  fi
fi

if [[ -n "${DP_CMD:-}" ]]; then
  CMD+=(--dp-cmd "${DP_CMD}")
fi

if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  CMD+=(--continue-on-error)
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
