#!/usr/bin/env bash
set -euo pipefail

uv run --python 3.11 pm1h-emergency-stop "$@"
