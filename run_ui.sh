#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
./style/bin/python app.py "$@"
