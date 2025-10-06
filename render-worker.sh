#!/usr/bin/env bash
set -e
echo "=== Starting Coinryze Worker ==="
cd "$(dirname "$0")"

# Ensure frontend folder & CSV exist
mkdir -p frontend
if [ ! -f "frontend/coinryze_history.csv" ]; then
  echo "issue_id,timestamp,number,color,size,odd_even" > frontend/coinryze_history.csv
fi

# Run fetcher with unbuffered output so logs appear in Render
python -u fetcher/fetch_coinryze.py
