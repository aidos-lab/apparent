#!/bin/bash

# This script downloads the necessary database file and launches Datasette
# for running integration tests.

set -e

DB_URL="https://apparent.topology.rocks/us_physician_referral_networks.db"
DB_FILE="data/us_physician_referral_networks.db"

if [ ! -f "$DB_FILE" ]; then
  echo "Downloading database from $DB_URL..."
  curl -L -o "$DB_FILE" "$DB_URL"
else
  echo "Database file already exists."
fi

echo "Launching Datasette in the background..."
datasette serve "$DB_FILE" --setting sql_time_limit_ms 500000 --setting max_returned_rows 200000 --setting allow_csv_stream off --reload &
DATASENT_PID=$!
echo "Datasette started with PID: $DATASENT_PID"

echo "Setting LOCAL_URL environment variable..."
LOCAL_URL="http://127.0.0.1:8001/us_physician_referral_networks.csv"
echo "LOCAL_URL=$LOCAL_URL" > .env

echo "Running integration tests..."
python -m pytest tests/ -v -m integration

echo "Stopping Datasette..."
kill $DATASENT_PID