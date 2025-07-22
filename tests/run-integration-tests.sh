#!/bin/bash

# This script downloads the necessary database file and launches Datasette
# for running integration tests.

set -e
source .venv/bin/activate

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

echo "Waiting for Datasette to be ready..."
# Wait for Datasette to be ready by checking if it responds
for i in {1..30}; do
  if curl -s http://127.0.0.1:8001/ > /dev/null 2>&1; then
    echo "Datasette is ready!"
    break
  fi
  echo "Waiting for Datasette... ($i/30)"
  sleep 2
done

# Check if Datasette is actually ready
if ! curl -s http://127.0.0.1:8001/ > /dev/null 2>&1; then
  echo "ERROR: Datasette failed to start after 60 seconds"
  kill $DATASENT_PID 2>/dev/null || true
  exit 1
fi

echo "Setting LOCAL_URL environment variable..."
# add LOCAL_URL in a .env file for integration tests
APPARENT_URL="https://apparent.topology.rocks/us_physician_referral_networks.csv"
LOCAL_URL="http://127.0.0.1:8001/us_physician_referral_networks.csv"
# Reinitialize the .env with APPARENT_URL (not needed for integration testing, but desirable to keep for future usage)
echo "APPARENT_URL"=$APPARENT_URL > .env
# Add LOCAL_URL to the .env file for integration testing
echo "LOCAL_URL=$LOCAL_URL" >> .env

echo "Running integration tests..."
python -m pytest tests/ -v -m integration

echo "Stopping Datasette..."
sleep 10
kill $DATASENT_PID