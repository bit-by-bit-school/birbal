#!/bin/bash

echo "Setting up local Postgres database for Birbal..."

# Run as the default macOS Postgres superuser to create the extension
psql postgres -c "CREATE USER birbal WITH PASSWORD 'birbal';"
psql postgres -c "CREATE DATABASE birbal OWNER birbal;"
psql birbal -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql birbal -c "CREATE EXTENSION IF NOT EXISTS pg_textsearch;"

echo "Database 'birbal' created and extensions installed!"