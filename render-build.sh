#!/bin/bash
set -e

# Install Python backend
pip install ".[web]"

# Build React frontend
cd frontend
npm install
npm run build
cd ..

echo "Build complete. Frontend dist:"
ls frontend/dist/
