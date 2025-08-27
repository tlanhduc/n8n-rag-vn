#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p app/services

# Set permissions
echo "Setting permissions..."
chmod +x build.sh

echo "Build completed successfully!"
