#!/bin/bash

# AI-Based Shipping Line Tracking System
# Run script for Unix systems (macOS/Linux)
# Usage: ./run.sh <booking_id>

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if booking ID is provided
if [ $# -eq 0 ]; then
    print_error "No booking ID provided"
    echo "Usage: $0 <booking_id>"
    echo "Example: $0 SINI25432400"
    exit 1
fi

BOOKING_ID="$1"

print_status "Starting AI-Based Shipping Line Tracking System"
print_status "Booking ID: $BOOKING_ID"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    print_status "Please install Python 3.11+ from https://python.org"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    print_warning "No virtual environment found"
    print_status "Creating virtual environment..."

    if command -v uv &> /dev/null; then
        print_status "Using uv to create virtual environment"
        uv venv --python 3.11
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        fi
    else
        print_status "Using standard venv"
        python3 -m venv venv
        source venv/bin/activate
    fi

    print_success "Virtual environment created"
else
    print_status "Activating virtual environment..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
fi

# Check if requirements are installed
print_status "Checking dependencies..."
if ! python -c "import browser_use" 2>/dev/null; then
    print_warning "Dependencies not found, installing..."

    if command -v uv &> /dev/null; then
        print_status "Installing dependencies with uv..."
        uv pip install -r requirements.txt
    else
        print_status "Installing dependencies with pip..."
        pip install -r requirements.txt
    fi

    print_status "Installing Playwright browsers..."
    playwright install

    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning "No .env file found"
    if [ -f ".env.example" ]; then
        print_status "Copying .env.example to .env"
        cp .env.example .env
        print_warning "Please edit .env file and add your API key(s) for the desired LLM provider (e.g., GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)"
        print_status "Then run this script again"
        exit 1
    else
        print_error "No .env.example file found"
        exit 1
    fi
fi

# Run the tracking system
print_status "Starting shipment tracking..."
print_status "This may take 30-60 seconds depending on website response time"

if python agent.py "$BOOKING_ID"; then
    print_success "Tracking completed successfully!"
else
    print_error "Tracking failed. Check the logs for details."
    exit 1
fi
