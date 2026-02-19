#!/bin/bash
# Flotorch Evaluation MCP Server - EC2 Deployment Script
# Usage: ./deploy/ec2/deploy.sh [--build]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

cd "$PROJECT_ROOT"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for docker compose
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "Error: Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

# Ensure .env exists (create from .env.example if missing)
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        echo "Creating .env from .env.example..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "Please edit .env and add your FLOTORCH_API_KEY and FLOTORCH_BASE_URL."
        read -p "Continue with deployment? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Error: .env not found. Create .env with FLOTORCH_API_KEY and FLOTORCH_BASE_URL."
        exit 1
    fi
fi

# Build if requested
if [[ "$1" == "--build" ]]; then
    echo "Building Docker image..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build --no-cache
fi

echo "Starting Flotorch Evaluation MCP Server..."
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d

echo ""
echo "Deployment complete. Server should be available at:"
echo "  http://localhost:8080"
echo "  Discovery: http://localhost:8080/.well-known/flotorch-mcp"
echo ""
echo "To view logs: $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo "To stop:       $COMPOSE_CMD -f $COMPOSE_FILE down"
