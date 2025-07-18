# run.sh
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Keyword Clustering Tool${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# App Configuration
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_SAMPLE_SIZE=20
DEFAULT_MIN_FREQUENCY=2
DEFAULT_SEMANTIC_WEIGHT=0.7
EOF
    echo -e "${RED}❌ Please edit .env file with your API keys before running again${NC}"
    exit 1
fi

# Create data and output directories
mkdir -p data output

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Stop existing container if running
echo -e "${YELLOW}🔄 Stopping existing containers...${NC}"
docker-compose down

# Build and start
echo -e "${YELLOW}🏗️  Building and starting containers...${NC}"
docker-compose up --build