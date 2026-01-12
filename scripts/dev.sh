#!/bin/bash
# Development helper script for local-llm-sim

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start         Start local-llm-sim server only"
    echo "  webui         Start Open WebUI only (requires local-llm-sim running)"
    echo "  start-all     Start both local-llm-sim and Open WebUI"
    echo "  install-webui Install Open WebUI"
    echo "  test          Run tests"
    echo "  curl-test     Run curl tests against the API"
    echo ""
}

check_api_key() {
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OPENROUTER_API_KEY not set${NC}"
        echo "Set it with: export OPENROUTER_API_KEY='sk-or-...'"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        export OPENROUTER_API_KEY="not-set"
    fi
}

ensure_venv() {
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    # Check if local-llm-sim is installed
    if ! pip show local-llm-sim &>/dev/null; then
        echo "Installing local-llm-sim..."
        pip install -e ".[dev]"
    fi
}

start_server() {
    check_api_key
    ensure_venv
    echo -e "${GREEN}Starting local-llm-sim server on http://localhost:11434${NC}"
    local-llm-sim serve
}

install_webui() {
    ensure_venv
    echo -e "${GREEN}Installing Open WebUI...${NC}"
    pip install open-webui
    echo -e "${GREEN}Done! Run '$0 webui' to start Open WebUI${NC}"
}

start_webui() {
    ensure_venv

    # Check if open-webui is installed
    if ! pip show open-webui &>/dev/null; then
        echo -e "${YELLOW}Open WebUI not installed. Installing...${NC}"
        pip install open-webui
    fi

    echo -e "${GREEN}Starting Open WebUI on http://localhost:3000${NC}"
    echo -e "${BLUE}Make sure local-llm-sim is running on port 11434${NC}"
    echo ""
    OLLAMA_BASE_URL=http://localhost:11434 open-webui serve --port 3000
}

start_all() {
    check_api_key
    ensure_venv

    # Check if open-webui is installed
    if ! pip show open-webui &>/dev/null; then
        echo -e "${YELLOW}Open WebUI not installed. Installing...${NC}"
        pip install open-webui
    fi

    echo -e "${GREEN}Starting local-llm-sim + Open WebUI...${NC}"
    echo ""

    # Start local-llm-sim in background
    echo -e "${BLUE}Starting local-llm-sim on http://localhost:11434${NC}"
    local-llm-sim serve &
    LLM_PID=$!

    # Wait for it to be ready
    sleep 2

    # Start Open WebUI
    echo -e "${BLUE}Starting Open WebUI on http://localhost:3000${NC}"
    OLLAMA_BASE_URL=http://localhost:11434 open-webui serve --port 3000 &
    WEBUI_PID=$!

    echo ""
    echo -e "${GREEN}Both services started!${NC}"
    echo "  - Open WebUI:    http://localhost:3000"
    echo "  - Ollama API:    http://localhost:11434"
    echo "  - Stats:         http://localhost:11434/api/stats"
    echo ""
    echo "Press Ctrl+C to stop both services"

    # Wait for Ctrl+C
    trap "kill $LLM_PID $WEBUI_PID 2>/dev/null; exit" INT TERM
    wait
}

run_tests() {
    ensure_venv
    pytest -v
}

curl_test() {
    BASE_URL="${1:-http://localhost:11434}"
    echo -e "${GREEN}Testing API at $BASE_URL${NC}"
    echo ""

    echo "1. Health check:"
    curl -s "$BASE_URL/" | python3 -m json.tool
    echo ""

    echo "2. List models:"
    curl -s "$BASE_URL/api/tags" | python3 -m json.tool
    echo ""

    echo "3. Stats:"
    curl -s "$BASE_URL/api/stats" | python3 -m json.tool
    echo ""

    echo "4. Show model (deepseek-v3:q4):"
    curl -s -X POST "$BASE_URL/api/show" -d '{"name":"deepseek-v3:q4"}' | python3 -m json.tool
}

case "${1:-}" in
    start)
        start_server
        ;;
    webui)
        start_webui
        ;;
    start-all)
        start_all
        ;;
    install-webui)
        install_webui
        ;;
    test)
        run_tests
        ;;
    curl-test)
        curl_test "${2:-}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
