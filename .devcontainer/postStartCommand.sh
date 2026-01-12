#!/bin/bash
# Post-start script for local-llm-sim dev container
# - Installs latest Claude CLI
# - Auto-activates virtual environment
# - Displays helpful startup information

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Install/update Claude CLI
echo -e "${CYAN}🤖 Checking Claude CLI...${RESET}"
if ! command -v claude &> /dev/null || [[ "$FORCE_CLAUDE_UPDATE" == "true" ]]; then
    echo -e "${YELLOW}📦 Installing/updating Claude CLI...${RESET}"
    curl -fsSL https://claude.ai/install.sh | bash
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Claude CLI installed successfully${RESET}"
    else
        echo -e "${RED}✗ Failed to install Claude CLI${RESET}"
    fi
else
    echo -e "${GREEN}✓ Claude CLI already installed${RESET}"
fi

# Auto-activate virtual environment
if [[ "$PWD" == /workspaces/local-llm-sim* ]] && [[ -f /workspaces/local-llm-sim/backend/.venv/bin/activate ]]; then
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source /workspaces/local-llm-sim/backend/.venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${RESET}"
    fi
fi

# Display startup information
echo -e ""
echo -e "${BOLD}${MAGENTA}╔════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${MAGENTA}║     🚀 local-llm-sim Dev Environment      ║${RESET}"
echo -e "${BOLD}${MAGENTA}╚════════════════════════════════════════════╝${RESET}"
echo -e ""
echo -e "${BOLD}${CYAN}📋 Quick Start:${RESET}"
echo -e ""
echo -e "${YELLOW}1.${RESET} ${BOLD}Set your API key:${RESET}"
echo -e "   ${GREEN}export OPENROUTER_API_KEY=\"sk-or-...\"${RESET}"
echo -e "   ${BLUE}(or create .env file with OPENROUTER_API_KEY=sk-or-...)${RESET}"
echo -e ""
echo -e "${YELLOW}2.${RESET} ${BOLD}Start services:${RESET}"
echo -e "   ${GREEN}source .venv/bin/activate${RESET}"
echo -e "   ${GREEN}./scripts/dev.sh start-all${RESET}"
echo -e ""
echo -e "${YELLOW}3.${RESET} ${BOLD}Access:${RESET}"
echo -e "   ${CYAN}🌐 Open WebUI:${RESET}  http://localhost:3000"
echo -e "   ${CYAN}🔌 API Server:${RESET}  http://localhost:11434"
echo -e ""
echo -e "${BLUE}💡 Tip: Use 'claude' command to interact with Claude AI${RESET}"
echo -e ""

