#!/bin/bash
# Auto-activate uv virtual environment for local-llm-sim workspace

if [[ "$PWD" == /workspaces/local-llm-sim* ]] && [[ -f /workspaces/local-llm-sim/backend/.venv/bin/activate ]]; then
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source /workspaces/local-llm-sim/backend/.venv/bin/activate
    fi
fi

