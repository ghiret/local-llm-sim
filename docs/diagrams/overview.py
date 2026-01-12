#!/usr/bin/env python3
"""Generate simplified architecture overview diagram."""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.general import InternetGateway
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.programming.framework import FastAPI

# Graph attributes for cleaner look
graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "spline",
}

node_attr = {
    "fontsize": "12",
}

edge_attr = {
    "fontsize": "10",
}

with Diagram(
    "local-llm-sim Overview",
    filename="docs/diagrams/overview",
    outformat="png",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    # Client tools
    with Cluster("Client Tools"):
        clients = [
            Users("Open WebUI"),
            Users("aider"),
            Users("opencode"),
            Users("Agents"),
        ]

    # local-llm-sim server
    with Cluster("local-llm-sim"):
        with Cluster("API Layer"):
            ollama_api = FastAPI("Ollama API\n/api/*")
            metrics_api = FastAPI("Metrics\n/api/stats")

        with Cluster("Core"):
            registry = Server("Model\nRegistry")
            latency = Server("Latency\nSimulator")
            proxy = Server("OpenRouter\nProxy")

    # OpenRouter (cloud API)
    openrouter = InternetGateway("OpenRouter\nAPI")

    # Connections
    for client in clients:
        client >> Edge(label="Ollama Protocol") >> ollama_api

    ollama_api >> registry
    ollama_api >> latency
    ollama_api >> proxy
    ollama_api >> metrics_api

    proxy >> Edge(label="HTTPS") >> openrouter
