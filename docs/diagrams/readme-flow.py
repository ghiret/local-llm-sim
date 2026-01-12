#!/usr/bin/env python3
"""Generate simplified flow diagram for README."""

from diagrams import Diagram, Edge, Cluster
from diagrams.aws.general import InternetGateway
from diagrams.onprem.client import Users
from diagrams.custom import Custom
from diagrams.programming.framework import FastAPI
import os

graph_attr = {
    "fontsize": "12",
    "bgcolor": "white",
    "pad": "0.3",
    "splines": "ortho",
    "nodesep": "1.0",
    "ranksep": "1.2",
}

node_attr = {
    "fontsize": "11",
    "fontname": "Sans-Serif",
}

edge_attr = {
    "fontsize": "10",
    "fontname": "Sans-Serif",
}

with Diagram(
    "",
    filename="docs/diagrams/readme-flow",
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    # Client tools
    with Cluster("Your Tools"):
        clients = Users("Open WebUI\naider\nopencode\nagents")

    # local-llm-sim - simpler cluster
    with Cluster("local-llm-sim (:11434)"):
        sim = FastAPI("Adds Latency\n~400 t/s prefill\n~20 t/s decode")

    # OpenRouter
    openrouter = InternetGateway("OpenRouter")

    # Simple flow
    clients >> Edge(label="Ollama API") >> sim >> Edge(label="proxied") >> openrouter
