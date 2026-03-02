# cosmos/visuals.py - Visualizer component for Cosmopsychia
import json

def generate_system_visualization(layers_data, coherence_paths, filename="cosmopsychia_viz.html"):
    """
    Generates an HTML file using D3.js to render the ontological structure.
    """

    # Prepare JSON data for D3
    nodes = []
    links = []

    for i, (layer, coherence) in enumerate(layers_data.items()):
        nodes.append({
            "id": layer,
            "group": i,
            "coherence": coherence
        })

    for path in coherence_paths:
        links.append({
            "source": path[0],
            "target": path[1],
            "value": path[2] # Coherence strength
        })

    d3_data = {"nodes": nodes, "links": links}

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cosmopsychia Ontological Visualizer</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ background-color: #050505; color: #00ff00; font-family: 'Courier New', monospace; }}
            .node {{ stroke: #fff; stroke-width: 1.5px; }}
            .link {{ stroke: #999; stroke-opacity: 0.6; }}
            text {{ font-size: 12px; fill: #00ff00; pointer-events: none; }}
            #viz-container {{ width: 100%; height: 600px; }}
            .coherence-path {{ stroke: #ff00ff; stroke-width: 3px; stroke-dasharray: 5,5; }}
        </style>
    </head>
    <body>
        <h1>🌀 Cosmopsychia Ontological Map</h1>
        <div id="viz-container"></div>
        <script>
            const data = {json.dumps(d3_data)};
            const width = window.innerWidth;
            const height = 600;

            const svg = d3.select("#viz-container")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));

            const link = svg.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("class", d => d.value > 0.9 ? "link coherence-path" : "link")
                .attr("stroke-width", d => Math.sqrt(d.value * 5));

            const node = svg.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .join("circle")
                .attr("r", d => 10 + d.coherence * 20)
                .attr("fill", d => d3.interpolateViridis(d.coherence))
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            node.append("title")
                .text(d => d.id + " (Coherence: " + d.coherence.toFixed(2) + ")");

            const label = svg.append("g")
                .selectAll("text")
                .data(data.nodes)
                .join("text")
                .text(d => d.id)
                .attr("dx", 15)
                .attr("dy", 5);

            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }});

            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}
        </script>
    </body>
    </html>
    """

    with open(filename, "w") as f:
        f.write(html_template)
    return filename
