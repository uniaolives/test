"""
Arkhe Visualization Module - AUV and Interactive Reporting
"""

import json

class AUV:
    @staticmethod
    def load_snapshot(snapshot_id: str):
        """Loads a digital snapshot of a location."""
        print("🌆 Carregando Vila Madalena digital...")
        if snapshot_id == "vila_madalena_20260213":
            print("   Snapshot: 14/02/2026 12:47:33 UTC")
        return AUV()

    def generate_interactive_report(self, extraction_report, output_path="report.html"):
        """Generates an HTML report with entity-to-bbox highlighting."""

        facts_json = json.dumps([f.model_dump() for f in extraction_report.facts])

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arkhe(N) Extraction Report</title>
            <style>
                body {{ font-family: sans-serif; display: flex; }}
                #list {{ width: 30%; border-right: 1px solid #ccc; padding: 20px; overflow-y: auto; }}
                #viewer {{ width: 70%; position: relative; padding: 20px; }}
                .fact {{ padding: 10px; margin: 5px; border: 1px solid #eee; cursor: pointer; }}
                .fact:hover {{ background: #f0f0f0; }}
                .bbox {{ position: absolute; border: 2px solid red; background: rgba(255,0,0,0.1); display: none; pointer-events: none; }}
                #page {{ width: 100%; border: 1px solid #000; height: 800px; background: #fff; position: relative; }}
            </style>
        </head>
        <body>
            <div id="list">
                <h2>Extracted Facts</h2>
                <div id="facts-container"></div>
            </div>
            <div id="viewer">
                <h2>Document Viewer (Page 1)</h2>
                <div id="page">
                    <div id="highlight" class="bbox"></div>
                </div>
            </div>

            <script>
                const facts = {facts_json};
                const container = document.getElementById('facts-container');
                const highlight = document.getElementById('highlight');

                facts.forEach((fact, index) => {{
                    const div = document.createElement('div');
                    div.className = 'fact';
                    div.innerHTML = `<strong>${{fact.description}}</strong>: ${{fact.value}} ${{fact.unit}}`;
                    div.onclick = () => {{
                        const bbox = fact.provenance.bbox; // [x0, y0, x1, y1]
                        highlight.style.left = bbox[0] + 'px';
                        highlight.style.top = bbox[1] + 'px';
                        highlight.style.width = (bbox[2] - bbox[0]) + 'px';
                        highlight.style.height = (bbox[3] - bbox[1]) + 'px';
                        highlight.style.display = 'block';
                    }};
                    container.appendChild(div);
                }});
            </script>
        </body>
        </html>
        """
        with open(output_path, "w") as f:
            f.write(html_template)
        print(f"📊 Relatório interativo gerado em {output_path}")
