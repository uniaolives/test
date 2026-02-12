"""
Arkhe Visualization Module - AUV and Interactive Reporting
"""

import json
import base64

class AUV:
    @staticmethod
    def load_snapshot(snapshot_id: str):
        """Loads a digital snapshot of a location."""
        print("🌆 Carregando Vila Madalena digital...")
        if snapshot_id == "vila_madalena_20260213":
            print("   Snapshot: 14/02/2026 12:47:33 UTC")
        return AUV()

    def generate_interactive_report(self, extraction_report, doc_image_path=None, output_path="report.html"):
        """Generates an HTML report with entity-to-bbox highlighting on a real document image."""

        facts_json = json.dumps([f.model_dump() for f in extraction_report.facts])

        img_data = ""
        if doc_image_path:
            try:
                with open(doc_image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ Could not load document image: {e}")

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arkhe(N) Extraction Report</title>
            <style>
                body {{ font-family: sans-serif; display: flex; height: 100vh; margin: 0; }}
                #list {{ width: 300px; border-right: 1px solid #ccc; padding: 20px; overflow-y: auto; background: #f9f9f9; }}
                #viewer {{ flex-grow: 1; position: relative; padding: 20px; overflow: auto; background: #525659; display: flex; justify-content: center; }}
                .fact {{ padding: 12px; margin-bottom: 10px; border: 1px solid #ddd; cursor: pointer; background: white; border-radius: 4px; }}
                .fact:hover {{ background: #eef; border-color: #aaf; }}
                .bbox {{ position: absolute; border: 2px solid #ff4d4d; background: rgba(255, 77, 77, 0.2); display: none; pointer-events: none; z-index: 10; }}
                #page-container {{ position: relative; background: white; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }}
                #doc-image {{ max-width: 100%; display: block; }}
                h2 {{ margin-top: 0; }}
            </style>
        </head>
        <body>
            <div id="list">
                <h2>Facts</h2>
                <div id="facts-container"></div>
            </div>
            <div id="viewer">
                <div id="page-container">
                    <img id="doc-image" src="data:image/png;base64,{img_data}" alt="Document Page" />
                    <div id="highlight" class="bbox"></div>
                </div>
            </div>

            <script>
                const facts = {facts_json};
                const container = document.getElementById('facts-container');
                const highlight = document.getElementById('highlight');
                const img = document.getElementById('doc-image');

                facts.forEach((fact, index) => {{
                    const div = document.createElement('div');
                    div.className = 'fact';
                    div.innerHTML = `<strong>${{fact.description}}</strong><br/>${{fact.value}} ${{fact.unit}}`;
                    div.onclick = () => {{
                        const bbox = fact.provenance.bbox; // [x0, y0, x1, y1]
                        // Simple coordinate scaling could be added here if needed
                        highlight.style.left = bbox[0] + 'px';
                        highlight.style.top = bbox[1] + 'px';
                        highlight.style.width = (bbox[2] - bbox[0]) + 'px';
                        highlight.style.height = (bbox[3] - bbox[1]) + 'px';
                        highlight.style.display = 'block';
                        highlight.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
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
