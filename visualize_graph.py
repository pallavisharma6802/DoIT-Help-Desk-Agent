"""
Generates an interactive HTML visualization of the KB knowledge graph.
Run: python3 visualize_graph.py
Opens kb_graph.html in your default browser.
"""

import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retriever import build_graph
from pyvis.network import Network

CATEGORY_COLORS = {
    "O365":     "#4A90D9",
    "NetID":    "#7ED321",
    "Duo_MFA":  "#F5A623",
    "VPN":      "#9B59B6",
    "WiFi":     "#1ABC9C",
    "Printing": "#E74C3C",
}

G = build_graph()

net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="white", cdn_resources="in_line")
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

for node_id, data in G.nodes(data=True):
    cat = data.get("category", "")
    color = CATEGORY_COLORS.get(cat, "#888888")
    title = data.get("title", node_id)
    url = data.get("url", "")
    net.add_node(
        node_id,
        label=node_id,
        title=f"<b>{title}</b><br>Category: {cat}<br><a href='{url}' target='_blank'>{url}</a>",
        color=color,
        size=18,
    )

for u, v, data in G.edges(data=True):
    reason = data.get("reason", "")
    if reason == "same_category":
        net.add_edge(u, v, color="rgba(255,255,255,0.08)", width=1)
    else:
        net.add_edge(u, v, color="#FFD700", width=2.5, title="keyword overlap")

# Legend as a fixed HTML element
legend_html = "".join(
    f'<span style="background:{c};padding:3px 8px;border-radius:4px;margin:3px;font-size:13px">{cat}</span>'
    for cat, c in CATEGORY_COLORS.items()
)
net.html = net.html  # ensure html attribute exists after generate_html

out_path = Path(__file__).parent / "kb_graph.html"
html = net.generate_html()

# Inject legend before closing </body>
legend_block = f"""
<div style="position:fixed;bottom:20px;left:20px;background:rgba(0,0,0,0.7);
            padding:10px 14px;border-radius:8px;z-index:9999">
  <div style="color:white;font-family:sans-serif;font-size:13px;margin-bottom:6px">
    <b>Categories</b>
  </div>
  {legend_html}
</div>
"""
html = html.replace("</body>", legend_block + "</body>")
out_path.write_text(html)

print(f"Saved: {out_path}")
webbrowser.open(str(out_path))
