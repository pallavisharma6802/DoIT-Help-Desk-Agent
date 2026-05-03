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
    "O365":     "#4E9AF1",
    "NetID":    "#2ECC71",
    "Duo_MFA":  "#F39C12",
    "VPN":      "#A569BD",
    "WiFi":     "#1ABC9C",
    "Printing": "#E74C3C",
}

CATEGORY_ICONS = {
    "O365":     "✉",
    "NetID":    "🔑",
    "Duo_MFA":  "🔐",
    "VPN":      "🌐",
    "WiFi":     "📶",
    "Printing": "🖨",
}

G = build_graph()

net = Network(
    height="100vh",
    width="100%",
    bgcolor="#0f0f1a",
    font_color="#e0e0e0",
    cdn_resources="in_line",
)
net.barnes_hut(
    gravity=-4000,
    central_gravity=0.05,
    spring_length=180,
    spring_strength=0.02,
    damping=0.9,
)

# Node degree for sizing
degrees = dict(G.degree())
max_deg = max(degrees.values()) if degrees else 1

for node_id, data in G.nodes(data=True):
    cat = data.get("category", "")
    color = CATEGORY_COLORS.get(cat, "#888888")
    title = data.get("title", node_id)
    url = data.get("url", "")
    deg = degrees.get(node_id, 0)

    # Scale node size by degree: range 14-36
    size = 14 + int((deg / max_deg) * 22)

    # Short label: first 3 words of title
    short_label = " ".join(title.split()[:3])

    hover = (
        f"<div style='font-family:sans-serif;max-width:280px'>"
        f"<b style='font-size:13px'>{title}</b><br>"
        f"<span style='color:#aaa;font-size:11px'>Category: {cat} &nbsp;|&nbsp; Connections: {deg}</span><br>"
        f"<a href='{url}' target='_blank' style='color:#4E9AF1;font-size:11px'>{url}</a>"
        f"</div>"
    )

    net.add_node(
        node_id,
        label=short_label,
        title=hover,
        color={
            "background": color,
            "border": "#ffffff22",
            "highlight": {"background": "#ffffff", "border": color},
            "hover": {"background": "#ffffff", "border": color},
        },
        size=size,
        font={"size": 10, "color": "#ffffff", "strokeWidth": 2, "strokeColor": "#00000088"},
        borderWidth=1,
        borderWidthSelected=3,
        shadow={"enabled": True, "color": color + "88", "size": 10},
    )

for u, v, data in G.edges(data=True):
    sim = data.get("weight", 0.65)
    same_cat = G.nodes[u].get("category") == G.nodes[v].get("category")

    if same_cat:
        cat = G.nodes[u].get("category", "")
        base_color = CATEGORY_COLORS.get(cat, "#888888")
        width = 1.5 + (sim - 0.65) * 6
        color = base_color + "99"   # category color at ~60% opacity
    else:
        width = 3.0
        color = "#FFD700"

    net.add_edge(
        u, v,
        width=round(width, 1),
        color=color,
        title=f"similarity: {sim:.3f}",
        smooth={"type": "curvedCW", "roundness": 0.15},
    )

# Generate HTML and inject UI chrome
html = net.generate_html()

# Category legend chips
legend_chips = "".join(
    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">'
    f'<div style="width:12px;height:12px;border-radius:50%;background:{c};flex-shrink:0"></div>'
    f'<span style="font-size:12px;color:#ddd">{cat}</span>'
    f'</div>'
    for cat, c in CATEGORY_COLORS.items()
)

# Stats
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
cross_edges = sum(
    1 for u, v in G.edges()
    if G.nodes[u].get("category") != G.nodes[v].get("category")
)

ui_block = f"""
<style>
  body {{ margin: 0; overflow: hidden; }}
  #graph-ui {{
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
  }}
  .panel {{
    background: rgba(10,10,25,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
    backdrop-filter: blur(8px);
    pointer-events: auto;
  }}
  .panel-title {{
    font-family: sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }}
  .stat-row {{
    display: flex;
    gap: 18px;
  }}
  .stat {{
    font-family: sans-serif;
    text-align: center;
  }}
  .stat-value {{
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    line-height: 1;
  }}
  .stat-label {{
    font-size: 10px;
    color: #777;
    margin-top: 2px;
  }}
  .hint {{
    font-family: sans-serif;
    font-size: 11px;
    color: #666;
    margin-top: 8px;
    line-height: 1.5;
  }}
</style>

<div id="graph-ui">
  <div class="panel">
    <div class="panel-title">DoIT KB Knowledge Graph</div>
    <div class="stat-row">
      <div class="stat">
        <div class="stat-value">{n_nodes}</div>
        <div class="stat-label">Articles</div>
      </div>
      <div class="stat">
        <div class="stat-value">{n_edges}</div>
        <div class="stat-label">Edges</div>
      </div>
      <div class="stat">
        <div class="stat-value" style="color:#FFD700">{cross_edges}</div>
        <div class="stat-label">Cross-category</div>
      </div>
    </div>
    <div class="hint">Hover to inspect &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Drag to pan</div>
  </div>

  <div class="panel">
    <div class="panel-title">Categories</div>
    {legend_chips}
    <div class="hint" style="margin-top:10px">
      <span style="display:inline-block;width:20px;height:2px;background:rgba(255,255,255,0.25);vertical-align:middle"></span>
      &nbsp;Same-category &nbsp;&nbsp;
      <span style="display:inline-block;width:20px;height:2px;background:#FFD700;vertical-align:middle"></span>
      &nbsp;Cross-category
    </div>
  </div>
</div>
"""

html = html.replace("</body>", ui_block + "</body>")

out_path = Path(__file__).parent / "kb_graph.html"
out_path.write_text(html)
print(f"Saved: {out_path}")
webbrowser.open(str(out_path))
