"""
Generates an interactive HTML visualization of the KB knowledge graph.
Run: python3 visualize_graph.py
Opens kb_graph.html in your default browser.

Interactivity:
  - Click a node    → dims unrelated nodes/edges; highlights neighbors + shows info panel
  - Click legend    → filters graph to that category
  - Click canvas    → resets everything
  - Scroll/drag     → zoom / pan
"""

import sys
import json
import colorsys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from retriever import build_graph
from pyvis.network import Network

# Maximally-distinct colors — evenly spaced hues via HSL
def _build_palette(categories: list) -> dict:
    """
    Assign one color per category by distributing hues evenly around the color
    wheel with high saturation + medium-high lightness (visible on dark bg).
    No two adjacent categories share a similar hue.
    """
    n = max(len(categories), 1)
    palette = {}
    for i, cat in enumerate(sorted(categories)):
        hue = (i / n + 0.03) % 1.0          # offset so first hue ≠ pure red
        r, g, b = colorsys.hls_to_rgb(hue, 0.60, 0.82)
        palette[cat] = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
    return palette

# Build graph
G = build_graph()

all_cats = sorted(
    {data.get("category", "") for _, data in G.nodes(data=True)
     if data.get("category")}
)
PALETTE = _build_palette(all_cats)


def _cat_color(cat: str) -> str:
    return PALETTE.get(cat, "#aaaaaa")

# pyvis Network
net = Network(
    height="100vh",
    width="100%",
    bgcolor="#0d0d1a",
    font_color="#e0e0e0",
    cdn_resources="in_line",
)
net.barnes_hut(
    gravity=-4500,
    central_gravity=0.04,
    spring_length=200,
    spring_strength=0.018,
    damping=0.92,
)

degrees  = dict(G.degree())
max_deg  = max(degrees.values()) if degrees else 1

# Collect metadata for the JS info panel
node_meta = {}  # str(node_id) -> {title, category, url, color}

for node_id, data in G.nodes(data=True):
    cat   = data.get("category", "")
    color = _cat_color(cat)
    title = data.get("title", str(node_id))
    url   = data.get("url", "")
    deg   = degrees.get(node_id, 0)
    size  = 14 + int((deg / max_deg) * 22)
    short_label = " ".join(title.split()[:3])

    node_meta[str(node_id)] = {
        "title":    title,
        "category": cat,
        "url":      url,
        "color":    color,
    }

    net.add_node(
        node_id,
        label=short_label,
        title="",   # disable default tooltip — we use our own panel
        color={
            "background": color,
            "border":     "#ffffff18",
            "highlight":  {"background": "#ffffff", "border": color},
            "hover":      {"background": "#ffffffcc", "border": color},
        },
        size=size,
        font={
            "size": 10, "color": "#ffffff",
            "strokeWidth": 2, "strokeColor": "#00000099",
        },
        borderWidth=1,
        borderWidthSelected=3,
        shadow={"enabled": True, "color": color + "55", "size": 12},
    )

for u, v, data in G.edges(data=True):
    sim      = data.get("weight", 0.65)
    same_cat = G.nodes[u].get("category") == G.nodes[v].get("category")

    if same_cat:
        base  = _cat_color(G.nodes[u].get("category", ""))
        width = 1.2 + (sim - 0.65) * 6
        color = base + "88"
    else:
        width = 2.5
        color = "#FFD70099"

    net.add_edge(
        u, v,
        width=round(width, 1),
        color=color,
        title=f"similarity: {sim:.3f}",
        smooth={"type": "curvedCW", "roundness": 0.15},
    )


# Generate base HTML
html = net.generate_html()


# Stats
n_nodes     = G.number_of_nodes()
n_edges     = G.number_of_edges()
cross_edges = sum(
    1 for u, v in G.edges()
    if G.nodes[u].get("category") != G.nodes[v].get("category")
)


# Legend chips (pre-computed to keep f-string clean)
legend_chips_html = "\n".join(
    f'<div class="legend-chip" onclick="filterCategory(\'{cat}\')">'
    f'  <div style="width:9px;height:9px;border-radius:50%;'
    f'background:{_cat_color(cat)};flex-shrink:0"></div>'
    f'  <span style="font-size:11px;color:#bbb">{cat}</span>'
    f'</div>'
    for cat in all_cats
)


# JavaScript — click-to-highlight + category filter + info panel
# Injected AFTER pyvis's drawGraph() runs, so `network` is already global.
node_meta_js = json.dumps(node_meta, ensure_ascii=False)

INTERACT_JS = (
    "<script>\n"
    "(function init() {\n"
    "  // network is assigned globally by pyvis's drawGraph(); poll until ready\n"
    "  if (typeof network === 'undefined') { setTimeout(init, 80); return; }\n"
    "\n"
    "  var NODE_META = " + node_meta_js + ";\n"
    "\n"
    "  var allNodeIds = network.body.data.nodes.getIds();\n"
    "  var allEdgeIds = network.body.data.edges.getIds();\n"
    "\n"
    "  // Snapshot original colors so we can restore them\n"
    "  var origNode = {};\n"
    "  allNodeIds.forEach(function(id) {\n"
    "    var n = network.body.data.nodes.get(id);\n"
    "    origNode[id] = { color: n.color, opacity: 1 };\n"
    "  });\n"
    "  var origEdge = {};\n"
    "  allEdgeIds.forEach(function(id) {\n"
    "    var e = network.body.data.edges.get(id);\n"
    "    origEdge[id] = { color: e.color, width: e.width };\n"
    "  });\n"
    "\n"
    "  var isHighlighted = false;\n"
    "  var panel = document.getElementById('node-info-panel');\n"
    "\n"
    "  // ---- Info panel helpers --------------------------------------------------\n"
    "  function showPanel(nodeId) {\n"
    "    var m = NODE_META[String(nodeId)];\n"
    "    if (!m) return;\n"
    "    panel.innerHTML =\n"
    "      '<div style=\"font-size:11px;font-weight:600;color:#666;text-transform:uppercase;"
    "letter-spacing:.08em;margin-bottom:8px\">Selected Article</div>' +\n"
    "      '<div style=\"font-size:13px;font-weight:700;color:#fff;margin-bottom:6px;"
    "line-height:1.4\">' + m.title + '</div>' +\n"
    "      '<div style=\"display:flex;align-items:center;gap:6px;margin-bottom:10px\">' +\n"
    "      '  <div style=\"width:9px;height:9px;border-radius:50%;background:' + m.color + "
    "';flex-shrink:0\"></div>' +\n"
    "      '  <span style=\"font-size:11px;color:#999\">' + m.category + '</span>' +\n"
    "      '</div>' +\n"
    "      '<a href=\"' + m.url + '\" target=\"_blank\" style=\"display:block;font-size:11px;"
    "color:#5ba3f5;word-break:break-all;line-height:1.5\">' + m.url + '</a>' +\n"
    "      '<div style=\"margin-top:10px;font-size:10px;color:#444\">Click canvas to reset</div>';\n"
    "    panel.style.display = 'block';\n"
    "  }\n"
    "\n"
    "  function hidePanel() {\n"
    "    panel.style.display = 'none';\n"
    "    panel.innerHTML = '';\n"
    "  }\n"
    "\n"
    "  // ---- Apply highlight for a single node + its neighbors ------------------\n"
    "  function highlightNode(clickedId) {\n"
    "    var neighbors    = new Set(network.getConnectedNodes(clickedId));\n"
    "    neighbors.add(String(clickedId));\n"
    "    var connEdges    = new Set(network.getConnectedEdges(clickedId));\n"
    "\n"
    "    var nodeUp = allNodeIds.map(function(id) {\n"
    "      return neighbors.has(String(id))\n"
    "        ? { id: id, color: origNode[id].color, opacity: 1.0 }\n"
    "        : { id: id, color: { background: '#151525', border: '#ffffff0a' }, opacity: 0.15 };\n"
    "    });\n"
    "    network.body.data.nodes.update(nodeUp);\n"
    "\n"
    "    var edgeUp = allEdgeIds.map(function(id) {\n"
    "      return connEdges.has(id)\n"
    "        ? { id: id, color: origEdge[id].color, width: origEdge[id].width + 0.8, opacity: 1.0 }\n"
    "        : { id: id, color: '#ffffff06', width: 0.4, opacity: 0.08 };\n"
    "    });\n"
    "    network.body.data.edges.update(edgeUp);\n"
    "\n"
    "    isHighlighted = true;\n"
    "    showPanel(clickedId);\n"
    "  }\n"
    "\n"
    "  // ---- Reset to original state --------------------------------------------\n"
    "  function resetAll() {\n"
    "    network.body.data.nodes.update(\n"
    "      allNodeIds.map(function(id) {\n"
    "        return { id: id, color: origNode[id].color, opacity: 1.0 };\n"
    "      })\n"
    "    );\n"
    "    network.body.data.edges.update(\n"
    "      allEdgeIds.map(function(id) {\n"
    "        return { id: id, color: origEdge[id].color, width: origEdge[id].width, opacity: 1.0 };\n"
    "      })\n"
    "    );\n"
    "    isHighlighted = false;\n"
    "    hidePanel();\n"
    "  }\n"
    "\n"
    "  // ---- Click handler -------------------------------------------------------\n"
    "  network.on('click', function(params) {\n"
    "    if (params.nodes.length > 0) {\n"
    "      highlightNode(params.nodes[0]);\n"
    "    } else {\n"
    "      if (isHighlighted) resetAll();\n"
    "    }\n"
    "  });\n"
    "\n"
    "  // ---- Category filter (called from legend chips) -------------------------\n"
    "  window.filterCategory = function(cat) {\n"
    "    var catSet = new Set(\n"
    "      allNodeIds.filter(function(id) {\n"
    "        return NODE_META[String(id)] && NODE_META[String(id)].category === cat;\n"
    "      }).map(String)\n"
    "    );\n"
    "    if (catSet.size === 0) return;\n"
    "\n"
    "    var catEdges = new Set();\n"
    "    allEdgeIds.forEach(function(eid) {\n"
    "      var e = network.body.data.edges.get(eid);\n"
    "      if (catSet.has(String(e.from)) && catSet.has(String(e.to))) catEdges.add(eid);\n"
    "    });\n"
    "\n"
    "    network.body.data.nodes.update(\n"
    "      allNodeIds.map(function(id) {\n"
    "        return catSet.has(String(id))\n"
    "          ? { id: id, color: origNode[id].color, opacity: 1.0 }\n"
    "          : { id: id, color: { background: '#151525', border: '#ffffff0a' }, opacity: 0.15 };\n"
    "      })\n"
    "    );\n"
    "    network.body.data.edges.update(\n"
    "      allEdgeIds.map(function(id) {\n"
    "        return catEdges.has(id)\n"
    "          ? { id: id, color: origEdge[id].color, width: origEdge[id].width, opacity: 1.0 }\n"
    "          : { id: id, color: '#ffffff06', width: 0.4, opacity: 0.08 };\n"
    "      })\n"
    "    );\n"
    "\n"
    "    isHighlighted = true;\n"
    "    hidePanel();\n"
    "    network.fit({\n"
    "      nodes: Array.from(catSet),\n"
    "      animation: { duration: 700, easingFunction: 'easeInOutQuad' }\n"
    "    });\n"
    "  };\n"
    "\n"
    "  // ---- Reset button -------------------------------------------------------\n"
    "  window.resetGraph = function() { resetAll(); };\n"
    "\n"
    "})();\n"
    "</script>\n"
)


# UI block (CSS + HTML panels + JS)
UI_BLOCK = f"""
<style>
  body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}

  /* ---- Left panel strip ---- */
  #graph-ui {{
    position: fixed; top: 16px; left: 16px; z-index: 9999;
    display: flex; flex-direction: column; gap: 10px;
    pointer-events: none;
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #333 transparent;
  }}
  .panel {{
    background: rgba(8,8,22,0.90);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 12px 16px;
    backdrop-filter: blur(12px);
    pointer-events: auto;
    min-width: 180px;
  }}
  .panel-title {{
    font-size: 10px; font-weight: 700; color: #555;
    text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px;
  }}
  .stat-row {{ display: flex; gap: 16px; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 22px; font-weight: 700; color: #fff; line-height: 1; }}
  .stat-label {{ font-size: 10px; color: #555; margin-top: 2px; }}
  .hint {{ font-size: 10px; color: #484860; margin-top: 8px; line-height: 1.6; }}

  /* ---- Legend chips ---- */
  .legend-chip {{
    display: flex; align-items: center; gap: 7px;
    margin-bottom: 3px; cursor: pointer;
    border-radius: 5px; padding: 3px 5px;
    transition: background .12s;
  }}
  .legend-chip:hover {{ background: rgba(255,255,255,0.07); }}

  /* ---- Reset button ---- */
  .reset-btn {{
    display: block; width: 100%; margin-top: 8px;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    color: #888; font-size: 11px; border-radius: 6px;
    padding: 5px 10px; cursor: pointer; text-align: center;
    transition: background .15s;
  }}
  .reset-btn:hover {{ background: rgba(255,255,255,0.12); color: #ccc; }}

  /* ---- Right info panel (shown on node click) ---- */
  #node-info-panel {{
    position: fixed; top: 16px; right: 16px; z-index: 9999;
    background: rgba(8,8,22,0.93);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 18px;
    backdrop-filter: blur(14px);
    max-width: 300px;
    display: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }}
</style>

<!-- ========= LEFT UI ========= -->
<div id="graph-ui">

  <!-- Stats card -->
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
        <div class="stat-label">Cross-cat</div>
      </div>
    </div>
    <div class="hint">
      🔵 Click node → highlight neighbors<br>
      🏷️ Click legend → filter category<br>
      ⬛ Click canvas → reset
    </div>
  </div>

  <!-- Category legend -->
  <div class="panel">
    <div class="panel-title">Categories</div>
    {legend_chips_html}
    <button class="reset-btn" onclick="resetGraph()">↺ Reset view</button>
    <div class="hint" style="margin-top:8px">
      <span style="display:inline-block;width:16px;height:2px;
            background:rgba(255,255,255,0.18);vertical-align:middle"></span>
      &thinsp;Same-cat &nbsp;
      <span style="display:inline-block;width:16px;height:2px;
            background:#FFD700;vertical-align:middle"></span>
      &thinsp;Cross-cat
    </div>
  </div>

</div>

<!-- ========= RIGHT INFO PANEL ========= -->
<div id="node-info-panel"></div>

{INTERACT_JS}
"""

html = html.replace("</body>", UI_BLOCK + "\n</body>")

out_path = Path(__file__).parent / "kb_graph.html"
out_path.write_text(html, encoding="utf-8")
print(f"✓ Saved: {out_path}")
print(f"  {n_nodes} articles · {n_edges} edges · {cross_edges} cross-category edges")
webbrowser.open(str(out_path))
