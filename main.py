import os
import json
import heapq
import requests
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input, State

# ======================================================
# === CONFIG & CONSTANTS ===
# ======================================================
FILE_PATH = os.path.join(os.path.dirname(__file__), "kaggledata.txt")
assert os.path.exists(FILE_PATH), "Graph file does not exist!"

# --- DeepSeek config ---
DEEPSEEK_MODEL = "deepseek/deepseek-r1:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

_JSON_INSTRUCTIONS = """
You are a clinical text symptom extractor. Your only task is to read a patient case or medical note and return symptoms from the ALLOWED_SYMPTOM_CODES list.

RULES:
1. Consider demographic details (age, gender, occupation, etc.) 
2. Identify symptom descriptions even if they are expressed indirectly (e.g., "pain in his throat" → sore_throat, "nose is super stuffy" → nasal_congestion).
3. Match based on meaning/semantic similarity. Synonyms, slang, and descriptive phrases should be mapped to the correct code.
4. Only include codes exactly as they appear in the allowed list.
5. Do NOT include diagnoses, diseases, causes, or risk factors (e.g., pneumonia, COVID-19, allergies).
6. If no symptoms from the allowed list are present, return {"selected": []}.
7. Output MUST be a single valid JSON object in the format below.
8. No text or explanation before or after the JSON.

CRITICAL: Reply with ONLY valid JSON in the exact shape:
{"selected": ["symptom_code1", "symptom_code2"]}
"""

# ======================================================
# === UTILITY FUNCTIONS ===
# ======================================================
def normalize_code(name):
    return name.strip().lower().replace(" ", "_")

def _extract_json_object(text):
    """Extract the first valid JSON object from text."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return {}
    return {}

def extract_symptoms_with_ai(user_text, allowed_codes):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _JSON_INSTRUCTIONS + "\nALLOWED_SYMPTOM_CODES:\n" + json.dumps(allowed_codes)},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    ai_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        result = json.loads(ai_text)
        return result.get("selected", [])
    except:
        # Graceful fallback—return none if into parsing issues
        return []

# ======================================================
# === GRAPH LOADING ===
# ======================================================
def load_graph(file_path):
    graph, roles = {}, {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == 'ROLE':
                roles[normalize_code(parts[1])] = parts[2].lower()
                continue
            if len(parts) != 4:
                continue
            u, v, w, direction = normalize_code(parts[0]), normalize_code(parts[1]), float(parts[2]), parts[3]
            graph.setdefault(u, []).append((v, w, direction))
            graph.setdefault(v, [])
            if direction == 'bi':
                graph[v].append((u, w, direction))
    return graph, list(graph.keys()), roles

graph, vertices, roles = load_graph(FILE_PATH)
SYMPTOM_CODES = sorted(k for k, v in roles.items() if v in ("causer", "both"))
CONDITION_CODES = sorted(k for k, v in roles.items() if v in ("effect", "both"))

# ======================================================
# === AI SYMPTOM EXTRACTION ===
# ======================================================
def ai_extract_symptoms(user_input, roles):
    allowed = SYMPTOM_CODES
    try:
        selected = extract_symptoms_with_ai(user_input, allowed)
        return sorted(set(normalize_code(s) for s in selected if normalize_code(s) in allowed))
    except Exception as e:
        print("AI extraction failed:", e)
        return []

# ======================================================
# === MULTIPLICATIVE DIJKSTRA ===
# ======================================================
def multiplicative_dijkstra(graph, start):
    adj = {u: [(v, w) for v, w, _ in edges] for u, edges in graph.items()}
    distances = {v: float('inf') for v in graph}
    distances[start] = 1
    visited, heap = set(), [(1, start)]
    while heap:
        curr_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            new_dist = curr_dist * w
            if new_dist <= distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return distances

# ======================================================
# === BUILD FILTERED GRAPH ===
# ======================================================
def build_filtered_graph(graph, start_nodes, hops=2):
    G = nx.Graph()
    for u, edges in graph.items():
        for v, w, _ in edges:
            G.add_edge(u, v, weight=w)

    nodes_to_include, frontier = set(start_nodes), set(start_nodes)
    for _ in range(hops):
        new_frontier = {n for node in frontier for n in G.neighbors(node)}
        nodes_to_include.update(new_frontier)
        frontier = new_frontier

    subG = G.subgraph(nodes_to_include).copy()
    pos = nx.spring_layout(subG, k=0.7, seed=42)

    # Edges
    edge_x, edge_y, edge_text_x, edge_text_y, edge_vals = [], [], [], [], []
    for u, v in subG.edges():
        x0, y0, x1, y1 = *pos[u], *pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_vals.append(f"{subG[u][v]['weight']:.2f}")
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                             hoverinfo='none', mode='lines')
    edge_label_trace = go.Scatter(x=edge_text_x, y=edge_text_y, text=edge_vals,
                                  mode='text', textfont=dict(color='darkred', size=12),
                                  hoverinfo='none', showlegend=False)

    # Nodes
    node_x, node_y, node_text = zip(*[(pos[n][0], pos[n][1], n) for n in subG.nodes()])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                             hoverinfo='text', text=node_text,
                             marker=dict(color='skyblue', size=20, line_width=2, line_color='black'))
    return edge_trace, edge_label_trace, node_trace

# ======================================================
# === DASH APP ===
# ======================================================
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "maxWidth": "900px",
        "margin": "auto",
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "12px",
        "boxShadow": "0px 4px 10px rgba(0,0,0,0.1)"
    },
    children=[
        html.H1(
            "🩺 Symptom & Condition Explorer",
            style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"}
        ),

        html.Div([
            html.Label("Select Symptoms:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id='start-node',
                options=[{'label': r.replace("_", " "), 'value': r} for r in SYMPTOM_CODES],
                value=[],
                multi=True,
                style={"marginBottom": "20px"}
            ),
        ]),

        html.Div([
            html.Label("Or Describe Your Symptoms:", style={"fontWeight": "bold"}),
            dcc.Textarea(
                id='symptom-text',
                placeholder="Example: My throat feels sore and my nose is stuffy...",
                style={
                    'width': '100%',
                    'height': '100px',
                    'marginBottom': '10px',
                    'borderRadius': '8px',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'resize': 'none'
                }
            ),
            html.Button(
                "➕ Add Symptoms",
                id='add-symptoms-btn',
                n_clicks=0,
                style={
                    "backgroundColor": "#27ae60",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 15px",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            ),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Button(
                "📊 Show/Hide Graph",
                id='toggle-button',
                n_clicks=0,
                style={
                    "backgroundColor": "#2980b9",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 15px",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "marginBottom": "10px"
                }
            ),
            dcc.Store(id='graph-visible', data=False),
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=html.Div(id='graph-container')
            ),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.H3("📋 Results", style={"marginBottom": "10px", "color": "#34495e"}),
            dcc.Loading(
                id="loading-output",
                type="circle",
                children=html.Div(
                    id='output',
                    style={
                        'fontSize': '16px',
                        'backgroundColor': 'white',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'border': '1px solid #ddd'
                    }
                )
            ),
        ])
    ]
)

# ======================================================
# === CALLBACKS ===
# ======================================================
@app.callback(
    Output('start-node', 'value'),
    Input('add-symptoms-btn', 'n_clicks'),
    State('symptom-text', 'value'),
    State('start-node', 'value'),
    prevent_initial_call=True
)
def add_symptoms_from_text(n_clicks, text_value, current_selection):
    current_selection = current_selection or []
    if not text_value:
        return current_selection
    inferred = ai_extract_symptoms(text_value, roles)
    return current_selection + [s for s in inferred if s not in current_selection]

@app.callback(
    Output('graph-container', 'children'),
    Output('toggle-button', 'children'),
    Input('graph-visible', 'data'),
    Input('start-node', 'value')
)
def update_graph_display(is_visible, selected_nodes):
    if not is_visible or not selected_nodes:
        return html.Div(), "Show Graph"
    edge_trace, edge_label_trace, node_trace = build_filtered_graph(graph, selected_nodes, hops=2)
    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     height=600))
    return dcc.Graph(id='graph', figure=fig), "Hide Graph"

@app.callback(
    Output('output', 'children'),
    Input('start-node', 'value'),
    State('symptom-text', 'value')
)
def display_with_natural_language(selected_nodes, text_value):
    matched = [normalize_code(m) for m in (selected_nodes or [])]

    # If both inputs are empty
    if not matched and not (text_value and text_value.strip()):
        return "Waiting on input"

    # If user provided input but nothing matched
    if not matched:
        return "No symptoms matched. Please try again."

    # We have matches → mark as complete and show results
    per_symptom_dist = [multiplicative_dijkstra(graph, s) for s in matched]
    candidates = [
        n for n in vertices
        if roles.get(n, "") in ['effect', 'both'] and n not in matched
    ]

    avg_scores, match_counts = {}, {}
    for n in candidates:
        total, matches = 0.0, 0
        for dist in per_symptom_dist:
            d = dist.get(n, float('inf'))
            if d != float('inf'):
                total += d
                matches += 1
        if matches > 0:
            avg_scores[n] = total / len(matched)
            match_counts[n] = matches

    closest = sorted(
        avg_scores.items(),
        key=lambda x: (x[1], match_counts.get(x[0], 0)),
        reverse=True
    )[:10]

    return html.Div([
        html.Div("Complete", style={"fontWeight": "bold", "marginBottom": "10px"}),
        html.Ul([
            html.Li(f"{i+1}. {n.replace('_', ' ')} "
                    f"(avg weight: {avg_scores[n]:.4f}; matches: {match_counts[n]})")
            for i, (n, _) in enumerate(closest)
        ])
    ])

@app.callback(
    Output('output', 'children', allow_duplicate=True),
    Input('add-symptoms-btn', 'n_clicks'),
    prevent_initial_call=True
)
def show_loading_message(n_clicks):
    return "Loading..."

@app.callback(
    Output('graph-visible', 'data'),
    Input('toggle-button', 'n_clicks'),
    State('graph-visible', 'data'),
    prevent_initial_call=True 
)
def toggle_graph(n_clicks, current_state):
    if n_clicks is None:
        return current_state
    return not current_state

# ======================================================
# === RUN APP ===
# ======================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=false)


