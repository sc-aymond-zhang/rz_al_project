import networkx as nx
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
import heapq
import os

# === Load Graph and Roles ===
def load_graph_from_txt(file_path):
    graph = {}
    roles = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == 'ROLE':
                node, role = parts[1], parts[2]
                roles[node] = role
                continue

            if len(parts) != 4:
                continue  # skip malformed lines

            u, v, w_str, direction = parts
            w = float(w_str)

            if u not in graph:
                graph[u] = []
            graph[u].append((v, w, direction))

            if direction == 'bi':
                if v not in graph:
                    graph[v] = []
                graph[v].append((u, w, direction))

            if v not in graph:
                graph[v] = []
    
    vertices = list(graph.keys())
    return graph, vertices, roles

file_path = os.path.join(os.path.dirname(__file__), "graph.txt")
assert os.path.exists(file_path), "Graph file does not exist!"
graph, vertices, roles = load_graph_from_txt(file_path)

# === Modified Dijkstra using multiplication
def multiplicative_dijkstra(graph, vertices, start):
    adj = {v: [] for v in vertices}
    for u in graph:
        for v, w, direction in graph[u]:
            adj[u].append((v, w))
            if direction == 'bi':
                adj[v].append((u, w))

    distances = {v: float('inf') for v in vertices}
    distances[start] = 1  # multiplicative identity

    visited = set()
    heap = [(1, start)]

    while heap:
        curr_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            new_dist = curr_dist * w
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return distances

# === Build NetworkX Graph for visualization
def build_nx_graph(graph):
    G_directed = nx.DiGraph()
    G_undirected = nx.Graph()
    for u in graph:
        for v, w, direction in graph[u]:
            if direction == 'bi':
                G_undirected.add_edge(u, v, weight=w)
            else:
                G_directed.add_edge(u, v, weight=w)
    return G_directed, G_undirected

G_directed, G_undirected = build_nx_graph(graph)

# Combine for layout
G_all = nx.Graph()
G_all.add_nodes_from(G_directed.nodes())
G_all.add_nodes_from(G_undirected.nodes())
pos = nx.circular_layout(G_all)

# === Plotly traces for edges and nodes
edge_x = []
edge_y = []
for u, v in G_directed.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
for u, v in G_undirected.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Edge labels
edge_text_x = []
edge_text_y = []
edge_text_values = []
for u, v in G_directed.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_text_x.append((x0 + x1) / 2)
    edge_text_y.append((y0 + y1) / 2)
    edge_text_values.append(str(G_directed[u][v]['weight']))
for u, v in G_undirected.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_text_x.append((x0 + x1) / 2)
    edge_text_y.append((y0 + y1) / 2)
    edge_text_values.append(str(G_undirected[u][v]['weight']))

edge_label_trace = go.Scatter(
    x=edge_text_x,
    y=edge_text_y,
    text=edge_text_values,
    mode='text',
    textfont=dict(color='darkred', size=14),
    hoverinfo='none',
    showlegend=False
)

# Node positions and labels
node_x = []
node_y = []
for node in G_all.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=list(G_all.nodes()),
    textposition="bottom center",
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color='skyblue',
        size=30,
        line_width=2,
        line_color='black'
    )
)

# === Dash App Setup
app = Dash(__name__)
server = app.server  # <-- Add this line for deployment

app.layout = html.Div([
    html.H2("Select a symptom to find top 10 most likely conditions to cause it"),

    dcc.Dropdown(
        id='start-node',
        options=[{'label': v, 'value': v} for v in vertices],
        placeholder="Select a starting node...",
        style={'width': '300px', 'marginBottom': '20px'}
    ),

    html.Button("Show Graph", id="toggle-button", n_clicks=0),

    dcc.Store(id='graph-visible', data=False),

    html.Div(id='graph-container'),

    html.Div(id='output', style={'fontSize': '18px', 'marginTop': '10px'})
])

@app.callback(
    Output('graph-visible', 'data'),
    Input('toggle-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_graph(n_clicks):
    return n_clicks % 2 == 1

@app.callback(
    Output('graph-container', 'children'),
    Output('toggle-button', 'children'),
    Input('graph-visible', 'data')
)
def update_graph_display(is_visible):
    if is_visible:
        return (
            dcc.Graph(
                id='graph',
                figure=go.Figure(
                    data=[edge_trace, edge_label_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    )
                )
            ),
            "Hide Graph"
        )
    else:
        return html.Div(), "Show Graph"

@app.callback(
    Output('output', 'children'),
    Input('start-node', 'value'),
)
def display_top_causers(selected_node):
    if not selected_node:
        return "Select a symptom to see the top 10 medical conditions and their path weights."

    distances = multiplicative_dijkstra(graph, vertices, selected_node)

    filtered = {
        node: dist for node, dist in distances.items()
        if node != selected_node and dist != float('inf') and roles.get(node) in ['causer', 'both']
    }

    if not filtered:
        return f"No reachable medical conditions from {selected_node}."

    # Sort by greatest to least path length
    closest = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:10]
    output = [f"Top 10 medical conditions from '{selected_node}' by descending multiplicative path weight:"]
    for i, (node, dist) in enumerate(closest, 1):
        output.append(f"{i}. {node} (multiplicative path weight: {dist:.4f})")

    return html.Ul([html.Li(item) for item in output])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default to 10000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)