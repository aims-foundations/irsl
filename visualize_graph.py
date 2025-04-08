import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # W = torch.load('W.pt').detach().cpu().numpy()
    df = pd.read_csv('ggm_gsm_network_matrix.csv')
    W = df.values
    print(W.shape)
    
    # Calculate and print percentage of zeros in the matrix
    total_elements = W.size
    zero_count = np.sum(W == 0)
    zero_percentage = (zero_count / total_elements) * 100
    print("Percentage of zeros in the matrix: {:.2f}%".format(zero_percentage))
    
    # Visualize the matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap='viridis')
    plt.title("Matrix Visualization")
    plt.colorbar(label="Value")
    plt.savefig("matrix_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Read JSON mapping from column names to z values
    with open('diffeasy_to_z_17.json', 'r') as f:
        z_mapping = json.load(f)
    
    # Find minimum and maximum z values from the mapping for normalization
    z_values = list(z_mapping.values())
    min_z = min(z_values)
    max_z = max(z_values)
    
    # Create an undirected graph
    G = nx.Graph()
    
    # Determine number of nodes from the CSV columns and get node names
    num_nodes = W.shape[1]
    node_names = df.columns.tolist()
    
    # Add nodes: use each CSV column name as the node name.
    # For the node color, use the z value from the JSON mapping.
    # The color is computed using the Reds colormap, where larger z values are darker red.
    for i in range(num_nodes):
        col_name = node_names[i]
        z_value = z_mapping.get(col_name)  # fallback to min_z if not found
        # Normalize the z value (and adjust to the 0.1 to 0.9 range)
        norm = (z_value - min_z) / (max_z - min_z)
        adjusted_norm = 0.1 + norm * 0.8
        node_color = plt.cm.Reds(adjusted_norm)
        G.add_node(i, name=col_name, color=node_color)
    
    # Add edges based on the rule:
    # An edge exists between nodes i and j if both W[i, j] and W[j, i] are nonzero.
    # The edge weight is the average of W[i, j] and W[j, i].
    edge_weights = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if W[i, j] != 0 and W[j, i] != 0:
                weight = (W[i, j] + W[j, i]) / 2.0
                G.add_edge(i, j, weight=weight)
                edge_weights.append(weight)
    
    # Compute edge colors and widths based on weight:
    # Normalize the edge weight to the range [0.1, 0.9] and use the Blues colormap.
    min_edge = min(edge_weights)
    max_edge = max(edge_weights)
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        norm = (weight - min_edge) / (max_edge - min_edge)
        adjusted_norm = 0.1 + norm * 0.8  # normalize to [0.1, 0.9]
        color = plt.cm.Blues(adjusted_norm)
        width = norm * 3  # Edge width scaled between 0 and 
        edge_colors.append(color)
        edge_widths.append(width)
    
    # Prepare node colors and labels for drawing
    node_color_list = [data['color'] for _, data in G.nodes(data=True)]
    node_labels = {i: data['name'] for i, data in G.nodes(data=True)}
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # layout for positioning nodes
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=300)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    
    plt.axis('off')
    plt.title("Graph Visualization")
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()