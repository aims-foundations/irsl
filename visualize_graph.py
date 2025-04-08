import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # W = torch.load('W.pt').detach().cpu().numpy()
    # print(W.shape)

    df = pd.read_csv('ggm_gsm_network_matrix.csv')
    W = torch.tensor(df.values, dtype=torch.float32)
    
    # Calculate and print percentage of zeros in the matrix
    total_elements = W.size
    zero_count = np.sum(W == 0)
    zero_percentage = (zero_count / total_elements) * 100
    print("Percentage of zeros in the matrix: {:.2f}%".format(zero_percentage))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(W, cmap='viridis')
    plt.title("Matrix Visualization")
    plt.colorbar(label="Value")
    plt.savefig("matrix_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create an undirected graph
    G = nx.Graph()
    
    # Determine number of nodes from the CSV columns and get node names
    num_nodes = W.shape[1]
    node_names = df.columns.tolist()
    
    # Add nodes: use each CSV column name as the node name and assign all nodes the color red
    for i in range(num_nodes):
        G.add_node(i, name=node_names[i], color="red")
    
    # Add edges based on the rule:
    # An edge exists between nodes i and j if both W[i, j] and W[j, i] are nonzero.
    # The edge weight is the average of W[i, j] and W[j, i].
    edge_weights = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if W[i, j] != 0 and W[j, i] != 0:
                weight = (W[i, j] + W[j, i]) / 2.0
                weight = float(weight.item()) if hasattr(weight, 'item') else float(weight)
                G.add_edge(i, j, weight=weight)
                edge_weights.append(weight)
    
    # Determine minimum and maximum edge weights (using raw values, without absolute conversion)
    if edge_weights:
        min_edge = min(edge_weights)
        max_edge = max(edge_weights)
    else:
        min_edge = max_edge = 0  # fallback if no edges are added
    
    # Compute edge colors and widths based on weight:
    # For each edge, normalize its weight in the range [0,1] and then map to an adjusted value
    # in the interval [0.1, 0.9]. This adjusted value is then fed to the Blues colormap.
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if max_edge != min_edge:
            norm = (weight - min_edge) / (max_edge - min_edge)
        else:
            norm = 0
        # Adjust the normalized value to be within [0.1, 0.9]
        adjusted_norm = 0.1 + norm * 0.8
        color = plt.cm.Blues(adjusted_norm)
        width = 1 + norm * 5  # Edge width scaled between 1 and 6
        edge_colors.append(color)
        edge_widths.append(width)
    
    # Prepare node colors and labels for drawing
    node_color_list = [data['color'] for _, data in G.nodes(data=True)]
    node_labels = {i: data['name'] for i, data in G.nodes(data=True)}
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # layout for positioning nodes
    
    # Draw nodes with their colors
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=300)
    # Draw node labels using the node names from the CSV file
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    # Draw edges using the computed blue colors and edge widths
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    
    plt.axis('off')
    plt.title("Graph Visualization")
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()