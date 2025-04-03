import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    W = torch.load('W.pt').detach().cpu().numpy()
    print(W.shape)

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

    # Add nodes with names and color attributes:
    # The first 100 nodes are 'e1' to 'e100' (blue) and the next 100 nodes are 'd1' to 'd100' (red)
    for i in range(200):
        if i < 100:
            node_name = f"e{i+1}"
            node_color = "blue"
        else:
            node_name = f"d{i-99}"
            node_color = "red"
        G.add_node(i, name=node_name, color=node_color)

    # Add edges based on the rule:
    # An edge exists between node i and node j if both W[i,j] and W[j,i] are nonzero.
    # The edge weight is the average of W[i,j] and W[j,i].
    edge_weights = []
    for i in range(200):
        for j in range(i+1, 200):
            if W[i, j] != 0 and W[j, i] != 0:
                weight = (W[i, j] + W[j, i]) / 2.0
                if weight > 2:  # Only add edges with weight greater than a threshold
                    G.add_edge(i, j, weight=weight)
                    edge_weights.append(weight)

    # Determine minimum and maximum edge weights (to map weights to grayscale)
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)

    # Compute edge colors based on weight:
    # High weight -> darker (closer to black), low weight -> lighter (gray)
    edge_colors = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        # Normalize the weight between 0 and 1
        if max_weight > min_weight:
            norm = (weight - min_weight) / (max_weight - min_weight)
        else:
            norm = 0
        # Invert normalized value: high weight gives low intensity (darker)
        intensity = 1 - norm
        # Use matplotlib's grayscale colormap to obtain an RGBA color tuple
        edge_colors.append(plt.cm.Greys(intensity))

    # Prepare node colors and labels for drawing
    node_color_list = [data['color'] for _, data in G.nodes(data=True)]
    node_labels = {i: data['name'] for i, data in G.nodes(data=True)}

    # Draw the graph
    plt.figure(figsize=(30, 30))
    pos = nx.spring_layout(G, seed=42)  # layout for positioning nodes

    # Draw nodes with their colors
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list, node_size=300)
    # Draw node labels (display the node name)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    # Draw edges with very thin lines and the computed grayscale colors
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color=edge_colors)

    plt.axis('off')
    plt.title("Graph Visualization")
    plt.savefig("graph_visualization.png", dpi=600, bbox_inches="tight")
    plt.close()
