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
    for i in range(17):
        if i < 10:
            node_name = f"e{i+1}"
            node_color = "green"
        else:
            node_name = f"d{i-9}"
            node_color = "purple"
        G.add_node(i, name=node_name, color=node_color)

    # Add edges based on the rule:
    # An edge exists between node i and node j if both W[i,j] and W[j,i] are nonzero.
    # The edge weight is the average of W[i,j] and W[j,i].
    edge_weights = []
    for i in range(17):
        for j in range(i+1, 17):
            if W[i, j] != 0 and W[j, i] != 0:
                weight = (W[i, j] + W[j, i]) / 2.0
                # if weight > 2:  # Only add edges with weight greater than a threshold
                G.add_edge(i, j, weight=weight)
                edge_weights.append(weight)

    pos_weights = [w for w in edge_weights if w > 0]
    neg_weights = [w for w in edge_weights if w < 0]
    if pos_weights:
        pos_min = min(pos_weights)
        pos_max = max(pos_weights)
    if neg_weights:
        # For negatives, we'll normalize based on the absolute values.
        neg_min_abs = min(abs(w) for w in neg_weights)
        neg_max_abs = max(abs(w) for w in neg_weights)

    # Compute edge colors and widths based on weight:
    # Positive weights: use blue (darker blue for larger positive values)
    # Negative weights: use red (darker red for larger absolute negative values)
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight > 0:
            if pos_max > pos_min:
                norm = (weight - pos_min) / (pos_max - pos_min)
            else:
                norm = 0
            # Map normalized positive weight to blue: higher norm -> darker blue
            color = plt.cm.Blues(norm)
            width = 1 + norm * 5  # edge width scaled between 1 and 6
        elif weight < 0:
            abs_weight = abs(weight)
            if neg_max_abs > neg_min_abs:
                norm = (abs_weight - neg_min_abs) / (neg_max_abs - neg_min_abs)
            else:
                norm = 0
            # Map normalized absolute negative weight to red: higher norm -> darker red
            color = plt.cm.Reds(norm)
            width = 1 + norm * 5  # edge width scaled between 1 and 6
        else:
            # Fallback for zero (if ever encountered)
            color = (0.5, 0.5, 0.5, 1)  # gray
            width = 1
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
    # Draw node labels (display the node name)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    # Draw edges with very thin lines and the computed grayscale colors
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)

    plt.axis('off')
    plt.title("Graph Visualization")
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
