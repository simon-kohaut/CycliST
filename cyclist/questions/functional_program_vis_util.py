import networkx as nx
import matplotlib.pyplot as plt

def visualize_functional_program(program_steps):
    """
    Visualizes a functional program represented as a list of dicts using networkx.
    Each step should include 'type', 'inputs', '_output', and 'value_inputs'.
    """
    G = nx.DiGraph()

    # Map node index to step for lookup and visualization
    node_labels = {}
    value_nodes = []

    for idx, step in enumerate(program_steps):
        node_name = f"{idx}:{step['type']}"
        G.add_node(node_name)
        node_labels[node_name] = node_name

        # Add edges for inputs
        for inp in step.get("inputs", []):
            inp_step = f"{inp}:{program_steps[inp]['type']}"
            G.add_edge(inp_step, node_name)

        # Add value inputs as separate nodes
        for i, val in enumerate(step.get("value_inputs", [])):
            val_node = f"{node_name}_val{i}"
            G.add_node(val_node)
            G.add_edge(val_node, node_name)
            node_labels[val_node] = f"value: {val}"
            value_nodes.append(val_node)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['lightblue' if n not in value_nodes else 'lightgreen' for n in G.nodes]
    nx.layout.bipartite_layout(G=G,nodes=node_labels)  # for consistent layout
    nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=node_colors,
            node_size=1000, font_size=8, font_weight='bold', arrows=True, arrowstyle='-|>')


    plt.title("Functional Program Dependency Graph")
    plt.tight_layout()
    plt.show()