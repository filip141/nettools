import os
import pymnet
import numpy as np

NX_CENTRALITY = {
    "degree": "degree_centrality",
    "closeness": "closeness_centrality",
    "betweenness": "betweenness_centrality",
    "eigenvector": "eigenvector_centrality",
    "pagerank": "pagerank_numpy",
    "hits": "hits_numpy",
    "k-shell": "core_number",
    "voterank": True,
}

DB_LIST = {
    "london": {
        "files": ("london_transport_layers.txt", "london_transport_multiplex.edges",
                  "london_transport_nodes.txt"),
        "directed": False
    },
    "fao": {
        "files": ("fao_trade_layers.txt", "fao_trade_multiplex.edges", "fao_trade_nodes.txt"),
        "directed": True
    },
}


def load_multinet_by_name(name):
    db_tuple = DB_LIST[name]['files']
    return load_multinet(db_tuple[0], db_tuple[1], db_tuple[2], directed=bool(DB_LIST[name]['directed']))


def load_multinet(path_layers, path_edges, path_nodes, directed=False):
    # Set database paths
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    database_path_nodes = os.path.join(curr_dir, "..", "..", "data/networks", path_nodes)
    database_path_edges = os.path.join(curr_dir, "..", "..", "data/networks", path_edges)
    database_path_layers = os.path.join(curr_dir, "..", "..", "data/networks", path_layers)

    # Define empty graph
    layers_attr = {}
    loaded_network = pymnet.MultiplexNetwork(couplings='categorical')
    node2id = {}
    id2node = {}

    # Load layers
    with open(database_path_layers) as db_layers:
        for l_line in db_layers.readlines():
            if "layerID" in l_line:
                continue
            line_split = l_line.split(" ")
            l_nr, l_name = (int(line_split[0]), " ".join(line_split[1:])[:-1])
            layers_attr[l_nr] = l_name
            # Load layer
            loaded_network.add_layer(l_nr)

    # Load nodes from nodes file
    counter = 0
    with open(database_path_nodes) as db_nodes:
        node_lines = db_nodes.readlines()
        nodes_number = len(node_lines)
        for node_l in node_lines:
            if 'nodeID' in node_l:
                continue
            node_name = node_l.split(" ")[0]
            node_attrs = node_l.split(" ")[1:]
            id2node[counter] = [node_name, node_attrs]
            node2id[node_name] = counter
            counter += 1

    # Define network numpy graph
    mappings = (id2node, node2id)
    network_weights_np = np.zeros((nodes_number, nodes_number, len(layers_attr)))
    network_graph_np = np.zeros((nodes_number, nodes_number, len(layers_attr)))

    # Database edges
    with open(database_path_edges) as db_edges:
        for l_line in db_edges.readlines():
            line_split = l_line.split(" ")
            l_nr, ed_o, ed_t, weight = line_split
            loaded_network.A[int(l_nr)][int(ed_o), int(ed_t)] = 1
            network_graph_np[node2id[ed_o], node2id[ed_t], int(l_nr) - 1] = 1
            network_weights_np[node2id[ed_o], node2id[ed_t], int(l_nr) - 1] = float(weight)
            # Symmetry for undirected networks
            if not directed:
                network_graph_np[node2id[ed_t], node2id[ed_o], int(l_nr) - 1] = 1
                network_weights_np[node2id[ed_t], node2id[ed_o], int(l_nr) - 1] = float(weight)
    return loaded_network, network_graph_np, network_weights_np, mappings, layers_attr


if __name__ == '__main__':
    load_multinet_by_name('london')
