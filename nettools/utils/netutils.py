import os
import random
import numpy as np
from nettools.multiplex.syn_mul_gen import MultiplexNetwork
from nettools.monoplex.syn_net_gen import Network

NX_CENTRALITY = {
    "degree": "degree_centrality",
    "closeness": "closeness_centrality",
    "betweenness": "betweenness_centrality",
    "eigenvector": "eigenvector_centrality",
    "pagerank": "pagerank_numpy",
    "hits": "hits_numpy",
    "k-shell": True,
    "voterank": True,
    "supernode": True
}

MONOPLEX_DB = {
    "arxiv-physics": "ca-AstroPh.mtx",
    "facebook": "socfb-Berkeley13.mtx",
    "messages": "ia-fb-messages.mtx",
    "edu": "web-edu.mtx",
    "usa-airport": "usairport.mtx",
    "facebook_small": "socfb-Amherst41.mtx"
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
    "EUAir": {
        "files": ("EUAirTransportation_layers.txt",
                  "EUAirTransportation_multiplex.edges", "EUAirTransportation_nodes.txt"),
        "directed": False
    },
}


def load_monoplex_by_name(name):
    return load_mtx(MONOPLEX_DB[name])


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
            network_graph_np[node2id[ed_o], node2id[ed_t], int(l_nr) - 1] = 1
            network_weights_np[node2id[ed_o], node2id[ed_t], int(l_nr) - 1] = float(weight)
            # Symmetry for undirected networks
            if not directed:
                network_graph_np[node2id[ed_t], node2id[ed_o], int(l_nr) - 1] = 1
                network_weights_np[node2id[ed_t], node2id[ed_o], int(l_nr) - 1] = float(weight)
    return MultiplexNetwork(network_graph_np, weights_layer=network_weights_np,
                            weighted=True, mappings=mappings, layers_attr=layers_attr)


def load_mtx(path_nodes):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    db_edges = os.path.join(curr_dir, "..", "..", "data/networks", path_nodes)

    with open(db_edges, 'r') as db_edges:
        file_lines = db_edges.readlines()

    mat_size = int(file_lines[1].split()[0])
    numpy_mat = np.zeros((mat_size, mat_size), dtype=np.uint8)
    for x_line in file_lines[2:]:
        cord_1, cord_2 = x_line.split()[:2]
        numpy_mat[int(cord_1) - 1, int(cord_2) - 1] = 1
    return Network(numpy_mat)


def sample_from_dist(dist, n_samples=1):
    # Compute cumsum
    samples = []
    cum_sum = np.cumsum(dist, axis=0)
    iteration = 0
    while iteration < n_samples:
        rnd = random.uniform(0, 1)
        # If small than first
        if rnd < cum_sum[0]:
            samples.append(0)
            continue
        # for others
        counter = 0
        while rnd >= cum_sum[counter]:
            counter += 1
        # No repeats
        if counter in samples:
            continue
        samples.append(counter)
        iteration += 1
    return samples


if __name__ == '__main__':
    from nettools.monoplex.centrality import CentralityMeasure
    import matplotlib.pyplot as plt
    net = load_mtx("socfb-Berkeley13.mtx")
    # net.plot_degree_dist()
    cm = CentralityMeasure(net.network)
    result = cm.kshell(no_crust=False)
    best_nodes = sorted(result.items(), key=lambda x: x[1])[::-1]
    nodes_sc = [x[1] for x in best_nodes]
    plt.plot(np.sort(nodes_sc))
    plt.show()
    print()
    # load_multinet_by_name('london')
