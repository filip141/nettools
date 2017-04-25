import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from nettools.multiplex import MultiplexNetwork


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def running_mean(x, nnodes):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nnodes:] - cumsum[:-nnodes]) / nnodes


class Percolation(object):

    def __init__(self, network):
        # Convert to multiplex object if adjacency passed
        if isinstance(network, np.ndarray):
            self.network = MultiplexNetwork(network)
        elif isinstance(network, MultiplexNetwork):
            self.network = network
        else:
            raise ValueError("Network should be MultiplexNetwork object or numpy.ndarray")

    def remove_nodes(self, probability):
        new_net = MultiplexNetwork(self.network.network.copy())
        per_prob = np.random.uniform(0, 1, size=(self.network.get_nodes_num(),))
        for rmnode in np.where(per_prob < probability)[0]:
            new_net.remove_node(rmnode)
        while True:
            rmdeg = np.sum(new_net.network, axis=1)
            rmsum = np.sum(rmdeg, axis=1)
            rmprod = np.prod(rmdeg, axis=1)
            rmdiff = set(np.where(rmsum > 0)[0]) - set(np.where(rmprod > 0)[0])
            if not rmdiff:
                break
            for rmnode in rmdiff:
                new_net.remove_node(rmnode)
        return len(new_net.giant_connected_component())

    def run(self, visualize=False, npoints=100, log=False, smooth=True, colour='b'):
        gcc_nodes = []
        for proba_nmul in range(0, npoints):
            probability = proba_nmul / float(npoints)
            gcc_nodes.append(self.remove_nodes(probability))
            if log:
                logger.info("Nodes in gcc: {}, probability: {}".format(gcc_nodes[-1], probability))
        # Normalize
        nodes_num = float(self.network.network.shape[0])
        gcc_nodes = np.array(gcc_nodes) / nodes_num
        x_p = np.arange(0, npoints) / float(npoints)
        # Visualize
        if visualize:
            ax = plt.gca()
            ax.set_xlim([0, 1.0])
            ax.set_ylim([0, 1.0])
            if not smooth:
                plt.plot(x_p, gcc_nodes, hold=True)
            plt.plot(x_p[19:], running_mean(gcc_nodes, 20), colour, hold=True)
        return x_p, gcc_nodes


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator, Network
    from nettools.multiplex import MultiplexConstructor
    nodes_nm = 1200
    ng = NetworkGenerator(nodes=nodes_nm)
    # ba1 = ng.bb_network(m0=1)
    ba1 = ng.bb_network(m0=2)
    # ba1 = Network(nx.to_numpy_matrix(nx.barabasi_albert_graph(nodes_nm, m=2)), n_type="BA")
    # er1 = ng.er_network(p=4.0 / float(nodes_nm))
    # er1 = Network(nx.to_numpy_matrix(nx.erdos_renyi_graph(nodes_nm, p=4.0 / float(nodes_nm - 1))), n_type='ER')
    er1 = ng.er_network(p=4.0 / float(nodes_nm - 1))
    mc = MultiplexConstructor()
    # er1 = ng.er_network(p=4.0 / float(nodes_nm))
    # ba1 = ng.ba_network(m0=2)
    # er1 = Network(nx.to_numpy_matrix(nx.erdos_renyi_graph(nodes_nm, p=4.0 / float(nodes_nm))), n_type="ER")
    # ba1 = Network(nx.to_numpy_matrix(nx.barabasi_albert_graph(nodes_nm, m=2)), n_type="BA")
    # bb1 = ng.bb_network(m0=2)
    mnet_er = mc.construct(er1, ba1)
    mnet_ba = mc.construct(ba1)
    # mnet_bb = mc.construct(bb1)
    per_er = Percolation(mnet_er)
    per_ba = Percolation(mnet_ba)
    # per_bb = Percolation(mnet_bb)
    per_er.run(visualize=True, npoints=1000, colour='blue', log=True)
    per_ba.run(visualize=True, npoints=1000, colour='red', log=True)
    # res_bb = per_bb.run(visualize=True, npoints=3000, colour='orange')
    plt.show()


