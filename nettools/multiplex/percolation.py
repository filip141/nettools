import os
import logging
import numpy as np
import networkx as nx
import nettools.multiplex
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "..", "..", "data")
log_path = os.path.join(data_dir, "ctest_log.log")
logging.basicConfig(filename=log_path, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def running_mean(x, nnodes):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nnodes:] - cumsum[:-nnodes]) / nnodes


class Percolation(object):

    def __init__(self, network):
        # Convert to multiplex object if adjacency passed
        if isinstance(network, np.ndarray):
            self.network = nettools.multiplex.MultiplexNetwork(network)
        elif isinstance(network, nettools.multiplex.MultiplexNetwork):
            self.network = network
        else:
            raise ValueError("Network should be MultiplexNetwork object or numpy.ndarray")

    def remove_nodes(self, probability):
        new_net = nettools.multiplex.MultiplexNetwork(self.network.network.copy())
        per_prob = np.random.uniform(0, 1, size=(self.network.get_nodes_num(),))
        for rmnode in np.where(per_prob < 1 - probability)[0]:
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

    def run(self, visualize=False, n_times=1, npoints=100, log=False, smooth=True, colour='b'):
        avg_mat = np.zeros((n_times, npoints))
        for avg_idx in range(n_times):
            gcc_nodes = []
            logger.info("Percolation: Average step {}".format(avg_idx))
            for proba_nmul in range(0, npoints):
                probability = proba_nmul / float(npoints)
                gcc_nodes.append(self.remove_nodes(probability))
                if log:
                    logger.info("Percolation: Nodes in gcc: {}, probability: {}".format(gcc_nodes[-1], probability))
            # Normalize
            nodes_num = float(self.network.network.shape[0])
            gcc_nodes = np.array(gcc_nodes) / nodes_num
            x_p = np.arange(0, npoints) / float(npoints)
            avg_mat[avg_idx] = x_p
        x_avg = np.mean(avg_mat, axis=0)
        # Visualize
        if visualize:
            ax = plt.gca()
            ax.set_xlim([0, 1.0])
            ax.set_ylim([0, 1.0])
            if not smooth:
                plt.plot(x_avg, gcc_nodes, hold=True)
            plt.plot(x_avg[19:], running_mean(gcc_nodes, 20), colour, hold=True)
        return x_avg, gcc_nodes


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator, Network
    from nettools.multiplex import MultiplexConstructor
    nodes_nm = 3000
    ng = NetworkGenerator(nodes=nodes_nm)
    # ba1 = ng.bb_network(m0=1)
    ba1 = ng.ba_network(m0=2)
    # ba1 = Network(nx.to_numpy_matrix(nx.barabasi_albert_graph(nodes_nm, m=2)), n_type="BA")
    # er1 = ng.er_network(p=4.0 / float(nodes_nm))
    # er1 = Network(nx.to_numpy_matrix(nx.erdos_renyi_graph(nodes_nm, p=4.0 / float(nodes_nm - 1))), n_type='ER')
    er1 = ng.er_network(p=4.0 / float(nodes_nm - 1))
    er2 = ng.er_network(p=4.0 / float(nodes_nm - 1))
    mc = MultiplexConstructor()
    # er1 = ng.er_network(p=4.0 / float(nodes_nm))
    # ba1 = ng.ba_network(m0=2)
    # er1 = Network(nx.to_numpy_matrix(nx.erdos_renyi_graph(nodes_nm, p=4.0 / float(nodes_nm))), n_type="ER")
    # ba1 = Network(nx.to_numpy_matrix(nx.barabasi_albert_graph(nodes_nm, m=2)), n_type="BA")
    bb1 = ng.bb_network(m0=2)
    bb2 = ng.bb_network(m0=2)
    mnet_er = mc.construct(er1)
    mnet_ba = mc.construct(ba1)
    mnet_bb = mc.construct(bb1)
    mnet_erer = mc.construct(er1, er2)
    mnet_bbbb = mc.construct(bb1, bb2)
    per_er = Percolation(mnet_er)
    per_ba = Percolation(mnet_ba)
    per_bb = Percolation(mnet_bb)
    per_bbbb = Percolation(mnet_erer)
    per_erer = Percolation(mnet_bbbb)
    per_er.run(visualize=True, colour='blue', log=True, n_times=2, npoints=3000)
    per_ba.run(visualize=True, colour='red', log=True, n_times=2, npoints=3000)
    per_bb.run(visualize=True, npoints=3000, colour='orange', n_times=2)
    per_bbbb.run(visualize=True, npoints=3000, colour='green', n_times=2)
    per_erer.run(visualize=True, npoints=3000, colour='magenta', n_times=2)
    plt.show()


