import logging
import numpy as np
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
        for rmnode in np.where(per_prob < (1 - probability))[0]:
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

    def run(self, visualize=False, npoints=100, log=False):
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
        x_p = x_p * np.mean(np.sum(self.network.network, axis=1))
        # Visualize
        if visualize:
            plt.plot(x_p, gcc_nodes)
            plt.hold(True)
            plt.plot(x_p[19:], running_mean(gcc_nodes, 20), 'r')
            plt.show()
        return x_p, gcc_nodes


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator, Network
    from nettools.multiplex import MultiplexConstructor
    nodes_nm = 100
    ng = NetworkGenerator(nodes=nodes_nm)
    # ba1 = ng.bb_network(m0=1)
    # ba2 = ng.bb_network(m0=1)
    er1 = ng.er_network(p=2.5 / float(nodes_nm))
    er2 = ng.er_network(p=2.5 / float(nodes_nm))
    mc = MultiplexConstructor()
    # bac = mc.rewire_hubs(ba1, rsteps=2000)
    mnet = mc.construct(er1, er2)
    per = Percolation(mnet)
    per.run(visualize=True, npoints=3000, log=True)


