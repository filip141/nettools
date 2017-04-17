import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from nettools.utils import sample_from_dist


class Network(object):

    def __init__(self, network, weights_layer=None, n_type="ER", weighted=False):
        self.type = n_type
        self.network = network
        self.network_weighted = weights_layer
        self.weighted = weighted

    def network_degrees(self):
        """
        Method calculate degree for each node by summing adjacency matrix.

        :return: Numpy array with nodes degree.
        """
        return np.sum(self.network, axis=1)

    def degree_distribution(self):
        """
        Method calculate degree distribution using numpy bincount.

        :return: Degree distribution (Numpy array).
        """
        count_degs = np.bincount(self.network_degrees().astype(np.int32))
        args = np.argsort(count_degs)[::-1]
        deg_dst = np.sort(count_degs)[::-1]
        deg_dst = np.trim_zeros(deg_dst)
        args = args[:len(deg_dst)]
        return deg_dst.astype(np.float64) / np.sum(deg_dst), args

    def plot_loglog(self):
        """
        Plot log log degree distribution using degree_distribution method.
        """
        ax = plt.gca()
        deg_dst, x_axis = self.degree_distribution()
        ax.set_xlim([np.min(x_axis), np.max(x_axis)])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.scatter(x_axis, deg_dst, marker='o', color='r')
        plt.show()

    def plot_degree_dist(self):
        """
        Plot degree distribution using degree_distribution method.
        """
        deg_dst, x_axis = self.degree_distribution()
        plt.scatter(x_axis, deg_dst, marker='o', color='r')
        plt.show()

    def show(self):
        """
        Draw networkx graph.
        """
        nx_graph = nx.from_numpy_matrix(self.network)
        nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), node_size=20)
        plt.show()


class NetworkGenerator(object):

    def __init__(self, nodes):
        self.num_nodes = nodes

    def ba_network(self, m0=3):
        """
        Method for generating Barabassi Albert network,
        first m0 random nodes are initialized, next other zero degree nodes
        are connected to other nodes using preferential attachment.

        :param m0: How many nodes new node should choose to associate connection,
        :return: Network object, containing information about network (adjacency, type)
        """
        ba_net = np.zeros((self.num_nodes, self.num_nodes))

        # Initialize graph
        for m in range(m0):
            rand_conn_h = random.randint(0, self.num_nodes - 1)
            rand_conn_w = random.randint(0, self.num_nodes - 1)
            ba_net[rand_conn_h, rand_conn_w] = 1
            ba_net[rand_conn_w, rand_conn_h] = 1

        # Simulate growth process
        for nnode in range(self.num_nodes):
            # Omit earlier initialized nodes
            if np.sum(ba_net[nnode]) > 0:
                continue

            not_norm_dist = np.sum(ba_net, axis=1)
            degree_dist = not_norm_dist / np.sum(not_norm_dist)
            dist_samples = sample_from_dist(degree_dist, n_samples=m0)
            for rand_sample in dist_samples:
                ba_net[nnode, rand_sample] = 1
                ba_net[rand_sample, nnode] = 1
        np.fill_diagonal(ba_net, 0)
        return Network(ba_net, n_type="BA")

    def bb_network(self, m0=3, fitness=None):
        """
        Method for generating Barabassi Bianconi network, this model besides of
        preferential attachment and growth as BA model have also parameter called
        fitness, this fact facilitate younger nodes to overrun older ones.
        :param m0:          How many nodes new node should choose to associate connection,
        :param fitness:     Fitness parameter gives information about node importance, if
                            None will be set at random.
        :return: Network object, containing information about network (adjacency, type)
        """
        bb_net = np.zeros((self.num_nodes, self.num_nodes))

        # Initialize fitness if None
        if fitness is None:
            fitness = np.random.uniform(0, 1, size=(self.num_nodes,))

        # Initialize graph
        for m in range(m0):
            rand_conn_h = random.randint(0, self.num_nodes - 1)
            rand_conn_w = random.randint(0, self.num_nodes - 1)
            bb_net[rand_conn_h, rand_conn_w] = 1
            bb_net[rand_conn_w, rand_conn_h] = 1

        # Simulate growth process
        for nnode in range(self.num_nodes):
            # Omit earlier initialized nodes
            if np.sum(bb_net[nnode]) > 0:
                continue

            not_norm_dist = np.sum(bb_net, axis=1)
            degree_dist = (not_norm_dist / np.sum(not_norm_dist)) * fitness
            degree_dist = degree_dist / np.sum(degree_dist)
            dist_samples = sample_from_dist(degree_dist, n_samples=m0)
            for rand_sample in dist_samples:
                bb_net[nnode, rand_sample] = 1
                bb_net[rand_sample, nnode] = 1
        np.fill_diagonal(bb_net, 0)
        return Network(bb_net, n_type="BB")

    def er_network(self, p=0.5):
        """
        Method for generating Erdos-Renyi network. ER network is other name for random graph.
        First N n connected points are generated, next each edge is connected or not with
        probability p.
        :param p:   Random network probability, connect or not,
        :return: Network object, containing information about network (adjacency, type)
        """
        er_net = np.random.uniform(0, 1, size=(self.num_nodes, self.num_nodes))
        er_net *= np.tri(*er_net.shape)
        er_net += er_net.T
        result = er_net.copy()
        result[er_net < p] = 1
        result[er_net > p] = 0
        np.fill_diagonal(result, 0)
        return Network(result, n_type="ER")


if __name__ == '__main__':
    ng = NetworkGenerator(1000)
    net = ng.bb_network()
    net.plot_loglog()