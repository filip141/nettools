import types
import pymnet
import logging
import copy_reg
import numpy as np
import networkx as nx
import scipy.stats as stats
import multiprocessing as mp
from nettools.utils.netutils import load_multinet_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class InterMeasures(object):

    def __init__(self, db_name='london', network_attr=None):
        # Load network
        if network_attr is None:
            network_prop = load_multinet_by_name(db_name)
            self.loaded_network = network_prop[0]
            self.network_graph_np = network_prop[1]
            self.network_weights_np = network_prop[2]
            self.id2node, self.node2id = network_prop[3]
            self.layers_attr = network_prop[4]
        else:
            self.loaded_network = network_attr['loaded_network']
            self.network_graph_np = network_attr['network_graph_np']
            self.network_weights_np = network_attr['network_weights_np']
            self.id2node, self.node2id = network_attr['mapping']
            self.layers_attr = network_attr['layers_attr']

        # Aggregate network
        self.agg_net = self.aggregate(self.network_graph_np)
        self.agg_onet = np.sum(self.network_weights_np, axis=1)

    @staticmethod
    def one_triad_clustering(network):
        """
        Clustering coefficient non multiprocess method.
        Very, very, very slow.
        In future should be replaced for method using cython or cuda.

        :param network: Network adjacency matrix,
        :return: One triad clustering coefficient.
        """
        layers_num = network.shape[-1]
        result = np.zeros(network.shape[:-1])
        # Compute for each network layer
        for row_l_idx in range(layers_num):
            for row_ul_idx in range(layers_num):
                for i in range(network.shape[0]):
                    for j in range(network.shape[0]):
                        for m in range(network.shape[0]):
                            if row_l_idx != row_ul_idx and i != m:
                                if np.all([network[i, j, row_l_idx],
                                           network[j, m, row_ul_idx], network[m, i, row_l_idx]]):
                                    result[0, i] += network[i, j, row_l_idx] * \
                                                    network[j, m, row_ul_idx] * network[m, i, row_l_idx]
            logging.info("Computed for layer: {}".format(row_l_idx))
        k_dist = np.sum(network, axis=1)
        k_dist = np.sum(k_dist * (k_dist - 1), axis=1)
        result = result / k_dist
        result = np.nan_to_num(result)
        return result

    def one_triad_clustering_pool(self):
        """
        Clustering coefficient is very computational efficient method,
        for that reason method uses multiprocess library for computation.
        Using this method can improve performance but only when reasonably small
        number of nodes is in network.

        :return: One triad clustering coefficient.
        """
        number_of_processors = mp.cpu_count()
        pool = mp.Pool(number_of_processors)
        results = pool.map(self.worker_method, range(self.network_graph_np.shape[0]))
        results = np.array([x[0] for x in sorted(results, key=lambda tup: tup[1])])
        # Normalize
        k_deg = np.sum(self.network_graph_np, axis=1)
        results = results / np.sum(k_deg * (k_deg - 1), axis=1)
        return results

    # noinspection PyMethodMayBeStatic
    def worker_method(self, idx):
        """
        Worker method for multiprocess pool.

        :param idx: clustering coefficient index/ node index [i]
        """
        layers_num = self.network_graph_np.shape[-1]
        results = 0
        for row_l_idx in range(layers_num):
            for row_ul_idx in range(layers_num):
                for j in range(self.network_graph_np.shape[0]):
                    for m in range(self.network_graph_np.shape[0]):
                        if row_l_idx != row_ul_idx and idx != m:
                            results += self.network_graph_np[idx, j, row_l_idx] * \
                                       self.network_graph_np[j, m, row_ul_idx] * \
                                       self.network_graph_np[m, idx, row_l_idx]
        logging.info("Computed for i: {}".format(idx))
        return results, idx

    @staticmethod
    def interdependence(network, layer):
        """
        Interdependence is measure of reachability or layer importance in case
        of measuring path lengths between nodes.

        :param network: Network adjacency matrix,
        :param layer: for which interdependence will be measured,
        :return: Interdependence for layer, Path proportion between nodes.
        """
        # Aggregate network
        agg_net = InterMeasures.aggregate(network)
        agg_nx = nx.from_numpy_matrix(agg_net)
        net_layer_nx = nx.from_numpy_matrix(network[:, :, layer])

        # Get network size
        net_size = agg_net.shape
        results = np.zeros(net_size)
        for i_node in range(net_size[0]):
            for j_node in range(net_size[1]):
                # In case of no path exception
                try:
                    s_num_agg = len(list(nx.all_shortest_paths(agg_nx, source=i_node, target=j_node)))
                    s_num_layer = len(list(nx.all_shortest_paths(net_layer_nx, source=i_node, target=j_node)))
                    results[i_node, j_node] = s_num_layer / float(s_num_agg)
                except nx.NetworkXNoPath:
                    results[i_node, j_node] = 0
        return np.sum(results) / (net_size[0] * net_size[0]), results

    @staticmethod
    def network_interdependence(network):
        """
        Method for measuring interdependence for whole network,
        Interdependence is measure of reachability or layer importance in case
        of measuring path lengths between nodes. In this method interdependence is measured for
        each layer and next mean value is returned for network as a whole.

        :param network: Network adjacency matrix,
        :return: Interdependence for network (float), Interdependence's for each layer (list)
        """
        net_size = network.shape
        layer_scores = []
        for layer_num in range(net_size[-1]):
            layer_scores.append(InterMeasures.interdependence(network, layer_num)[0])
        return (1/float(net_size[-1])) * np.sum(layer_scores), layer_scores

    @staticmethod
    def link_layer_dependence(network, weight_network=None):
        """
        Method generates matrix with [layer x layer] elements,
        each element contains probability of finding link at test layer
        given reference layer.

        :param network: Network adjacency matrix,
        :param weight_network: if weighted network should be used as reference,
        :return: Probability of finding a link on test layer given reference between each layer.
        """
        if weight_network is None:
            weight_network = network
        layers_num = network.shape[-1]
        result = np.zeros((layers_num, layers_num))
        for row_l_idx in range(layers_num):
            for row_ul_idx in range(layers_num):
                result[row_l_idx, row_ul_idx] = InterMeasures.link_conditional(network[:, :, row_l_idx],
                                                                               weight_network[:, :, row_ul_idx])
        return result

    @staticmethod
    def link_conditional(ref_layer, test_layer):
        """
        Probability of finding a link at layer ref_layer
        given the presence of an edge between the same nodes at
        layer test_layer.

        :param ref_layer: reference layer,
        :param test_layer: test layer,
        :return: Probability of finding a link on test layer given reference.
        """
        return np.sum(ref_layer * test_layer) / np.sum(ref_layer)

    @staticmethod
    def participation_coeff(network):
        """
        Participation coefficient:
        "Metrics for the analysis of multiplex networks" Battiston et. al.
        Measure introduced for measuring participation in each layer,
        P_i takes value between [0, 1] and measures whether the links of node i
        are uniformly distributed among the M layers, or are instead primarily
        concentrated in just one or a few layers. Namely, the coefficient P_i is equal
         to 0 when all the edges of i lie in one layer, while Pi = 1 only when node i has exactly
         the same number of edges on each of the M layers

        :param network: Network adjacency matrix,
        :return: Participation coefficient value.
        """
        net_size = network.shape

        # Reshape overlap network
        overlap_net = np.sum(network, axis=1)
        over_rep = np.repeat(np.sum(overlap_net, axis=1)[:, np.newaxis], net_size[-1], axis=1)

        # Compute coefficient
        d_net = (np.sum(network, axis=1) / over_rep)**2
        d_net = np.nan_to_num(d_net)
        return (net_size[-1] / (net_size[-1] - 1)) * (1 - np.sum(d_net, axis=1))

    @staticmethod
    def entropy_coeff(network):
        """
        Participation coefficient:
        "Metrics for the analysis of multiplex networks" Battiston et. al.
        Entropy measure introduced for measuring participation in each layer,
        Entropy is 0 when node participate only in one layer, Entropy attain maximum
        when node degree is equal distributed between layers.

        :param network: Network adjacency matrix,
        :return: Entropy coefficient value.
        """
        net_size = network.shape

        # Reshape overlap network
        overlap_net = np.sum(network, axis=1)
        over_rep = np.repeat(np.sum(overlap_net, axis=1)[:, np.newaxis], net_size[-1], axis=1)

        # Compute entropy
        e_net = (np.sum(network, axis=1) / over_rep) * np.log(np.sum(network, axis=1) / over_rep)
        return -np.sum(e_net, axis=1)

    @staticmethod
    def kendal_corr(distribution):
        """
        Kendal correlation for network degree distribution. Method use degree distribution
        passed as input ands compute Kendal correlation between layers.
        Numpy array is returned in format [layer x layer]

        :param distribution: degree distribution for each layer,
        :return: Kendal correlation matrix between layers.
        """
        dist_size = distribution.shape[0]
        result = np.zeros((dist_size, dist_size))
        for row_c_idx in range(dist_size):
            for row_uc_idx in range(dist_size):
                tau, p_value = stats.kendalltau(
                    distribution[row_c_idx],
                    distribution[row_uc_idx]
                )
                result[row_c_idx, row_uc_idx] = tau
        return result

    @staticmethod
    def degree_distribution(net):
        """
        Method compute degree distribution from adjacency matrix.
        Degree distribution is computed for each layer separately.

        :param net: Adjacency matrix - numpy array [nodes x nodes x layers],
        :return: Degree distribution given layer.
        """
        net_size = net.shape
        if len(net_size) == 2:
            return np.sum(net, axis=1)
        else:
            degree_mat = np.transpose(np.sum(net, axis=1), (1, 0))
            return degree_mat

    @staticmethod
    def degree_conditional(ref_layer, test_layer):
        """
        Probability of finding same degree at layer ref_layer
        given the presence node with examined degree at
        layer test_layer.

        :param ref_layer: reference layer,
        :param test_layer: test layer,
        :return: Probability of finding a link on test layer given reference.
        """
        deg_dist_ref = InterMeasures.degree_distribution(ref_layer)
        deg_dist_test = InterMeasures.degree_distribution(test_layer)
        equal_degrees = np.sum(deg_dist_ref == deg_dist_test)
        return equal_degrees / float(deg_dist_ref.shape[0])

    @staticmethod
    def aggregate(net):
        """
        Method for aggregating network, method accepts network on input and
        return aggregated network. Input network should be in image style
        number of node x number of nodes x layers.

        :param net: Adjacency matrix - numpy array [nodes x nodes x layers]
        :return: Aggregated network [nodes x nodes]
        """
        agg_net = np.sum(net, axis=2)
        agg_net[agg_net > 0] = 1
        return agg_net

    @staticmethod
    def overlap(net):
        """
        Method for computing overlap network, method accepts network on input and
        return overlapped network. Input network should be in image style
        number of node x number of nodes x layers.

        :param net: Adjacency matrix - numpy array [nodes x nodes x layers]
        :return: Overlapped network [nodes x nodes]
        """
        over_net = np.sum(net, axis=2)
        return over_net

    def get_network_adjacency(self, weighted=False):
        """
        Method return adjacency matrix for analyzed network

        :param weighted: If weighted or not (bool)
        :return: Adjacency matrix
        """
        if weighted:
            return self.network_weights_np
        else:
            return self.network_graph_np

    def get_network_pymnet(self):
        """
        Method return pymnet network
        """
        return self.loaded_network

    def get_network_info(self):
        """
        Method return mapping arrays and information about layers
        """
        return self.id2node, self.node2id, self.layers_attr


if __name__ == '__main__':
    # im = InterMeasures('london')
    from nettools.monoplex import NetworkGenerator
    from nettools.multiplex import MultiplexConstructor
    avg_deg = 10.0
    network_size = 500
    ng = NetworkGenerator(network_size)
    network_ba_1 = ng.ba_network(m0=int(avg_deg / 2))
    network_ba_2 = ng.ba_network(m0=int(avg_deg / 2))
    network_er_1 = ng.er_network(p=(avg_deg / network_size))
    network_er_2 = ng.er_network(p=(avg_deg / network_size))
    network_bb_1 = ng.bb_network(m0=int(avg_deg / 2))
    mc = MultiplexConstructor()
    network_corr_ba_1 = mc.rewire_hubs(network_ba_1, rsteps=5000)
    multi_ba_ba_corr_1 = mc.construct(network_ba_1, network_corr_ba_1)
    pc_baba_corr_1 = InterMeasures.participation_coeff(multi_ba_ba_corr_1.network)
    print()
