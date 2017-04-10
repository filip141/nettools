import pymnet
import logging
import numpy as np
import networkx as nx
import scipy.stats as stats
import multiprocessing as mp
import matplotlib.pyplot as plt
from scripts.utils.netutils import load_multinet_by_name
from scripts.monoplex.centrality import CentralityMeasure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.agg_onet = self.aggregate(self.network_weights_np)
        self.one_triad_clustering_pool(self.network_graph_np)
        # score, results = self.network_interdependence(network=self.network_graph_np[:, :, :15])
        # plt.bar([x for x in range(len(results))], results)
        # plt.show()
        # print(score)
        # degree = self.degree_distribution(self.network_graph_np)
        # pdc = self.degree_layer_dependence(self.network_graph_np, weight_network=self.network_weights_np)
        # plt.figure()
        # plt.imshow(np.repeat(np.repeat(pdc, 10, axis=0), 10, axis=1))
        # plt.show(True)
        # print()

        # cn = CentralityMeasure(self.network_graph_np[:, :, 0], from_numpy=True)
        # korr = self.kendal_corr(degree[:10])
        # res_kor = np.repeat(np.repeat(korr, 100, axis=0), 100, axis=1)
        # plt.figure()
        # plt.imshow(res_kor)
        # plt.show()
        # res_kor = np.repeat(res_kor, 100, axis=1)
        # o = np.sum(self.agg_net, axis=1)
        # pc = self.participation_coeff(self.network_graph_np, self.agg_net)
        # plt.figure()
        # plt.plot(np.sort(pc)[::-1], '.r')
        # plt.figure()
        # plt.scatter(pc, (o - np.mean(o)) / np.std(o))
        # plt.show()
        # plt.imshow(res_kor)
        # print()

    @staticmethod
    def one_triad_clustering(network):
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

    def one_triad_clustering_pool(self, network):
        output = mp.Queue()
        # Initialize processes
        processes = [mp.Process(target=self.worker_method,
                                args=(network, idx, output)) for idx in range(network.shape[0])]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = np.array([output.get() for p in processes])
        k_dist = np.sum(network, axis=1)
        k_dist = np.sum(k_dist * (k_dist - 1), axis=1)
        results = results / k_dist
        results = np.nan_to_num(results)
        # Normalize
        return results

    def worker_method(self, network, idx, output):
        layers_num = network.shape[-1]
        results = 0
        for row_l_idx in range(layers_num):
            for row_ul_idx in range(layers_num):
                for j in range(network.shape[0]):
                    for m in range(network.shape[0]):
                        if row_l_idx != row_ul_idx and idx != m:
                            if np.all([network[idx, j, row_l_idx],
                                       network[j, m, row_ul_idx], network[m, idx, row_l_idx]]):
                                results += network[idx, j, row_l_idx] * \
                                           network[j, m, row_ul_idx] * network[m, idx, row_l_idx]
        logging.info("Computed for i: {}".format(idx))
        output.put(results)

    @staticmethod
    def interdependence(network, layer):
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
        net_size = network.shape
        layer_scores = []
        for layer_num in range(net_size[-1]):
            layer_scores.append(InterMeasures.interdependence(network, layer_num)[0])
        return (1/float(net_size[-1])) * np.sum(layer_scores), layer_scores

    @staticmethod
    def degree_layer_dependence(network, weight_network=None):
        if weight_network is None:
            weight_network = network
        layers_num = network.shape[-1]
        result = np.zeros((layers_num, layers_num))
        for row_l_idx in range(layers_num):
            for row_ul_idx in range(layers_num):
                result[row_l_idx, row_ul_idx] = InterMeasures.degree_conditional(weight_network[:, :, row_l_idx],
                                                                                 network[:, :, row_ul_idx])
        return result

    @staticmethod
    def degree_conditional(ref_layer, test_layer):
        return np.sum(ref_layer * test_layer) / np.sum(ref_layer)

    @staticmethod
    def participation_coeff(network, agg_net):
        net_size = network.shape

        # Reshape aggregated network
        agg_rep = np.repeat(np.sum(agg_net, axis=1)[:, np.newaxis], net_size[-1], axis=1)

        # Compute coefficient
        d_net = (np.sum(network, axis=1) / agg_rep)**2
        d_net = np.nan_to_num(d_net)
        return (net_size[-1] / (net_size[-1] - 1)) * (1 - np.sum(d_net, axis=1))

    @staticmethod
    def kendal_corr(distribution):
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
        net_size = net.shape
        if len(net_size) == 2:
            return np.sum(net, axis=1)
        else:
            degree_mat = np.transpose(np.sum(net, axis=1), (1, 0))
            return degree_mat

    @staticmethod
    def aggregate(net):
        agg_net = np.sum(net, axis=2)
        return agg_net

if __name__ == '__main__':
    im = InterMeasures('london')