import community
import numpy as np
import networkx as nx
import matplotlib
import nettools.utils
import nettools.monoplex

class CentralityMeasure(object):
    def __init__(self, graph, pymnet=False):
        # Load graph
        self.np_matrix = None
        if not pymnet:
            self.np_matrix = graph
            self.network_graph = nx.from_numpy_matrix(graph, create_using=nx.MultiDiGraph())
        elif isinstance(graph, nettools.monoplex.Network):
            self.network_graph = nx.from_numpy_matrix(graph.network, create_using=nx.MultiDiGraph())
        else:
            self.network_graph = nx.Graph(graph._net)
        # Number of nodes
        self.network_nodes = self.network_graph.nodes()

    def network_cn(self, method):
        # Convert to networkx graph
        nx_mth = nettools.utils.NX_CENTRALITY.get(method, None)

        # Check implementation
        if nx_mth is None:
            raise AttributeError("Not supported method for centrality measure.")

        # Get result
        if isinstance(nx_mth, str):
            meas_mth = getattr(nx, nx_mth)
            return meas_mth(self.network_graph)
        elif method == 'voterank':
            return self.voterank()
        elif method == 'supernode':
            return self.supernode_rank()
        elif method == 'k-shell':
            return self.kshell()
        else:
            return None

    @staticmethod
    def score_remove(k_net, score, sc_buff=None, crust=0, no_crust=False):
        if sc_buff is None:
            sc_buff = {}
        net_deg = np.sum(k_net, axis=1)
        zeros_idx = set(np.where(net_deg <= int(score))[0]) - set(np.where(net_deg == 0)[0])
        if not zeros_idx:
            return k_net, sc_buff
        for node in zeros_idx:
            k_net[node, :] = 0
            k_net[:, node] = 0
            sc_buff[node] = score + crust
        if not no_crust:
            crust += 0.1
        return CentralityMeasure.score_remove(k_net, score, sc_buff=sc_buff, crust=crust, no_crust=no_crust)

    def kshell(self, no_crust=False):
        """
        K-Shell algorithm is implemented using 
        http://www.ifr.ac.uk/netsci08/Download/CT26_Toro_network/CT265_Kitsak.pdf
        as reference.

        :return: K-shell scores (dict)
        """
        node_scores = {}
        k_shell_score = 1
        if self.np_matrix is None:
            net_shell = nx.to_numpy_matrix(self.network_graph)
        else:
            net_shell = self.np_matrix
        while True:
            net_shell, node_scores = self.score_remove(net_shell, k_shell_score, sc_buff=node_scores,
                                                       no_crust=no_crust)
            k_shell_score += 1
            if np.sum(net_shell) == 0:
                # Prevent not connected nodes
                for z_node in (set(range(net_shell.shape[0])) - set(node_scores.keys())):
                    node_scores[z_node] = 0
                return node_scores

    def voterank(self, k=None, f=None):
        """
        VoteRank algorithm described by Zhang et. al in Identifying a set of influential
        spreaders in complex networks paper.
        Algorithm can be used to find top-k influential spreaders in monoplex network

        :param k: Number of influential spreaders (int),
        :param f: VoteRank parameter used to decrease max spreader neighbors voting ability (float),
        :return: Node votes (dict)
        """

        top_r = {}
        v_ab = nx.to_numpy_matrix(self.network_graph)

        # Set k
        if k is None:
            k = self.network_graph.number_of_nodes()

        # Calculate f as 1/<k>
        if f is None:
            s = sum(dict(self.network_graph.degree()).values())
            f = 1 / (float(s) / float(self.network_graph.number_of_nodes()))

        # Iterate to find best nodes
        for _ in range(0, k):
            score = np.sum(v_ab, axis=1)
            max_node = np.argmax(score)
            max_score = np.max(score)
            v_ab[:, np.nonzero(v_ab[max_node])[1]] -= f
            v_ab[v_ab < 0] = 0
            v_ab[max_node, :] = 0
            top_r[self.network_nodes[max_node]] = max_score
        return top_r

    def supernode_rank(self, k=None):
        """
        A Novel Top-k Strategy for Influence Maximization in Complex Networks with Community Structure
        Jia-Lin He et. al. algorithm using network communities to find best spreaders.
        Each super spreader should be found in different community and can't have link to other community.
        Kitsak et. al. shown that using not connected spreaders gives better results.

        :param k: Number of influential spreaders (int),
        :return: Node votes (dict)
        """
        part_dict = {}
        adj_node = {}
        super_spreaders = {}
        # Compute the best partition
        parts = community.best_partition(self.network_graph)

        # Set k
        if k is None:
            k = self.network_graph.number_of_nodes()
        max_score = k

        # Create supernodes
        for node, comm in parts.items():
            ret_dict = part_dict.get(comm, None)
            if ret_dict is None:
                part_dict[comm] = {}
            # Check node adjacency
            adj_node[node] = set([parts[nd] for nd in self.network_graph[node].keys()])
            part_dict[comm][node] = self.network_graph[node]

        # Find centrality for each supernode
        for _ in range(0, k):
            visited_snd = []
            for comm, val in part_dict.items():
                comm_net = nx.Graph(val)
                dc = nx.degree_centrality(comm_net)
                # Iterate over centrality nodes
                for cent_node in sorted(dc, key=dc.get)[::-1]:
                    if len(adj_node[cent_node] - set(visited_snd)) == len(adj_node[cent_node]):
                        if not cent_node in super_spreaders.keys():
                            super_spreaders[cent_node] = max_score
                            max_score -= 1
                            visited_snd.append(comm)
                            break
        return super_spreaders


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    ng = NetworkGenerator(nodes=500)
    net = ng.er_network(p=10.0 / 500.0)
    cn = CentralityMeasure(net.network, pymnet=False)
    cent = cn.network_cn("degree")
