import community
import numpy as np
import networkx as nx
from ..utils.netutils import NX_CENTRALITY


class CentralityMeasure(object):
    def __init__(self, graph, pymnet=True):
        # Load graph
        if pymnet:
            self.network_graph = nx.from_numpy_matrix(graph)
        else:
            self.network_graph = nx.Graph(graph._net)
        # Number of nodes
        self.network_nodes = self.network_graph.nodes()

    def network_cn(self, method):
        # Convert to networkx graph
        nx_mth = NX_CENTRALITY.get(method, None)

        # Check implementation
        if nx_mth is None:
            raise AttributeError("Not supported method for centrality measure.")

        # Get result
        if isinstance(nx_mth, str):
            meas_mth = getattr(nx, nx_mth)
            return meas_mth(self.network_graph)
        elif method == 'voterank':
            return self.voterank()
        else:
            return None

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
        :param f: VoteRank parameter used to decrease max spreader neighbors voting ability (float),
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
                    if len(adj_node[cent_node] - set(visited_snd)) == len(adj_node[cent_node]) \
                            and not cent_node in super_spreaders.keys():
                        super_spreaders[cent_node] = max_score
                        max_score -= 1
                        visited_snd.append(comm)
                        break
        return super_spreaders
