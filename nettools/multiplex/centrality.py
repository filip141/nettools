import numpy as np
import nettools.monoplex
import nettools.multiplex


class CentralityMultiplex(object):

    def __init__(self, network, beta, mu):
        if not isinstance(network, nettools.multiplex.MultiplexNetwork):
            raise ValueError("Input network should be MultiplexNetwork object")
        # Epidemic params
        self.beta = beta
        self.mu = mu
        # Set network and its parameters
        self.network_numpy = network.network
        self.nodes_num = network.get_nodes_num()
        self.layers_num = network.get_layers_num()

    def ks_index(self, method="voterank"):
        # Iterate over layers
        inter_mat = np.zeros((self.layers_num, self.nodes_num))
        ks_mat = np.zeros((self.layers_num, self.nodes_num))
        # Intralayer connection
        for l_idx in range(self.layers_num):
            cn = nettools.monoplex.CentralityMeasure(self.network_numpy[:, :, l_idx])
            cnt_scores = cn.network_cn(method)
            if method == 'hits':
                cnt_scores = cnt_scores[1]
            cnt_arr_tmp = np.array([x_it[1] for x_it in sorted(cnt_scores.items(), key=cnt_scores.get)])
            cnt_arr = np.zeros((self.nodes_num,))
            cnt_arr[:cnt_arr_tmp.shape[0]] = cnt_arr_tmp
            ks_mat[l_idx] = cnt_arr
            # Get intralayer neighbors
            intra_layer = (self.beta[l_idx][l_idx] / self.mu[l_idx][l_idx]) * cnt_arr.astype(np.float64)
            int_rpt = np.repeat(intra_layer[np.newaxis, :], self.nodes_num, axis=0)
            zero_idxs = np.where(self.network_numpy[:, :, l_idx] == 0)
            int_rpt[zero_idxs] = 0
            inter_mat[l_idx] = np.sum(int_rpt, axis=1)
        inter_scores = np.sum(inter_mat, axis=0)
        # Inter layer connection
        for l_idx in range(self.layers_num):
            for lr_idx in range(self.layers_num):
                if l_idx == lr_idx:
                    continue
                p_div = self.beta[l_idx][lr_idx] / self.mu[l_idx][l_idx]
                inter_scores += p_div * ks_mat[lr_idx]
        ks_scores = inter_scores / np.max(inter_scores)
        ks_scores = dict(enumerate(ks_scores))
        return ks_scores

    def multi_pagerank(self, eps=1e-11, alpha=None, beta=1, gamma=1):
        n_nodes = self.network_numpy.shape[0]
        if alpha is None:
            alpha = [0.85 for _ in range(self.network_numpy.shape[-1])]
        if len(alpha) != self.network_numpy.shape[-1]:
            raise ValueError("Alpha should be defined for each layer.")
        # First pagerank iteration
        el_diff = 1.0
        out_degree = np.sum(self.network_numpy[:, :, 0], axis=1)
        max_out = np.maximum(1, out_degree)
        old_vec = np.array([1 / float(n_nodes) for xr in range(n_nodes)])
        while el_diff > eps:
            x = old_vec / max_out
            x = alpha[0] * np.dot(self.network_numpy[:, :, 0], x) + \
                (1 - alpha[0]) * (1 / float(n_nodes)) * np.ones((n_nodes,))
            el_diff = np.sum(np.abs(x - old_vec))
            old_vec = x

        # For multiplexes
        x_mat = [x]
        # Loop acros layers
        for l_idx in range(1, self.network_numpy.shape[-1]):
            el_diff = 1.0
            old_vec = np.array([1 / float(n_nodes) for xr in range(n_nodes)])
            g_val = np.dot(self.network_numpy[:, :, l_idx], x_mat[-1]**beta)
            g_val[g_val == 0] = 1.0
            while el_diff > eps:
                x_tmp = old_vec / g_val
                x_tmp_vec = alpha[l_idx] * np.dot(self.network_numpy[:, :, l_idx], x_tmp) * x_mat[-1]**beta
                x_fin = x_tmp_vec + (1 - alpha[l_idx]) * ((x_mat[-1]**gamma) / (n_nodes * np.mean(x_mat[-1]**gamma)))
                el_diff = np.sum(np.abs(x_fin - old_vec))
                old_vec = x_fin
            x_mat.append(x_fin)
        final_scores = x_mat[-1] / np.max(x_mat[-1])
        final_res = dict(enumerate(final_scores))
        return final_res


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    from nettools.multiplex import MultiplexConstructor, MultiplexNetwork
    nodes_nm = 20
    beta_param = {0: {0: 0.2, 1: 0.3}, 1: {0: 0.1, 1: 0.5}}
    rec_param = {0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 1.0}}
    ng = NetworkGenerator(nodes=nodes_nm)
    bb1 = ng.ba_network(m0=3)
    bb2 = ng.ba_network(m0=3)
    mc = MultiplexConstructor()
    mnet_bb = mc.construct(bb1, bb2, bb1)
    cm = CentralityMultiplex(mnet_bb, beta_param, rec_param)
    print(cm.multi_pagerank())

