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

    def network_cn(self, method, gamma=1, beta=1, ks_mth='k-shell'):
        if method == 'ks-index':
            return self.ks_index(ks_mth)
        elif method == 'multi_pagerank':
            return self.multi_pagerank(gamma=gamma, beta=beta)
        elif method == 'multi_pagerank_numpy':
            return self.multi_pagerank_numpy(gamma=gamma, beta=beta)
        else:
            raise AttributeError("{} method not implemented yet.".format(method))

    def ks_index(self, method="k-shell"):
        # Iterate over layers
        layer_weights = []
        inter_mat = np.zeros((self.layers_num, self.nodes_num))
        ks_mat = np.zeros((self.layers_num, self.nodes_num))
        agg_net = nettools.multiplex.InterMeasures.aggregate(self.network_numpy)
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
            zero_idxs = np.where(self.network_numpy[:, :, l_idx].T == 0)
            int_rpt[zero_idxs] = 0
            scale_layer = nettools.multiplex.InterMeasures.link_conditional(agg_net, self.network_numpy[:, :, l_idx])
            inter_mat[l_idx] = scale_layer * np.sum(int_rpt, axis=1)
            layer_weights.append(scale_layer)
        inter_scores = np.sum(inter_mat, axis=0)
        # Inter layer connection
        for l_idx in range(self.layers_num):
            for lr_idx in range(self.layers_num):
                if l_idx == lr_idx:
                    continue
                p_div = self.beta[l_idx][lr_idx] / self.mu[l_idx][l_idx]
                inter_scores += layer_weights[lr_idx] * p_div * ks_mat[lr_idx]
        ks_scores = inter_scores / np.max(inter_scores)
        ks_scores = dict(enumerate(ks_scores))
        return ks_scores

    def multi_pagerank_numpy(self, eps=1e-12, alpha=None, beta=1, gamma=1):
        n_nodes = self.network_numpy.shape[0]
        if alpha is None:
            alpha = [0.85 for _ in range(self.network_numpy.shape[-1])]
        if len(alpha) != self.network_numpy.shape[-1]:
            raise ValueError("Alpha should be defined for each layer.")
        # First pagerank iteration
        el_diff = 1.0
        out_degree = np.sum(self.network_numpy[:, :, 0], axis=1)
        max_out = np.maximum(1, out_degree)
        old_vec = (1 / float(n_nodes)) * np.ones((n_nodes,))
        while el_diff > eps:
            x = old_vec / max_out
            x = alpha[0] * np.dot(self.network_numpy[:, :, 0].T, x) + \
                (1 - alpha[0]) * (1 / float(n_nodes)) * np.ones((n_nodes,))
            x /= np.sum(x)
            el_diff = np.sum(np.abs(x - old_vec))
            old_vec = x

        # For multiplexes
        x /= np.sum(x)
        x_mat = [x]
        # Loop acros layers
        for l_idx in range(1, self.network_numpy.shape[-1]):
            el_diff = 1.0
            old_vec = (1 / float(n_nodes)) * np.ones((n_nodes,))
            g_val = np.dot(self.network_numpy[:, :, l_idx], x_mat[-1]**beta)
            g_val[g_val == 0] = 1.0
            while el_diff > eps:
                x_tmp = old_vec / g_val
                x_tmp_vec = alpha[l_idx] * np.dot(self.network_numpy[:, :, l_idx].T, x_tmp) * x_mat[-1]**beta
                x_fin = x_tmp_vec + (1 - alpha[l_idx]) * ((x_mat[-1]**gamma) / (n_nodes * np.mean(x_mat[-1]**gamma)))
                x_fin /= np.sum(x_fin)
                el_diff = np.sum(np.abs(x_fin - old_vec))
                old_vec = x_fin
            x_mat.append(x_fin)
        final_scores = x_mat[-1] / np.sum(x_mat[-1])
        final_res = dict(enumerate(final_scores))
        return final_res

    def multi_pagerank(self, eps=1e-12, alpha=None, beta=1, gamma=1):
        n_nodes = self.network_numpy.shape[0]
        if alpha is None:
            alpha = [0.85 for _ in range(self.network_numpy.shape[-1])]
        if len(alpha) != self.network_numpy.shape[-1]:
            raise ValueError("Alpha should be defined for each layer.")
        # First pagerank iteration
        el_diff = 1.0
        old_vec = (1 / float(n_nodes)) * np.ones((n_nodes,))
        # first pagerank iteration
        while el_diff > eps:
            cent_vec = np.zeros((n_nodes,))
            for n_idx in range(n_nodes):
                cent_acc = 0
                for nt_idx in range(n_nodes):
                    g_coef = np.sum(self.network_numpy[:, nt_idx, 0])
                    cent_acc += \
                        alpha[0] * self.network_numpy[n_idx, nt_idx, 0] * old_vec[nt_idx] / np.max([1.0, g_coef])
                cent_vec[n_idx] = cent_acc + (1 - alpha[0]) * 1.0 / float(n_nodes)
            cent_vec /= np.sum(cent_vec)
            el_diff = np.sum(np.abs(cent_vec - old_vec))
            old_vec = cent_vec
        x_mat = [cent_vec]
        # for other layers
        for l_idx in range(1, self.network_numpy.shape[-1]):
            el_diff = 1.0
            old_vec = (1 / float(n_nodes)) * np.ones((n_nodes,))
            while el_diff > eps:
                cent_vec = np.zeros((n_nodes,))
                for n_idx in range(n_nodes):
                    cent_acc = 0
                    for nt_idx in range(n_nodes):
                        # Calculate G matrix
                        g_div = 0
                        for r in range(n_nodes):
                            g_div += self.network_numpy[r, nt_idx, l_idx] * x_mat[-1][r]**beta
                        g_div = g_div if g_div != 0 else 1.0
                        cent_acc += alpha[l_idx] * (x_mat[-1][n_idx]**beta) * \
                                    self.network_numpy[n_idx, nt_idx, l_idx] * old_vec[nt_idx] / g_div
                    cent_vec[n_idx] = cent_acc + (1 - alpha[l_idx]) * \
                                                 ((x_mat[-1][nt_idx]**gamma) / (n_nodes * np.mean(x_mat[-1]**gamma)))
                cent_vec /= np.sum(cent_vec)
                el_diff = np.sum(np.abs(cent_vec - old_vec))
                old_vec = cent_vec
            x_mat.append(cent_vec)
        final_scores = x_mat[-1] / np.sum(x_mat[-1])
        final_res = dict(enumerate(final_scores))
        return final_res

if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    from nettools.multiplex import MultiplexConstructor, MultiplexNetwork
    nodes_nm = 30
    beta_param = {0: {0: 0.2, 1: 0.3}, 1: {0: 0.1, 1: 0.5}}
    rec_param = {0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 1.0}}
    ng = NetworkGenerator(nodes=nodes_nm)
    bb1 = ng.ba_network(m0=3)
    bb2 = ng.ba_network(m0=3)
    mc = MultiplexConstructor()
    mnet_bb = mc.construct(bb1, bb2, bb1)
    cm = CentralityMultiplex(mnet_bb, beta_param, rec_param)
    dr_1 = cm.multi_pagerank(eps=1e-15)
    dr_2 = cm.multi_pagerank_numpy(eps=1e-15)
    print(dr_1)
    print(dr_2)
    print(np.argsort(dr_1.values()))
    print(np.argsort(dr_2.values()))

