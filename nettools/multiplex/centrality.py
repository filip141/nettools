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


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    from nettools.multiplex import MultiplexConstructor, MultiplexNetwork
    nodes_nm = 1500
    beta_param = {0: {0: 0.2, 1: 0.3}, 1: {0: 0.1, 1: 0.5}}
    rec_param = {0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 1.0}}
    ng = NetworkGenerator(nodes=nodes_nm)
    bb1 = ng.ba_network(m0=3)
    bb2 = ng.ba_network(m0=3)
    mc = MultiplexConstructor()
    mnet_bb = mc.construct(bb1, bb2)
    cm = CentralityMultiplex(mnet_bb, beta_param, rec_param)
    cm.ks_index()

