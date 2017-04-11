import random
import numpy as np
from nettools.utils import sample_from_dist
from nettools.multiplex import InterMeasures
from nettools.monoplex import NetworkGenerator, Network


class MultiplexNetwork(object):

    def __init__(self, network, weights_layer=None, n_types=None, weighted=False):
        if n_types is None:
            self.type = ["ER", "ER"]
        self.network = network
        self.network_weighted = weights_layer
        self.weighted = weighted


class MultiplexConstructor(object):

    def __init__(self):
        pass

    @staticmethod
    def construct(*args):
        # Iterate over arguments
        n_layers = len(args)
        net_types = []
        n_nodes = args[0].network.shape[0]
        multiplex_network = np.zeros((n_nodes, n_nodes, n_layers))
        multiplex_network_weight = np.zeros((n_nodes, n_nodes, n_layers))
        for idx, layer in enumerate(args):
            if not isinstance(layer, Network):
                raise AttributeError("Layers should be Network objects.")
            net_types.append(layer.type)
            multiplex_network[:, :, idx] = layer.network
            multiplex_network_weight[:, :, idx] = layer.network_weighted
        return MultiplexNetwork(multiplex_network, weights_layer=multiplex_network_weight,
                                n_types=net_types, weighted=args[0].weighted)

    @staticmethod
    def rewire_hubs(network_obj, rsteps=20):
        network = network_obj.network.copy()
        network_w = None
        if network_obj.weighted:
            network_w = network_obj.network_weighted.copy()
        not_norm_dist = np.sum(network, axis=1)
        degree_dist = not_norm_dist / np.sum(not_norm_dist)
        for rs in range(rsteps):
            dist_smpl = sample_from_dist(degree_dist, n_samples=2)
            # Take random connection from first hub
            nz_0 = np.nonzero(network[dist_smpl[0]])
            # noinspection PyUnresolvedReferences
            nz_rand_0 = random.randint(0, nz_0[0].shape[0] - 1)
            # noinspection PyUnresolvedReferences
            elem_0 = nz_0[0][nz_rand_0]
            # Take random connection from second hub
            nz_1 = np.nonzero(network[dist_smpl[1]])
            # noinspection PyUnresolvedReferences
            nz_rand_1 = random.randint(0, nz_1[0].shape[0] - 1)
            # noinspection PyUnresolvedReferences
            elem_1 = nz_1[0][nz_rand_1]
            # Rewire
            network[dist_smpl[0], elem_0] = 0
            network[dist_smpl[1], elem_0] = 1

            network[dist_smpl[1], elem_1] = 0
            network[dist_smpl[0], elem_1] = 1

            # For weighted
            if network_obj.weighted:
                network_w[dist_smpl[1], elem_0] = \
                    network[dist_smpl[1], elem_1]
                network_w[dist_smpl[0], elem_1] = \
                    network[dist_smpl[0], elem_0]
                network_w[dist_smpl[0], elem_0] = 0
                network_w[dist_smpl[1], elem_1] = 0
        return Network(network, weights_layer=network_w,
                       n_type=network_obj.type, weighted=network_obj.weighted)


if __name__ == '__main__':
    ng = NetworkGenerator(nodes=100)
    ba1 = ng.ba_network()
    ba2 = ng.ba_network()
    ba3 = ng.ba_network()
    mc = MultiplexConstructor()
    test_net = mc.rewire_hubs(ba1, rsteps=100).network
    print(InterMeasures.degree_conditional(ba1.network, test_net))
    # mc.construct(ba1, mc.rewire_hubs(ba2), mc.rewire_hubs(ba3))
