import random
import numpy as np
# from pymnet import *
import networkx as nx

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
import nettools.multiplex


def visualize_epidemic(vnetwork, net_attrs):
    """
        Visualize network using Multilayer Network Library function *draw*.

        :param vnetwork: Visualized network
        :param net_attrs: Network attributes
    """
    # Colour nodes
    colours = {}
    for node, attrs in net_attrs.items():
        if attrs[1]['i']:
            n_colour = 'r'
        elif attrs[1]['s']:
            n_colour = 'g'
        else:
            n_colour = 'b'
        colours[attrs[0]] = n_colour
    draw(vnetwork, nodeColorDict=colours, show=False, layout="circular")
    plt.pause(2)
    return None


def visualize_epidemic_image_style(vnetwork, net_states, layers=None, labels=False, pause=2):
    """
        Visualize network in image style, [nodes x nodes x layers]

        :param pause: How many seconds wait for next plot
        :param vnetwork: Visualized network
        :param net_states: Network states
        :param layers: which layers should be plot
        :param labels: show node labels or not
    """
    if layers is None:
        layers = [0, ]
    # Iterate over layers
    for layer in layers:
        node_colors = []
        for node in range(net_states.shape[0]):
            if net_states[node, layer] == 1:
                n_colour = 'r'
            elif net_states[node, layer] == 0:
                n_colour = 'g'
            else:
                n_colour = 'b'
            node_colors.append(n_colour)
        plt.figure(layer)
        nx_graph = nx.from_numpy_matrix(vnetwork[:, :, layer])
        if labels:
            nx.draw_networkx(nx_graph, pos=nx.spring_layout(nx_graph), node_size=20, node_color=node_colors)
        else:
            nx.draw(nx_graph, pos=nx.spring_layout(nx_graph), node_size=20, node_color=node_colors)
        plt.hold(False)
    plt.pause(pause)
    return None


class EpidemicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def infect_node(self, node):
        pass

    @abstractmethod
    def recover_node(self, node):
        pass

    @abstractmethod
    def one_epoch(self):
        pass


class SIRModel(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4):
        EpidemicModel.__init__(self)
        # If seed node is None take random
        if seed_nodes is None:
            seed_nodes = [random.choice(network.nodes())]
        # Process properties
        self.mu = mu
        self.beta = beta
        self.states = ('s', "i")

        # Define class network
        self.network = network
        self.network_size = network.number_of_nodes()

        # All nodes are susceptible on beginning
        nodes_def_s = dict([(num, 1) for num in range(0, self.network_size)])
        nodes_def_i = dict([(num, 0) for num in range(0, self.network_size)])
        nodes_def_r = dict([(num, 0) for num in range(0, self.network_size)])
        nx.set_node_attributes(self.network, "s", nodes_def_s)
        nx.set_node_attributes(self.network, "i", nodes_def_i)
        nx.set_node_attributes(self.network, "r", nodes_def_r)

        # Infect seed nodes
        for seed in seed_nodes:
            self.infect_node(seed)

    def infect_node(self, node):
        self.network.node[node]['i'] = 1
        self.network.node[node]['s'] = 0

    def recover_node(self, node):
        self.network.node[node]['r'] = 1
        self.network.node[node]['i'] = 0

    def one_epoch(self):
        # Iterate over infected
        infected_attr = nx.get_node_attributes(self.network, 'i')
        for node, state in infected_attr.items():
            if state:
                # Infect neighbours
                nb_nodes = self.network.neighbors(node)
                for nb_nd in nb_nodes:
                    if not (self.network.node[nb_nd]['r'] or self.network.node[nb_nd]['i']):
                        dc_spread = random.uniform(0, 1)
                        if self.beta > dc_spread:
                            self.infect_node(nb_nd)

                # Recovery
                rc_spread = random.uniform(0, 1)
                if self.mu > rc_spread:
                    self.recover_node(node)

    def get(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        infected_attr = nx.get_node_attributes(self.network, state)
        return [n for n, s in infected_attr.items() if s]

    def get_num(self, state):
        return len(self.get(state))

    def run(self, epochs=200):
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            # Colour nodes
            colours = []
            new_nodes = self.network.nodes(data=True)
            for node, attrs in new_nodes:
                if attrs['i']:
                    n_colour = 'red'
                elif attrs['s']:
                    n_colour = 'green'
                else:
                    n_colour = 'blue'
                colours.append(n_colour)
            nx.draw(self.network, node_color=colours, node_size=25)
            plt.hold(False)
            plt.pause(0.5)

    def plot_epidemic(self, epochs=200):
        s, i, r = [], [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            r.append(self.get_num(state='r'))

        plt.figure()
        plt.hold(True)
        plt.plot(s, color='g')
        plt.plot(i, color='r')
        plt.plot(r, color='b')
        plt.show()


class SISModel(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4):
        EpidemicModel.__init__(self)
        # If seed node is None take random
        if seed_nodes is None:
            seed_nodes = [random.choice(network.nodes())]
        # Process properties
        self.mu = mu
        self.beta = beta
        self.states = ('s', "i")

        # Define class network
        self.network = network
        self.network_size = network.number_of_nodes()

        # All nodes are susceptible on beginning
        nodes_def_s = dict([(num, 1) for num in range(0, self.network_size)])
        nodes_def_i = dict([(num, 0) for num in range(0, self.network_size)])
        nx.set_node_attributes(self.network, "s", nodes_def_s)
        nx.set_node_attributes(self.network, "i", nodes_def_i)

        # Infect seed nodes
        for seed in seed_nodes:
            self.infect_node(seed)

    def infect_node(self, node):
        self.network.node[node]['i'] = 1
        self.network.node[node]['s'] = 0

    def recover_node(self, node):
        self.network.node[node]['i'] = 0
        self.network.node[node]['s'] = 1

    def one_epoch(self):
        # Iterate over infected
        infected_attr = nx.get_node_attributes(self.network, 'i')
        for node, state in infected_attr.items():
            if state:
                # Infect neighbours
                nb_nodes = self.network.neighbors(node)
                for nb_nd in nb_nodes:
                    if not self.network.node[nb_nd]['i']:
                        dc_spread = random.uniform(0, 1)
                        if self.beta > dc_spread:
                            self.infect_node(nb_nd)

                # Recovery
                rc_spread = random.uniform(0, 1)
                if self.mu > rc_spread:
                    self.recover_node(node)

    def get(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        infected_attr = nx.get_node_attributes(self.network, state)
        return [n for n, s in infected_attr.items() if s]

    def get_num(self, state):
        return len(self.get(state))

    def run(self, epochs=200):
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            # Colour nodes
            colours = []
            new_nodes = self.network.nodes(data=True)
            for node, attrs in new_nodes:
                if attrs['s']:
                    n_colour = 'green'
                else:
                    n_colour = 'red'
                colours.append(n_colour)
            nx.draw(self.network, node_color=colours, node_size=25,
                    layout=nx.spring_layout(self.network))
            plt.hold(False)
            plt.pause(0.5)

    def plot_epidemic(self, epochs=200):
        s, i, r = [], [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            r.append(self.get_num(state='r'))

        plt.figure()
        plt.hold(True)
        plt.plot(s, color='g')
        plt.plot(i, color='r')
        plt.plot(r, color='b')
        plt.show()


class SIRMultilayerPymnet(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4, inter_beta=0.9, inter_rec=0.3):
        super(SIRMultilayerPymnet, self).__init__()
        # If seed node is None take random
        network_adj = network.get_supra_adjacency_matrix()
        if seed_nodes is None:
            seed_nodes = [random.choice(network_adj[1])]
        # Process properties
        self.mu = mu
        self.beta = beta
        self.inter_beta = inter_beta
        self.inter_rec = inter_rec
        self.states = ('s', 'i', 'r')

        # Define class network
        self.attrs = {}
        self.network = network
        self.network_conn = network_adj[0]
        self.network_nodes = network_adj[1]
        self.network_size = [len(list(network.iter_nodes(layer))) for layer in network.get_layers()]

        # All nodes are susceptible on beginning
        self.network_attrs = dict([(n_idx, (node, {"i": 0, "s": 1, "r": 0}))
                                   for n_idx, node in enumerate(self.network_nodes)])
        self.node2ind = dict([(str(node), n_idx) for n_idx, node in enumerate(self.network_nodes)])

        # Infect seed nodes
        for seed in seed_nodes:
            self.infect_node(self.node2ind[str(seed)])

    def get_by_attr(self, attr, val):
        return [idx for idx, nd in self.network_attrs.items() if nd[1].get(attr, 0) == val]

    def get_attr(self, node, atrr):
        return self.network_attrs[node][1][atrr]

    def get_interconnected(self, node):
        node_cords = self.network_attrs[node][0]
        if isinstance(self.network, nettools.multiplex.MultiplexNetwork):
            return [[(node_cords[0], layer), None] for layer in self.network._nodeToLayers[node_cords[0]]]
        else:
            return [int_nd for int_nd in self.network._net[node_cords].keys() if node_cords[0] != int_nd[0]]

    def one_epoch(self):
        # Iterate over infected
        infected_attr = self.get_by_attr('i', 1)
        for node in infected_attr:
            # Infect neighbours
            nb_nodes = np.nonzero(self.network_conn[node])
            for nb_nd in nb_nodes[1]:
                node_layer = self.network_attrs[node][0][1]
                nb_layer = self.network_attrs[nb_nd][0][1]
                # Add additional weight
                if node_layer != nb_layer:
                    new_beta = self.inter_beta
                else:
                    new_beta = self.beta
                # Infect new node
                if not (self.get_attr(nb_nd, 'r') or self.get_attr(nb_nd, 'i')):
                    dc_spread = random.uniform(0, 1)
                    if new_beta > dc_spread:
                        self.infect_node(nb_nd)
            # Recovery
            rc_spread = random.uniform(0, 1)
            if self.mu > rc_spread:
                self.recover_node(node)
                inter_nodes = self.get_interconnected(node)
                for i_node, attr in inter_nodes:
                    # Recover inter connected node
                    rc_inter = random.uniform(0, 1)
                    if self.inter_rec > rc_inter and i_node != node:
                        self.recover_node(self.node2ind[str(i_node)])

    def recover_node(self, node):
        self.network_attrs[node][1]['r'] = 1
        self.network_attrs[node][1]['i'] = 0
        self.network_attrs[node][1]['s'] = 0

    def infect_node(self, node):
        self.network_attrs[node][1]['s'] = 0
        self.network_attrs[node][1]['i'] = 1

    def get(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        return self.get_by_attr(state, 1)

    def get_num(self, state):
        return len(self.get(state))

    def run(self, epochs=200, visualize=True):
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            if visualize:
                visualize_epidemic(self.network, self.network_attrs)

    def epidemic_data(self, epochs=50, show=True):
        s, i, r = [], [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            r.append(self.get_num(state='r'))
        if show:
            plt.figure()
            plt.hold(True)
            plt.plot(s, color='g')
            plt.plot(i, color='r')
            plt.plot(r, color='b')
            plt.show(True)
        else:
            return s, i, r


class SISMultilayerPymnet(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4, inter_beta=0.9, inter_rec=0.9):
        super(SISMultilayerPymnet, self).__init__()
        # If seed node is None take random
        network_adj = network.get_supra_adjacency_matrix()
        if seed_nodes is None:
            seed_nodes = [random.choice(network_adj[1])]
        # Process properties
        self.mu = mu
        self.beta = beta
        self.inter_rec = inter_rec
        self.inter_beta = inter_beta
        self.states = ('s', 'i')

        # Define class network
        self.attrs = {}
        self.network = network
        self.network_conn = network_adj[0]
        self.network_nodes = network_adj[1]
        self.network_size = [len(list(network.iter_nodes(layer))) for layer in network.get_layers()]

        # All nodes are susceptible on beginning
        self.network_attrs = dict([(n_idx, (node, {"i": 0, "s": 1}))
                                   for n_idx, node in enumerate(self.network_nodes)])
        self.node2ind = dict([(str(node), n_idx) for n_idx, node in enumerate(self.network_nodes)])

        # Infect seed nodes
        for seed in seed_nodes:
            self.infect_node(self.node2ind[str(seed)])

    def get_by_attr(self, attr, val):
        return [idx for idx, nd in self.network_attrs.items() if nd[1].get(attr, 0) == val]

    def get_attr(self, node, atrr):
        return self.network_attrs[node][1][atrr]

    def get_interconnected(self, node):
        node_cords = self.network_attrs[node][0]
        if isinstance(self.network, nettools.multiplex.MultiplexNetwork):
            return [[(node_cords[0], layer), None] for layer in self.network._nodeToLayers[node_cords[0]]]
        else:
            return [int_nd for int_nd in self.network._net[node_cords].keys() if node_cords[0] != int_nd[0]]

    def one_epoch(self):
        # Iterate over infected
        infected_attr = self.get_by_attr('i', 1)
        for node in infected_attr:
            # Infect neighbours
            nb_nodes = np.nonzero(self.network_conn[node])
            for nb_nd in nb_nodes[1]:
                node_layer = self.network_attrs[node][0][1]
                nb_layer = self.network_attrs[nb_nd][0][1]
                # Add additional weight
                if node_layer != nb_layer:
                    new_beta = self.inter_beta
                else:
                    new_beta = self.beta
                # Infect new node
                if not self.get_attr(nb_nd, 'i'):
                    dc_spread = random.uniform(0, 1)
                    if new_beta > dc_spread:
                        self.infect_node(nb_nd)
            # Recovery
            rc_spread = random.uniform(0, 1)
            if self.mu > rc_spread:
                self.recover_node(node)
                inter_nodes = self.get_interconnected(node)
                for i_node, attr in inter_nodes:
                    # Recover inter connected node
                    rc_inter = random.uniform(0, 1)
                    if self.inter_rec > rc_inter and i_node != node:
                        self.recover_node(self.node2ind[str(i_node)])

    def recover_node(self, node):
        self.network_attrs[node][1]['i'] = 0
        self.network_attrs[node][1]['s'] = 1

    def infect_node(self, node):
        self.network_attrs[node][1]['s'] = 0
        self.network_attrs[node][1]['i'] = 1

    def get(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        return self.get_by_attr(state, 1)

    def get_num(self, state):
        return len(self.get(state))

    def run(self, epochs=200, visualize=True):
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            # Visualize epidemic process
            if visualize:
                visualize_epidemic(self.network, self.network_attrs)

    def epidemic_data(self, epochs=50, show=False):
        s, i = [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
        if show:
            plt.figure()
            plt.hold(True)
            plt.plot(s, color='g')
            plt.plot(i, color='r')
            plt.show(True)
        return s, i


class SIRMultiplex(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4):
        super(SIRMultiplex, self).__init__()
        if isinstance(network, nettools.multiplex.MultiplexNetwork):
            network = network.network
        # If seed node is None take random
        if not seed_nodes:
            seed_n = random.randint(0, network.shape[0] - 1)
            seed_l = random.randint(0, network.shape[2] - 1)
            seed_nodes = [(seed_n, seed_l)]

        # Process properties
        self.mu = mu
        self.beta = beta
        self.states = ('s', 'i', 'r')

        # Define class network
        self.attrs = {}
        self.network = network
        self.network_size = network.shape

        # If int set same for all layers
        if isinstance(beta, float) or isinstance(beta, int):
            inter_beta_dict = dict([(l_idx, beta) for l_idx in range(self.network_size[2])])
            self.beta = dict([(l_idx, inter_beta_dict) for l_idx in range(self.network_size[2])])
        if isinstance(mu, float) or isinstance(mu, int):
            inter_rec_dict = dict([(l_idx, mu) for l_idx in range(self.network_size[2])])
            self.mu = dict([(l_idx, inter_rec_dict) for l_idx in range(self.network_size[2])])

        # Check inter dicts
        chk_inter_rec = [len(kk.keys()) for kk in self.beta.values()]
        chk_inter_beta = [len(kk.keys()) for kk in self.mu.values()]
        if len(chk_inter_beta) != self.network_size[2] and set(chk_inter_beta).pop() != self.network_size[2]:
            raise AttributeError("Interbeta should be integer or dictionary, with values for each layer.")
        if len(chk_inter_rec) != self.network_size[2] and set(chk_inter_rec).pop() != self.network_size[2]:
            raise AttributeError("Interbeta should be integer or dictionary, with values for each layer.")

        # All nodes are susceptible on beginning
        self.network_state = np.zeros(network.shape[1:])
        self.state2id = {"s": 0, "i": 1, "r": 2}

        # Infect seed nodes
        for seed in seed_nodes:
            if isinstance(seed, int):
                for layer_idx in range(self.network_size[2]):
                    self.infect_node((seed, layer_idx))
            else:
                self.infect_node(seed)

    def recover_node(self, node):
        if self.network_state[node] == self.state2id['i']:
            self.network_state[node] = self.state2id['r']

    def infect_node(self, node):
        if self.network_state[node] == self.state2id['s']:
            self.network_state[node] = self.state2id['i']

    def one_epoch(self):
        # Iterate over infected
        infected_pos = np.vstack(self.get_by_state('i'))
        infected_number = infected_pos.shape[1]
        for node_idx in range(infected_number):
            node = infected_pos[:, node_idx]
            # Infect neighbours
            nb_nodes = np.nonzero(self.network[node[0], :, node[1]])
            for nb_nd in nb_nodes[0]:
                # Infect new node same layer
                dc_spread = random.uniform(0, 1)
                if self.beta[node[1]][node[1]] > dc_spread:
                    self.infect_node((nb_nd, node[1]))
            # Infect new node on different layer
            layer_vote = np.random.uniform(0, 1, size=(self.network.shape[2],))
            # noinspection PyTypeChecker
            layer_betas = [x_it[1] for x_it in sorted(self.beta[node[1]].items(), key=self.beta[node[1]].get)]
            inter_infect = np.where(layer_vote < np.array(layer_betas))[0]
            for int_inf in inter_infect:
                self.infect_node((node[0], int_inf))

            # Recovery
            rc_spread = random.uniform(0, 1)
            if self.mu[node[1]][node[1]] > rc_spread:
                self.recover_node((node[0], node[1]))

            # Recover node on different layer
            layer_rc_vote = np.random.uniform(0, 1, size=(self.network.shape[2],))
            # noinspection PyTypeChecker
            layer_rec = [x_it[1] for x_it in sorted(self.mu[node[1]].items(), key=self.mu[node[1]].get)]
            inter_rec = np.where(layer_rc_vote < np.array(layer_rec))[0]
            for int_rec in inter_rec:
                if node[1] != int_rec:
                    self.recover_node((node[0], int_rec))

    def get_by_state(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        return np.where(self.network_state == self.state2id[state])

    def get_num(self, state):
        return len(self.get_by_state(state)[0])

    def run(self, epochs=200, visualize=False, layers=None, labels=False, pause=2):
        plt.ion()
        infected_list = []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            if visualize:
                visualize_epidemic_image_style(self.network, self.network_state, layers=layers,
                                               labels=labels, pause=pause)
            infected_list.append(self.get_num('i') + self.get_num('r'))
        return infected_list

    # noinspection PyAugmentAssignment
    def epidemic_data(self, epochs=50, show=True):
        s, i, r = [], [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            r.append(self.get_num(state='r'))
            self.one_epoch()
        if show:
            # Conversion
            s = np.array(s)
            nodes_layers = self.network_size[0] * self.network_size[2]
            s = s / float(nodes_layers)
            r = np.array(r)
            r = r / float(nodes_layers)
            i = np.array(i)
            i = i / float(nodes_layers)
            # Show
            plt.figure()
            plt.hold(True)
            plt.plot(s, color='g')
            plt.plot(i, color='r')
            plt.plot(r, color='b')
            plt.show()
        else:
            return s, i, r


class SISMultiplex(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4, inter_beta=0.9, inter_rec=0.3):
        super(SISMultiplex, self).__init__()
        if isinstance(network, nettools.multiplex.MultiplexNetwork):
            network = network.network
        # If seed node is None take random
        if not seed_nodes:
            seed_n = random.randint(0, network.shape[0] - 1)
            seed_l = random.randint(0, network.shape[2] - 1)
            seed_nodes = [(seed_n, seed_l)]

        # Process properties
        self.mu = mu
        self.beta = beta
        self.inter_beta = inter_beta
        self.inter_rec = inter_rec
        self.states = ('s', 'i')

        # Define class network
        self.attrs = {}
        self.network = network
        self.network_size = network.shape

        # All nodes are susceptible on beginning
        self.network_state = np.zeros(network.shape[1:])
        self.state2id = {"s": 0, "i": 1}

        # Infect seed nodes
        for seed in seed_nodes:
            if isinstance(seed, int):
                for layer_idx in range(self.network_size[2]):
                    self.infect_node((seed, layer_idx))
            else:
                self.infect_node(seed)

    def recover_node(self, node):
        if self.network_state[node] == self.state2id['i']:
            self.network_state[node] = self.state2id['s']

    def infect_node(self, node):
        if self.network_state[node] == self.state2id['s']:
            self.network_state[node] = self.state2id['i']

    def one_epoch(self):
        # Iterate over infected
        infected_pos = np.vstack(self.get_by_state('i'))
        infected_number = infected_pos.shape[1]
        for node_idx in range(infected_number):
            node = infected_pos[:, node_idx]
            # Infect neighbours
            nb_nodes = np.nonzero(self.network[node[0], :, node[1]])
            for nb_nd in nb_nodes[0]:
                # Infect new node same layer
                dc_spread = random.uniform(0, 1)
                if self.beta > dc_spread:
                    self.infect_node((nb_nd, node[1]))
            # Infect new node on different layer
            layer_vote = np.random.uniform(0, 1, size=(self.network.shape[2],))
            inter_infect = np.where([layer_vote < self.inter_beta])[1]
            inter_infected_number = len(inter_infect)
            for int_inf in range(inter_infected_number):
                self.infect_node((node[0], int_inf))

            # Recovery
            rc_spread = random.uniform(0, 1)
            if self.mu > rc_spread:
                self.recover_node((node[0], node[1]))

            # Recover node on different layer
            layer_rc_vote = np.random.uniform(0, 1, size=(self.network.shape[2],))
            inter_rec = np.where([layer_rc_vote < self.inter_rec])[1]
            inter_recovery_number = len(inter_rec)
            for int_rec in range(inter_recovery_number):
                if node[1] != int_rec:
                    self.recover_node((node[0], int_rec))

    def get_by_state(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIS model")
        return np.where(self.network_state == self.state2id[state])

    def get_num(self, state):
        return len(self.network_state[self.network_state == self.state2id[state]])

    def run(self, epochs=200, visualize=False, layers=None, labels=False, pause=2):
        plt.ion()
        infected_list = []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            if visualize:
                visualize_epidemic_image_style(self.network, self.network_state, layers=layers,
                                               labels=labels, pause=pause)
            infected_list.append(self.get_num('i'))
        return infected_list

    # noinspection PyAugmentAssignment
    def epidemic_data(self, epochs=50, show=True):
        s, i = [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            self.one_epoch()
        if show:
            # Conversion
            s = np.array(s)
            nodes_layers = self.network_size[0] * self.network_size[2]
            s = s / float(nodes_layers)
            i = np.array(i)
            i = i / float(nodes_layers)
            # Show
            plt.figure()
            plt.hold(True)
            plt.plot(s, color='g')
            plt.plot(i, color='r')
            plt.show()
        else:
            return s, i


class SIRMultiplexNumpy(EpidemicModel):
    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4):
        super(SIRMultiplexNumpy, self).__init__()
        if isinstance(network, nettools.multiplex.MultiplexNetwork):
            network = network.network
        # If seed node is None take random
        if not seed_nodes:
            seed_n = random.randint(0, network.shape[0] - 1)
            seed_l = random.randint(0, network.shape[2] - 1)
            seed_nodes = [(seed_n, seed_l)]

        # Process properties
        self.mu = mu
        self.beta = beta

        # Define class network
        self.attrs = {}
        self.network = network
        self.states = ('s', 'i', 'r')
        self.network_size = network.shape

        # If int set same for all layers
        if isinstance(beta, float) or isinstance(beta, int):
            inter_beta_dict = dict([(l_idx, beta) for l_idx in range(self.network_size[2])])
            self.beta = dict([(l_idx, inter_beta_dict) for l_idx in range(self.network_size[2])])
        if isinstance(mu, float) or isinstance(mu, int):
            inter_rec_dict = dict([(l_idx, mu) for l_idx in range(self.network_size[2])])
            self.mu = dict([(l_idx, inter_rec_dict) for l_idx in range(self.network_size[2])])

        # Check inter dicts
        chk_inter_rec = [len(kk.keys()) for kk in self.beta.values()]
        chk_inter_beta = [len(kk.keys()) for kk in self.mu.values()]
        if len(chk_inter_beta) != self.network_size[2] and set(chk_inter_beta).pop() != self.network_size[2]:
            raise AttributeError("Interbeta should be integer or dictionary, with values for each layer.")
        if len(chk_inter_rec) != self.network_size[2] and set(chk_inter_rec).pop() != self.network_size[2]:
            raise AttributeError("Interbeta should be integer or dictionary, with values for each layer.")

        # All nodes are susceptible on beginning
        self.infected_state = np.zeros((network.shape[2], network.shape[0]))
        self.recovery_state = np.zeros((network.shape[2], network.shape[0]))

        # Infect seed nodes
        for seed in seed_nodes:
            if isinstance(seed, int):
                for layer_idx in range(self.network_size[2]):
                    self.infect_node((seed, layer_idx))
            else:
                self.infect_node(seed)

    def recover_node(self, node):
        if self.network_state[node] == self.state2id['i']:
            self.network_state[node] = self.state2id['r']

    def infect_node(self, node):
        self.infected_state[node[1], node[0]] = 1

    def one_epoch(self):
        # Phase 1, Infection
        infected_idx = np.where((self.infected_state * (1 - self.recovery_state)) > 0)
        phase_mat = np.zeros(self.network.shape)
        phase_mat[infected_idx[1], :, infected_idx[0]] = self.network[infected_idx[1], :, infected_idx[0]]
        inf_mat_rnd = np.random.randint(0, 1000, self.network.shape) * 0.001

        # Beta scores
        beta_sc = np.array([[bb for bk, bb in x_b.items() if x_k == bk][0] for x_k, x_b in self.beta.items()])
        res_f_phase = (phase_mat * inf_mat_rnd)
        beta_rep = np.repeat(
            np.repeat(beta_sc[np.newaxis, np.newaxis, :], self.network.shape[0], axis=0), self.network.shape[1], axis=1
        )
        res_f_phase[res_f_phase != 0] = res_f_phase[res_f_phase != 0] < beta_rep[res_f_phase != 0]
        infected_n = np.transpose(np.sum(res_f_phase, axis=0), [1, 0]) + self.infected_state
        infected_n = infected_n.clip(0, 1)

        # Sinking
        beta_mat = np.array([[yf if yk != xk else 1 for yk, yf in sorted(xf.items())] for
                             xk, xf in sorted(self.beta.items())])
        beta_mat = np.repeat(beta_mat[:, :, np.newaxis], self.network.shape[0], axis=2)
        inf_sink = np.repeat(infected_n[:, np.newaxis, :], self.network.shape[2], axis=1)
        random_sink = np.random.randint(0, 1000, inf_sink.shape) * 0.001
        inf_rnd = inf_sink * random_sink
        inf_rnd[inf_rnd != 0] = inf_rnd[inf_rnd != 0] < beta_mat[inf_rnd != 0]

        after_sink = np.sum(inf_rnd, axis=0)
        rec_sink = np.repeat(self.infected_state[:, np.newaxis, :], self.network.shape[2], axis=1)
        rec_mu_mat = np.array([[yf for yk, yf in sorted(xf.items())] for xk, xf in sorted(self.mu.items())])
        rec_mu_mat = np.repeat(rec_mu_mat[:, :, np.newaxis], self.network.shape[0], axis=2)
        rec_st_rnd = np.random.randint(0, 1000, rec_sink.shape) * 0.001
        rec_sink_rnd = rec_sink * rec_st_rnd
        rec_sink_rnd[rec_sink_rnd != 0] = rec_sink_rnd[rec_sink_rnd != 0] < rec_mu_mat[rec_sink_rnd != 0]
        self.infected_state = after_sink.clip(0, 1)
        self.recovery_state = self.recovery_state + np.sum(rec_sink_rnd, axis=0)
        self.recovery_state = self.recovery_state.clip(0, 1)

    def get_by_state(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        elif state == "i":
            return np.where((self.infected_state * (1 - self.recovery_state)) > 0)
        elif state == "r":
            return np.where(self.recovery_state > 0)
        else:
            return np.where((self.recovery_state + self.infected_state) == 0)

    def get_num(self, state):
        # Check state
        if state not in self.states:
            raise ValueError("Wrong state for SIR model")
        elif state == "i":
            inf_mat = (self.infected_state * (1 - self.recovery_state))
            return len(inf_mat[inf_mat > 0])
        elif state == "r":
            return len(self.recovery_state[self.recovery_state > 0])
        else:
            sus_mat = self.recovery_state + self.infected_state
            return len(sus_mat[sus_mat == 0])

    def run(self, epochs=200, visualize=False, layers=None, labels=False, pause=2):
        plt.ion()
        infected_list = []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            self.one_epoch()
            if visualize:
                network_state = np.transpose(self.infected_state, [1, 0]) + np.transpose(self.recovery_state, [1, 0])
                visualize_epidemic_image_style(self.network, network_state, layers=layers,
                                               labels=labels, pause=pause)
            infected_list.append(self.get_num('i') + self.get_num('r'))
        return infected_list

    # noinspection PyAugmentAssignment
    def epidemic_data(self, epochs=50, show=True):
        s, i, r = [], [], []
        # Iterate over disease epochs
        for dt in range(0, epochs):
            s.append(self.get_num(state='s'))
            i.append(self.get_num(state='i'))
            r.append(self.get_num(state='r'))
            self.one_epoch()
        if show:
            # Conversion
            s = np.array(s)
            nodes_layers = self.network_size[0] * self.network_size[2]
            s = s / float(nodes_layers)
            r = np.array(r)
            r = r / float(nodes_layers)
            i = np.array(i)
            i = i / float(nodes_layers)
            # Show
            plt.figure()
            plt.hold(True)
            plt.plot(s, color='g')
            plt.plot(i, color='r')
            plt.plot(r, color='b')
            plt.show(True)
        else:
            return s, i, r

if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    from nettools.utils.netutils import load_mtx
    from nettools.multiplex import MultiplexConstructor
    ng = NetworkGenerator(nodes=15)
    ba2 = ng.ba_network(m0=3)
    er1 = ng.er_network(p=44.0 / 20.0)
    ba3 = ng.ba_network()
    net = load_mtx("socfb-Berkeley13.mtx")
    mc = MultiplexConstructor()
    mn = mc.construct(ba2, ba3)
    mn2 = mc.construct(net)
    beta_param = {0: {0: 0.05}}
    inter_beta_v2 = {0: {0: 0.1, 1: 0.1}, 1: {0: 0.1, 1: 0.1}}
    # sir = SIRMultiplexNumpy(mn2, beta=beta_param, mu=1.0)
    from timeit import Timer
    # t = Timer(lambda: sir.epidemic_data(epochs=10, show=False))
    # print t.timeit(number=1)
    # sir.run(visualize=True, layers=[0, 1], labels=True, pause=5)
    sir2 = SIRMultiplex(mn2, beta=beta_param, mu=1.0)
    t = Timer(lambda: sir2.epidemic_data(epochs=10, show=False))
    print t.timeit(number=1)
    # plt.show()
    # sir.run(visualize=False, labels=True, layers=[0])
    # test_net = mc.rewire_hubs(ba1, rsteps=100)
    # cnet = er([[y for y in range(20)] for x in range(3)], p=0.3, edges=None)
