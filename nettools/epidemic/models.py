import random
import numpy as np
import matplotlib.pyplot as plt

from pymnet import *
from abc import ABCMeta, abstractmethod

plt.ion()


def visualize_epidemic(vnetwork, net_attrs):
    """
        Visualize network using Multilayer Network Library function *draw*.

        :param vnetwork: Visualized network
        :param net_attrs: Network attributes
        :return: Node votes (dict)
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


class SIRMultilayer(EpidemicModel):

    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4, inter_beta=0.9, inter_rec=0.3):
        super(SIRMultilayer, self).__init__()
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
        if isinstance(self.network, MultiplexNetwork):
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


class SISMultilayer(EpidemicModel):

    def __init__(self, network, seed_nodes=None, mu=0.01, beta=0.4, inter_beta=0.9, inter_rec=0.9):
        super(SISMultilayer, self).__init__()
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
        if isinstance(self.network, MultiplexNetwork):
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


if __name__ == '__main__':
    cnet = er([[y for y in range(20)] for x in range(3)], p=0.3, edges=None)
    sir = SIRMultilayer(cnet, beta=0.4, mu=0.1, inter_beta=1.0, inter_rec=1.0)
    sir.run(epochs=50, visualize=True)

