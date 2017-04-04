from pymnet import *
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # G=nx.Graph()
    # G.add_node(1)
    # G.add_nodes_from([2,3])
    # G.add_edges_from([(1, 2), (1, 3)])
    # H = nx.path_graph(10)
    # G.add_edges_from(H.edges())
    # G.add_edges_from([(1, 2), (1, 3)])
    # G.add_edges_from([(1, 2), (1, 3), (5, 1), (5, 3)])
    # G.add_node(1)
    # G.add_edge(1, 2)
    # print(G.number_of_nodes())
    # H = nx.DiGraph(G)
    # H.edges()
    # edgelist = [(0, 1), (1, 2), (2, 3)]
    # H = nx.Graph(edgelist)
    # G[1][3]['color'] = 'blue'
    # import math
    # G.add_edge('y', 'x', function=math.cos)
    # G.add_node(math.cos)
    # nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color='b')
    # print(list(G.edges_iter(data='color', default='red')))
    # # nx.set_node_attributes(G, 'betweenness', 1.)
    # G = nx.path_graph(3)
    # bb = nx.betweenness_centrality(G)
    # nx.set_node_attributes(G, 'betweenness', bb)
    mplex = MultiplexNetwork(couplings=('categorical', 1))
    mplex[1, 'a'][2, 'a'] = 1
    mplex.A['a'][1, 3] = 1
    mnet = MultilayerNetwork(aspects=1)
    mnet[1, 'a'][2, 'b'] = 1
    mnet[1, 'a'][4, 'a'] = 1
    mnet[1, 'a'][5, 'a'] = 1
    mnet[1, 'a'][6, 'b'] = 1
    import matplotlib
    matplotlib.use('Agg')
    net = models.er_multilayer(5, 2, 0.2)
    fig = draw(er(10, 3 * [0.4]), layout="spring", show=True)