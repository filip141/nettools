import numpy as np
import matplotlib.pyplot as plt
from nettools.utils import NX_CENTRALITY
from nettools.epidemic import SIRMultiplex
from nettools.multiplex import InterMeasures
from nettools.monoplex import NetworkGenerator
from nettools.monoplex import CentralityMeasure
from nettools.multiplex import MultiplexConstructor


def centrality_method_test(test_properties=None):
    # Define test properties
    if test_properties is None:
        test_properties = {}
    # Complete dictionary
    if test_properties.get("nodes") is None:
        test_properties["nodes"] = 200
    if test_properties.get("points") is None:
        test_properties["points"] = [5, 10, 20, 30, 40]
    if test_properties.get("mu") is None:
        test_properties["mu"] = 0.1
    if test_properties.get("beta") is None:
        test_properties["beta"] = 0.3
    if test_properties.get("mean_num") is None:
        test_properties["mean_num"] = 50
    if test_properties.get("ntimes") is None:
        test_properties["ntimes"] = 50
    if test_properties.get("epochs") is None:
        test_properties["epochs"] = 50

    # Build network list
    # Test supported only for one network, can be multilayer
    if test_properties.get("networks") is None:
        test_properties["networks"] = [{"degree":  4.0, "type": "ER"}]
    else:
        test_properties["networks"] = test_properties["networks"][0]

    plt.ion()
    # Create networks
    nmethods = 8
    mth_val_norm = None
    ng = NetworkGenerator(nodes=test_properties["nodes"])
    mc = MultiplexConstructor()
    # Clear method scores
    method_scores = {}
    for method in NX_CENTRALITY.keys():
        method_scores[method] = 0
    print("Analysing spreading for BA Network")
    for rl_idx in range(test_properties["ntimes"]):
        # Examine centrality
        result_counter = 0
        method_list = []
        results_matrix = np.zeros((nmethods, test_properties["epochs"]))
        for idx, method in enumerate(NX_CENTRALITY.keys()):
            avg_results = np.zeros((test_properties["mean_num"], test_properties["epochs"]))
            if method == 'supernode':
                continue
            method_list.append(method)
            for n_time in range(0, test_properties["mean_num"]):
                network_list = []
                for network_pros in test_properties["networks"]:
                    net_type = network_pros['type']
                    net_deg = network_pros['degree']
                    if net_type.lower() == "er":
                        net = ng.er_network(p=net_deg / float(test_properties["nodes"]))
                    else:
                        net = ng.ba_network(m0=int(net_deg / 2.0))
                    network_list.append(net)
                mn = mc.construct(*network_list)
                cn = CentralityMeasure(InterMeasures.aggregate(mn.network))
                results_cn = cn.network_cn(method)
                if method == 'hits':
                    results_cn = results_cn[1]
                best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
                sir = SIRMultiplex(mn, beta=test_properties["beta"], mu=test_properties["mu"],
                                   inter_beta=0.9, inter_rec=0.9, seed_nodes=[best_nodes[0][0]])
                result = sir.run(epochs=test_properties["epochs"], visualize=False, layers=[0], labels=True, pause=2)
                avg_results[n_time] = np.array(result) / float(test_properties["nodes"])
            print("Analysed method: {}".format(method))
            results_matrix[result_counter] = np.mean(avg_results, axis=0)
            result_counter += 1

        print("Result functions completed, start voting, Number: {}, Range: {}".format(rl_idx,
                                                                                       test_properties["points"]))
        # Vote
        for point in test_properties["points"]:
            max_args = list(np.argsort(results_matrix[:, point])[::-1])
            vote = 9
            for mth_idx in max_args:
                method_scores[method_list[mth_idx]] += vote
                vote -= 1
        print("Score for each method: ")
        print(method_scores)
        mth_val = np.array(method_scores.values())
        mth_val_norm = mth_val / float(np.max(mth_val))
    return mth_val_norm


def centrality_recovery_rate_test(test_properties=None, visualise=False):
    # Define test properties
    if test_properties is None:
        test_properties = {}
    # Complete dictionary
    if test_properties.get("nodes") is None:
        test_properties["nodes"] = 200
    if test_properties.get("points") is None:
        test_properties["points"] = [5, 10, 20, 30, 40]
    if test_properties.get("mu") is None:
        test_properties["mu"] = 0.1
    if test_properties.get("beta") is None:
        test_properties["beta"] = 0.3
    if test_properties.get("mean_num") is None:
        test_properties["mean_num"] = 50
    if test_properties.get("ntimes") is None:
        test_properties["ntimes"] = 50
    if test_properties.get("epochs") is None:
        test_properties["epochs"] = 50

    # Build network list
    if test_properties.get("networks") is None:
        test_properties["networks"] = [[{"degree":  4.0, "type": "ER"}]]

    plt.ion()
    # Create networks
    nmethods = 8
    gen_network_list = test_properties["networks"]
    ng = NetworkGenerator(nodes=test_properties["nodes"])
    mc = MultiplexConstructor()

    print("Analysing recovery rate for BA Network")
    # Examine centrality
    result_counter = 0
    results_names = []
    results_matrix = np.zeros((len(gen_network_list) * nmethods, test_properties["epochs"]))
    for idx, method in enumerate(NX_CENTRALITY.keys()):
        for net_num, mult_network in enumerate(gen_network_list):
            avg_results = np.zeros((test_properties["mean_num"], test_properties["epochs"]))
            if method == 'supernode':
                continue
            for n_time in range(0, test_properties["mean_num"]):
                network_list = []
                for network_pros in mult_network:
                    net_type = network_pros['type']
                    net_deg = network_pros['degree']
                    if net_type.lower() == "er":
                        net = ng.er_network(p=net_deg / float(test_properties["nodes"]))
                    else:
                        net = ng.ba_network(m0=int(net_deg / 2.0))
                    network_list.append(net)
                mn = mc.construct(*network_list)
                cn = CentralityMeasure(InterMeasures.aggregate(mn.network))
                results_cn = cn.network_cn(method)
                if method == 'hits':
                    results_cn = results_cn[1]
                best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
                sir = SIRMultiplex(mn, beta=test_properties["beta"], mu=test_properties["mu"],
                                   inter_beta=0.9, inter_rec=0.9, seed_nodes=[best_nodes[0][0]])
                result = sir.run(epochs=test_properties["epochs"], visualize=False, layers=[0], labels=True, pause=2)
                avg_results[n_time] = np.array(result) / float(test_properties["nodes"])
            print("Analysed method: {}, Network: {}".format(method, net_num))
            results_names.append("network_{}_{}".format(net_num, method))
            results_matrix[result_counter] = np.mean(avg_results, axis=0)
            if visualise:
                plt.plot(results_matrix[result_counter], hold=True, label="network_{}_{}".format(net_num, method))
            result_counter += 1
    plt.legend()
    plt.show(True)
    return results_matrix, results_names


def spread_eff_centr_test(test_properties=None):
    pass


if __name__ == '__main__':
    test_props = {"networks": [[{"degree": 4.0, "type": "ER"},
                               {"degree": 4.0, "type": "ER"},
                               {"degree": 4.0, "type": "ER"}],
                               [{"degree": 4.0, "type": "BA"}]]}
    centrality_recovery_rate_test(test_props, visualise=True)
