import numpy as np
import matplotlib.pyplot as plt
from nettools.utils import NX_CENTRALITY
from nettools.epidemic import SIRMultiplex
from nettools.monoplex import NetworkGenerator
from nettools.monoplex import CentralityMeasure
from nettools.multiplex import MultiplexConstructor


def centrality_method_test(test_properties=None):
    # Define test properties
    is_er = False
    if test_properties is None:
        test_properties = {"network": "ER", "nodes": 200, "ntimes": 50,
                           "mean_num": 50, "degree": 4.0, "epochs": 50,
                           "beta": 0.3, "mu": 0.1, "points": [5, 10, 20, 30, 40]}
    # Complete dictionary
    if test_properties.get("network") is None:
        test_properties["network"] = "ER"
    if test_properties.get("nodes") is None:
        test_properties["nodes"] = 200
    if test_properties.get("points") is None:
        test_properties["points"] = [5, 10, 20, 30, 40]
    if test_properties.get("mu") is None:
        test_properties["mu"] = 0.1
    if test_properties.get("beta") is None:
        test_properties["beta"] = 0.3
    if test_properties.get("degree") is None:
        test_properties["degree"] = 4.0
    if test_properties.get("mean_num") is None:
        test_properties["mean_num"] = 50
    if test_properties.get("ntimes") is None:
        test_properties["ntimes"] = 50
    if test_properties['network'].lower() == "er":
        is_er = True
    plt.ion()
    # Create networks
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
        if is_er:
            results_matrix = np.zeros((8, 50))
        else:
            results_matrix = np.zeros((9, 50))
        for idx, method in enumerate(NX_CENTRALITY.keys()):
            avg_results = np.zeros((test_properties["mean_num"], 50))
            if method == 'supernode' and is_er:
                continue
            method_list.append(method)
            for n_time in range(0, test_properties["mean_num"]):
                if is_er:
                    net = ng.er_network(p=test_properties["degree"] / 200.0)
                else:
                    net = ng.ba_network(m0=test_properties["degree"] / 2.0)
                mn = mc.construct(net)
                cn = CentralityMeasure(net.network)
                results_cn = cn.network_cn(method)
                if method == 'hits':
                    results_cn = results_cn[1]
                best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
                sir = SIRMultiplex(mn, beta=test_properties["beta"], mu=test_properties["mu"],
                                   inter_beta=0.0, inter_rec=0.0, seed_nodes=[best_nodes[0][0]])
                result = sir.run(epochs=test_properties["epochs"], visualize=False, layers=[0], labels=True, pause=2)
                avg_results[n_time] = np.array(result)
            print("Analysed method: {}".format(method))
            results_matrix[result_counter] = np.mean(avg_results, axis=0)
            # plt.plot(np.mean(avg_results, axis=0), hold=True, label=method)
            result_counter += 1

        print("Result functions completed, start voting, Number: {}, Range: {}".format(rl_idx, [5, 10, 20, 30, 40]))
        # Vote
        for point in test_properties["points"]:
            max_args = list(np.argsort(results_matrix[:, point])[::-1])
            vote = 9
            for mth_idx in max_args:
                method_scores[method_list[mth_idx]] += vote
                vote -= 1
        print(method_scores)
        print(method_scores.values())
        mth_val = np.array(method_scores.values())
        mth_val_norm = mth_val / float(np.max(mth_val))
        return mth_val_norm

if __name__ == '__main__':
    centrality_method_test()