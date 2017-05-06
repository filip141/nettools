import colorsys
import matplotlib
import numpy as np
import multiprocessing.dummy as mp
import matplotlib.pyplot as plt
from nettools.utils.netutils import NX_CENTRALITY
from nettools.epidemic.models import SIRMultiplex
from nettools.multiplex.interdependence import InterMeasures
from nettools.monoplex.centrality import CentralityMeasure
from nettools.monoplex.syn_net_gen import NetworkGenerator, Network
from nettools.multiplex.syn_mul_gen import MultiplexConstructor, MultiplexNetwork

# Change backend
matplotlib.use('TkAgg')


def compute_spreading_for_method(param):
    network, method, test_properties, out = param[0], param[1], param[2], param[3]
    # Define matrices for scores
    cent_scores = np.zeros((network.get_nodes_num(),))
    spread_val = np.zeros((network.get_nodes_num(),))
    # Calculate centrality
    cn = CentralityMeasure(InterMeasures.aggregate(network.network))
    results_cn = cn.network_cn(method)
    print("Found centrality scores for {}".format(method))
    if method == 'hits':
        results_cn = results_cn[1]
    best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
    for cnode, cscore in best_nodes:
        avg_results = np.zeros((test_properties["mean_num"], test_properties["epochs"]))
        for n_time in range(0, test_properties["mean_num"]):
            sir = SIRMultiplex(network, beta=test_properties["beta"], mu=test_properties["mu"],
                               inter_beta=test_properties["inter_beta"], inter_rec=test_properties["inter_rec"],
                               seed_nodes=[cnode])
            result = sir.run(epochs=test_properties["epochs"])
            avg_results[n_time] = np.array(result) / float(network.get_nodes_num())
        spread_val[cnode] = np.sum(np.mean(avg_results, axis=0)) / test_properties["epochs"]
        cent_scores[cnode] = cscore
    print("Analyzed method: {}".format(method))
    out.put([spread_val, cent_scores, method])


def spread_eff_centr_test(network, test_properties=None, exclude=["supernode", "hits"]):
    # Define test properties
    if test_properties is None:
        test_properties = {}
    # Complete dictionary
    if test_properties.get("mu") is None:
        test_properties["mu"] = 0.1
    if test_properties.get("beta") is None:
        test_properties["beta"] = 0.3
    if test_properties.get("inter_beta") is None:
        test_properties["inter_beta"] = 0.9
    if test_properties.get("inter_rec") is None:
        test_properties["inter_rec"] = 0.9
    if test_properties.get("mean_num") is None:
        test_properties["mean_num"] = 50
    if test_properties.get("epochs") is None:
        test_properties["epochs"] = 50

    plt.ion()
    # Check network
    if isinstance(network, Network) or isinstance(network, np.ndarray):
        network = MultiplexNetwork(network.network)
    elif not isinstance(network, MultiplexNetwork):
        raise AttributeError("Network should be Network object or numpy ndarray.")

    # Create networks
    print("Analysing recovery rate for Network")
    # Define an output queue
    output = mp.Queue()
    arg_list = [(MultiplexNetwork(network.network), mth,
                 test_properties, output) for mth in NX_CENTRALITY.keys() if mth not in exclude]

    # Prepare pool
    process_list = []
    for worker_arg in arg_list:
        proc = mp.Process(target=compute_spreading_for_method, args=(worker_arg,))
        process_list.append(proc)
        proc.start()

    # Get result
    for p in process_list:
        p.join()
    results = [output.get() for p in process_list]
    spread_values = np.vstack([sp[0] for sp in results])
    cent_sc = np.vstack([sp[1] for sp in results])
    result_mth = [sp[2] for sp in results]
    return spread_values, cent_sc, result_mth


if __name__ == '__main__':
    nodes_nm = 200
    ng = NetworkGenerator(nodes=nodes_nm)
    bb1 = ng.ba_network(m0=15)
    bb2 = ng.er_network(p=8.0 / 20.0)
    bb3 = ng.bb_network(m0=10)
    mc = MultiplexConstructor()
    mnet_bb = mc.construct(bb1)
    print("Network generated and constructed!")
    test_props = {'mean_num': 10, "epochs": 50, "inter_beta": 0.5, "inter_rec": 0.5, "beta": 0.1, "mu": 0.2}
    print("Start process...")
    spread_val, cent_scores, results_names = spread_eff_centr_test(mnet_bb, test_properties=test_props)
    method_scores_spread = spread_val[4]
    method_scores_cent = cent_scores[4]
    method_scores_cent = 0.43 * (method_scores_cent / np.max(method_scores_cent))
    for node_id in range(nodes_nm):
        color_rgb = colorsys.hsv_to_rgb(0.56 + method_scores_cent[node_id], 0.5, 1.0)
        plt.scatter(method_scores_cent[node_id], method_scores_spread[node_id],
                    c=(color_rgb[0], color_rgb[1], color_rgb[2], 1), hold=True)
    plt.show(True)
    # test_props = {"networks": [[{"degree": 4.0, "type": "ER"},
    #                            {"degree": 4.0, "type": "ER"},
    #                            {"degree": 4.0, "type": "ER"}],
    #                            [{"degree": 4.0, "type": "BA"}]]}
    # centrality_recovery_rate_test(test_props, visualise=True)
