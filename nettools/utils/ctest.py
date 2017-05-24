import os
import logging
import colorsys
import numpy as np
import nettools.utils
import nettools.epidemic
import nettools.monoplex
import nettools.multiplex
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "..", "..", "data")
logging.basicConfig(filename=os.path.join(data_dir, "ctest_log.log"), level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    ng = nettools.monoplex.NetworkGenerator(nodes=test_properties["nodes"])
    mc = nettools.multiplex.MultiplexConstructor()
    # Clear method scores
    method_scores = {}
    for method in nettools.utils.NX_CENTRALITY.keys():
        method_scores[method] = 0
    print("Analysing spreading for BA Network")
    for rl_idx in range(test_properties["ntimes"]):
        # Examine centrality
        result_counter = 0
        method_list = []
        results_matrix = np.zeros((nmethods, test_properties["epochs"]))
        for idx, method in enumerate(nettools.utils.NX_CENTRALITY.keys()):
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
                cn = nettools.monoplex.CentralityMeasure(nettools.multiplex.InterMeasures.aggregate(mn.network))
                results_cn = cn.network_cn(method)
                if method == 'hits':
                    results_cn = results_cn[1]
                best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
                sir = nettools.epidemic.SIRMultiplex(mn, beta=test_properties["beta"], mu=test_properties["mu"],
                                                     seed_nodes=[best_nodes[0][0]])
                result = sir.run(epochs=test_properties["epochs"])
                avg_results[n_time] = np.array(result) / (len(network_list) * float(test_properties["nodes"]))
            print("Analysed method: {}".format(method))
            results_matrix[result_counter] = np.mean(avg_results, axis=0)
            result_counter += 1

        print("Result functions completed, start voting, Number: {}, Range: {}".format(rl_idx,
                                                                                       test_properties["points"]))
        # Vote
        if isinstance(test_properties["points"], str):
            max_args = list(np.argsort(np.sum(results_matrix, axis=1))[::-1])
            vote = 8
            for mth_idx in max_args:
                method_scores[method_list[mth_idx]] += vote
                vote -= 1
        else:
            for point in test_properties["points"]:
                max_args = list(np.argsort(results_matrix[:, point])[::-1])
                vote = 8
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
    ng = nettools.monoplex.NetworkGenerator(nodes=test_properties["nodes"])
    mc = nettools.multiplex.MultiplexConstructor()

    print("Analysing recovery rate for BA Network")
    # Examine centrality
    result_counter = 0
    results_names = []
    results_matrix = np.zeros((len(gen_network_list) * nmethods, test_properties["epochs"]))
    for idx, method in enumerate(nettools.utils.NX_CENTRALITY.keys()):
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
                cn = nettools.monoplex.CentralityMeasure(nettools.multiplex.InterMeasures.aggregate(mn.network))
                results_cn = cn.network_cn(method)
                if method == 'hits':
                    results_cn = results_cn[1]
                best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
                sir = nettools.epidemic.SIRMultiplex(mn, beta=test_properties["beta"], mu=test_properties["mu"],
                                                     seed_nodes=[best_nodes[0][0]])
                result = sir.run(epochs=test_properties["epochs"])
                avg_results[n_time] = np.array(result) / (len(network_list) * float(test_properties["nodes"]))
            print("Analysed method: {}, Network: {}".format(method, net_num))
            results_names.append("network_{}_{}".format(net_num, method))
            results_matrix[result_counter] = np.mean(avg_results, axis=0)
            if visualise:
                plt.plot(results_matrix[result_counter], hold=True, label="network_{}_{}".format(net_num, method))
            result_counter += 1
    plt.legend()
    plt.show(True)
    return results_matrix, results_names


def spread_eff_centr_test(network, test_properties=None, log_text=""):
    nmethods = 9
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
    sum_result = test_properties.get("sum_result", False)
    centrality_result = test_properties.get("centrality")
    if centrality_result is not None:
        nmethods = len(centrality_result.keys())

    plt.ion()
    # Check network
    if isinstance(network, nettools.monoplex.Network) or isinstance(network, np.ndarray):
        network = nettools.multiplex.MultiplexNetwork(network.network)
    elif not isinstance(network, nettools.multiplex.MultiplexNetwork):
        raise AttributeError("Network should be Network object or numpy ndarray.")

    # Create networks
    logger.debug("CTEST {}: Analysing recovery rate for Network".format(log_text))
    # Examine centrality
    results_names = []
    cent_scores = np.zeros((nmethods - 2, network.get_nodes_num()))
    spread_val = np.zeros((nmethods - 2, network.get_nodes_num()))

    # If centrality present add
    if centrality_result is not None:
        cnt_methods = centrality_result.keys()
    else:
        cnt_methods = nettools.utils.NX_CENTRALITY.keys()
    idx_cent = 0
    for method in cnt_methods:
        if method == 'supernode' or method == 'eigenvector':
            logger.debug("CTEST {}: {} method currently not supported".format(log_text, method))
            continue
        results_names.append(method)
        if centrality_result is None:
            cn = nettools.monoplex.CentralityMeasure(nettools.multiplex.InterMeasures.aggregate(network.network))
            results_cn = cn.network_cn(method)
        else:
            results_cn = centrality_result[method]
        logger.debug("CTEST {}: Found centrality scores.".format(log_text))
        if method == 'hits':
            results_cn = results_cn[1]
        best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
        for cnode, cscore in best_nodes:
            avg_results = np.zeros((test_properties["mean_num"], test_properties["epochs"]))
            for n_time in range(0, test_properties["mean_num"]):
                sir = nettools.epidemic.SIRMultiplex(network, beta=test_properties["beta"], mu=test_properties["mu"],
                                                     seed_nodes=[cnode])
                result = sir.run(epochs=test_properties["epochs"])
                avg_results[n_time] = np.array(result) / float(network.get_layers_num() * network.get_nodes_num())
            # If summ result option
            if sum_result:
                spread_val[idx_cent, cnode] = np.sum(np.mean(avg_results, axis=0)) / float(test_properties["epochs"])
            else:
                spread_val[idx_cent, cnode] = np.mean(avg_results, axis=0)[-1]
            cent_scores[idx_cent, cnode] = cscore
            logger.debug("CTEST {}: SIR: node {}.".format(log_text, cnode))
        idx_cent += 1
        logger.debug("CTEST {}: {}: Spreading completed.".format(log_text, method))
        logger.debug("CTEST {}: Analysed method: {}".format(log_text, method))
    return spread_val, cent_scores, results_names


class NetworkTester(object):

    def __init__(self):
        self.queue = {}
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.plot_main = os.path.join(curr_dir, "..", "..", "data", "plots")
        if not os.path.isdir(self.plot_main):
            os.mkdir(self.plot_main)

    def add(self, scenario_name, network, test_properties):
        self.queue[scenario_name] = (network, test_properties)

    def run(self, test_type):
        # iterate over scenarios
        for scn_name, scn_val in self.queue.items():
            logger.debug("NetworkTester: Scenario {}".format(scn_name))
            if test_type == 'node_spread':
                self.multinet_spreading_test(scn_val[0], scn_name, scn_val[1])
            elif test_type == 'threshold_test':
                self.et_test(network=scn_val[0], scn_name=scn_name, test_properties=scn_val[1])
            elif test_type == 'global_spread':
                self.global_spread(network=scn_val[0], scn_name=scn_name, test_properties=scn_val[1])

    def multinet_spreading_test(self, network, sc_name, test_properties):
        n_layers = network.network.shape[2]
        mc = nettools.multiplex.MultiplexConstructor()
        # Create folder for plots
        plot_dir = os.path.join(self.plot_main, sc_name)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        # iterate over layers
        for l_idx in range(n_layers):
            # Find centrality scores
            cent_dict_full = {}
            net_obj = nettools.monoplex.Network(network.network[:, :, l_idx], n_type="Real")
            network_con = mc.construct(net_obj)
            cm_eu = nettools.monoplex.CentralityMeasure(network.network[:, :, l_idx])

            # Save degree distribution
            ax_fig = plt.figure()
            ax = plt.gca()
            deg_dst, x_axis = net_obj.degree_distribution()
            deg_idx = np.where(x_axis > 0)
            deg_dst = deg_dst[deg_idx]
            x_axis = x_axis[deg_idx]
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.scatter(x_axis, deg_dst, marker='o', color='r')
            ax_fig.savefig(os.path.join(plot_dir, "degree_dist_{}".format(l_idx)))
            for method in nettools.utils.NX_CENTRALITY.keys():
                if method == 'supernode':
                    continue
                if method == 'eigenvector':
                    continue
                results = cm_eu.network_cn(method)
                if method == 'hits':
                    method = "Hits"
                    results = results[1]
                logger.debug("CENTRALITY: {} measure complete.".format(method))
                best_nodes = sorted(results.items(), key=lambda x: x[1])[::-1]
                cent_dict_full[method] = dict(best_nodes)

            # Find spreading efficiency
            print("Network generated and constructed!")
            print("Start process...")
            test_properties['centrality'] = cent_dict_full
            spread_val, cent_scores, results_names = spread_eff_centr_test(network_con,
                                                                           test_properties=test_properties,
                                                                           log_text=sc_name)
            logger.debug("NetworkTester {}:Spreading values computed for {} layer".format(sc_name, l_idx))
            # Plot
            fig_1 = plt.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
            fig_2 = plt.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
            for method_idx in range(0, spread_val.shape[0]):
                method_scores_spread = spread_val[method_idx]
                method_scores_cent = cent_scores[method_idx]
                method_scores_cent = method_scores_cent - np.min(method_scores_cent)
                method_scores_cent = 0.65 * (method_scores_cent / np.max(method_scores_cent))
                # Find data ranks
                temp_sort = np.argsort(method_scores_cent)
                data_centrality_rank = np.empty(len(method_scores_cent), int)
                data_centrality_rank[temp_sort] = np.arange(len(method_scores_cent))

                # Normalize
                norm_cent = (cent_scores[method_idx, :] - np.min(
                    cent_scores[method_idx, :])) / np.max(cent_scores[method_idx, :])
                norm_spread = (method_scores_spread - np.min(method_scores_spread)) / np.max(method_scores_spread)

                # Define figures
                sp = fig_1.add_subplot(240 + method_idx + 1)
                sp_log = fig_2.add_subplot(240 + method_idx + 1)
                for node_id in range(network.network.shape[0]):
                    color_rgb = colorsys.hsv_to_rgb(0.65 - method_scores_cent[node_id], 0.5, 1.0)
                    sp.scatter(data_centrality_rank[node_id], method_scores_spread[node_id],
                               c=(color_rgb[0], color_rgb[1], color_rgb[2], 1))
                    sp_log.scatter(np.log10(norm_cent[node_id]), np.log10(norm_spread[node_id]),
                                   c=(color_rgb[0], color_rgb[1], color_rgb[2], 1))
                sp.set_title(results_names[method_idx])
                sp.set_ylim([np.min(method_scores_spread), np.max(method_scores_spread)])
                # set next plot
                sp_log.set_title(results_names[method_idx])
            figure_path = os.path.join(plot_dir, "spreading_{}.png".format(l_idx))
            figure_path_log = os.path.join(plot_dir, "spreading_log_{}.png".format(l_idx))
            fig_1.savefig(figure_path)
            logger.debug("NetworkTester {}:Figure saved to file,path: {}".format(sc_name, figure_path))
            fig_2.savefig(figure_path_log)
            logger.debug("NetworkTester {}:Figure saved to file,path: {}".format(sc_name, figure_path_log))

    def et_test(self, network, scn_name, test_properties):
        # Define plot dir
        plot_dir = os.path.join(self.plot_main, scn_name)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        # Define values
        npoints = test_properties.get("et_points")
        epochs = test_properties.get('epochs', 10)
        mean_num = test_properties.get("mean_num", 10)
        et_method = test_properties.get("et_method", 'k-shell')
        if npoints is None:
            raise ValueError("Attribute et_points not defined, Please define it for this test.")
        elif network.network.shape[2] > 2:
            raise ValueError("Only 2 layer network supported")
        # Define centrality
        cn = nettools.monoplex.CentralityMeasure(nettools.multiplex.InterMeasures.aggregate(network.network))
        results_cn = cn.network_cn(et_method)
        best_nodes = sorted(results_cn.items(), key=lambda x: x[1])[::-1]
        spread_bt = np.zeros((npoints, npoints))
        for beta_1 in range(npoints):
            for beta_2 in range(npoints):
                avg_results = np.zeros((mean_num, epochs))
                beta_param = {0: {0: beta_1 / float(npoints), 1: 1.0}, 1: {1: beta_2 / float(npoints), 0: 1.0}}
                rec_param = {0: {0: 1.0, 1: 0.0}, 1: {1: 1.0, 0: 0.0}}
                for n_time in range(mean_num):
                    sir = nettools.epidemic.SIRMultiplex(network, beta=beta_param,
                                                         mu=rec_param, seed_nodes=[best_nodes[0][0]])
                    result = sir.run(epochs=epochs)
                    avg_results[n_time] = np.array(result) / float(network.get_layers_num() * network.get_nodes_num())
                spread_bt[beta_1, beta_2] = np.mean(avg_results, axis=0)[-1]
                logger.debug("NetworkTester {}:Threshold Test: {}-{} analyzed".format(scn_name, beta_1, beta_2))
        plt.imshow(spread_bt, extent=[0.0, 1.0, 1.0, 0.0], cmap=plt.get_cmap('plasma'), interpolation='none')
        figure_path = os.path.join(plot_dir, "ep_thresh_{}.png".format(scn_name))
        plt.savefig(figure_path)
        logger.debug("NetworkTester {}:Figure saved to file,path: {}".format(scn_name, figure_path))

    def global_spread(self, network, scn_name, test_properties):
        # Define plot dir
        plot_dir = os.path.join(self.plot_main, scn_name)
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        logger.debug("NetworkTester {}: Global Spread test started.".format(scn_name))
        spread_val, cent_scores, results_names = spread_eff_centr_test(network, test_properties=test_properties,
                                                                       log_text=scn_name)
        fig_1 = plt.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
        fig_2 = plt.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
        for method_idx in range(spread_val.shape[0]):
            method_scores_spread = spread_val[method_idx]
            method_scores_cent = cent_scores[method_idx]
            method_scores_cent = method_scores_cent - np.mean(method_scores_cent) * 0.7
            method_scores_cent = 0.65 * (method_scores_cent / np.max(method_scores_cent))
            # Find data ranks
            temp_sort = np.argsort(method_scores_cent)
            data_centrality_rank = np.empty(len(method_scores_cent), int)
            data_centrality_rank[temp_sort] = np.arange(len(method_scores_cent))

            # Normalize
            norm_cent = (cent_scores[method_idx, :] - np.min(
                cent_scores[method_idx, :])) / np.max(cent_scores[method_idx, :])
            norm_spread = (method_scores_spread - np.min(method_scores_spread)) / np.max(method_scores_spread)

            sp = fig_1.add_subplot(240 + method_idx + 1)
            sp_log = fig_2.add_subplot(240 + method_idx + 1)
            for node_id in range(network.get_nodes_num()):
                color_rgb = colorsys.hsv_to_rgb(0.65 - method_scores_cent[node_id], 0.5, 1.0)
                sp.scatter(data_centrality_rank[node_id], method_scores_spread[node_id],
                           c=(color_rgb[0], color_rgb[1], color_rgb[2], 1))
                sp_log.scatter(np.log10(norm_cent[node_id]), np.log10(norm_spread[node_id]),
                               c=(color_rgb[0], color_rgb[1], color_rgb[2], 1))
            sp.set_title(results_names[method_idx])
            sp.set_ylim([np.min(method_scores_spread), np.max(method_scores_spread)])
        figure_path = os.path.join(plot_dir, "global_multi_spread_{}.png".format(scn_name))
        figure_path_log = os.path.join(plot_dir, "global_multi_logspread_{}.png".format(scn_name))
        fig_1.savefig(figure_path)
        logger.debug("NetworkTester {}: Figure saved to file,path: {}".format(scn_name, figure_path))
        fig_2.savefig(figure_path_log)
        logger.debug("NetworkTester {}: Figure saved to file,path: {}".format(scn_name, figure_path_log))


if __name__ == '__main__':
    from nettools.monoplex import NetworkGenerator
    from nettools.multiplex import MultiplexConstructor, InterMeasures
    from nettools.utils import load_multinet_by_name, load_monoplex_by_name

    network_fb = load_monoplex_by_name('facebook')
    network_eu = load_multinet_by_name('EUAir')
    network_fao = load_multinet_by_name('fao')
    network_london = load_multinet_by_name('london')

    avg_deg = 6.0
    nodes_nm = 500
    mc = MultiplexConstructor()
    ng = NetworkGenerator(nodes=nodes_nm)
    ba1 = ng.ba_network(m0=int(avg_deg / 2.0))
    ba2 = ng.ba_network(m0=int(avg_deg / 2.0))
    er1 = ng.er_network(p=avg_deg / float(nodes_nm - 1))
    er2 = ng.er_network(p=avg_deg / float(nodes_nm - 1))
    ba_corr = mc.rewire_hubs(ba1, rsteps=5000)
    mc = MultiplexConstructor()
    mnet_ba = mc.construct(ba1, ba2)
    mnet_fb = mc.construct(network_fb)
    mnet_ba_c = mc.construct(ba1, ba_corr)

    print("Network generated and constructed!")
    # beta_param = {0: {0: 0.1, 1: 0.1, 2: 0.1}, 1: {0: 0.1, 1: 0.1, 2: 0.1}, 2: {0: 0.1, 1: 0.1, 2: 0.1}}
    # rec_param = {0: {0: 1.0, 1: 1.0, 2: 1.0}, 1: {0: 1.0, 1: 1.0, 2: 1.0}, 2: {0: 1.0, 1: 1.0, 2: 1.0}}
    # beta_param = {0: {0: 0.1, 1: 0.1}, 1: {0: 0.6, 1: 0.6}}
    # rec_param = {0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 1.0}}
    beta_param = {0: {0: 0.05}}
    rec_param = {0: {0: 1.0}}
    test_props = {'mean_num': 100, "epochs": 10, "beta": beta_param, "mu": rec_param, "et_points": 30}

    nt = NetworkTester()
    nt.add("Synth_BA_A0_1B0_6", mnet_ba, test_props)
    # nt.add("Synth_BA_A0_1B0_6_Correlated", mnet_ba_c, test_props)
    # nt.add("SynthCorrelated", mnet_ba_c, test_props)
    # nt.add("Facebook_100Mean", mnet_fb, test_props)
    # nt.add("EUNet", network_eu, test_props)
    # nt.add("London", network_london, test_props)
    # nt.add("Fao", network_fao, test_props)
    nt.run('global_spread')
