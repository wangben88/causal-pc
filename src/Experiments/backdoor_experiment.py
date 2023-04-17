import jax.numpy as jnp
from src.MDetNets.MDNet import LazyDenseMDNet
from jax import random, grad
from src.MDetNets.training import loss, train, train_em, train_det
import pandas as pd
from src.MDetNets.operations import lazy_product, lazy_marginalize, conditional
from sklearn.model_selection import train_test_split
from math import ceil
import time
import copy
from collections import defaultdict

def load_bn_dataset(filename, minibatch_size, test_size=0.1):
    df = pd.read_pickle("src/Experiments/data/" + filename + ".pkl")
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_x = train_df.to_numpy()
    test_x = test_df.to_numpy()
    var_names = list(df)
    print(var_names)

    train_x_batched = jnp.array_split(train_x, ceil(train_x.shape[0] / minibatch_size), axis=0)
    test_x_batched = jnp.array_split(test_x, ceil(train_x.shape[0] / minibatch_size), axis=0)

    inputs_train_batched = [{
        name: jnp.array(train_x_batch[:, i]) for i, name in enumerate(var_names)
    } for train_x_batch in train_x_batched]
    inputs_test_batched = [{
        name: jnp.array(test_x_batch[:, i]) for i, name in enumerate(var_names)
    } for test_x_batch in test_x_batched]

    return inputs_train_batched, inputs_test_batched

def run_naive_benchmark(dataset, target, x, y, z, x_val, y_val):
    df = pd.read_pickle("src/Experiments/data/" + dataset + ".pkl")
    arr = df.to_numpy()
    var_names = list(df)
    print(var_names)

    x_var = next(iter(x))
    y_var = next(iter(y))


    naive_start = time.time()

    target_y_dict = defaultdict(int)
    other_y_dict = defaultdict(int)
    z_dict = defaultdict(int)
    batch_size = len(df[x_var])
    for i in range(batch_size):
        z_set = frozenset([(z_var, df[z_var][i]) for z_var in z])
        z_dict[z_set] += 1
        if df[x_var][i] == x_val:
            other_y_dict[z_set] += 1
            if df[y_var][i] == y_val:
                target_y_dict[z_set] += 1


    naive_estimate = [
        target_y_dict[z_set] / other_y_dict[z_set] * z_dict[z_set] / batch_size if other_y_dict[z_set] > 0 else 0 for
        z_set in z_dict]
    print(naive_estimate)
    naive_estimate = jnp.sum(jnp.array(naive_estimate))


    naive_end = time.time()

    return "Estimate error: " + str(jnp.abs(naive_estimate - target)) + ", Time Taken: " + str(naive_end - naive_start)

def run_benchmark(dataset, target, x, y, z, x_val, y_val, runs=10, region_shape=(1,), group_size=(10,), leaf_region_shape=(1,),
                  leaf_group_size=(2,)):

    times = []
    errors = []



    for _ in range(runs):
        inputs_train_batched, _ = load_bn_dataset(dataset, minibatch_size=10000, test_size=0.0001)
        var_names = set(inputs_train_batched[0].keys())
        print(var_names)
        other_vars = var_names.difference(x).difference(y).difference(z)

        net = LazyDenseMDNet(list(var_names))
        net.generate_staged_binary_vtree(var_stages=[frozenset(x.union(z))],
                                         region_shape=region_shape, group_size=group_size,
                                         leaf_region_shape=leaf_region_shape, leaf_group_size=leaf_group_size)
        key = random.PRNGKey(0)
        net.generate_random_det_weights(key=key, max_products=1, respect_labels=True)

        train_det(net, inputs_train_batched)

        start = time.time()

        pyxz = lazy_marginalize(net, other_vars)
        pz = lazy_marginalize(pyxz, x.union(y))
        py_xz = conditional(pyxz, x.union(z))
        prod_1 = lazy_product(py_xz, pz)
        backnet = lazy_marginalize(prod_1, z)

        middle_1 = time.time()

        # first, compilation run (to avoid measuring JAX overhead)
        est = backnet.forward({
            next(iter(x)): jnp.array([x_val, x_val]),
            next(iter(y)): jnp.array([y_val, 1-y_val]),
        })

        middle_2 = time.time()
        est = backnet.forward({
            next(iter(x)): jnp.array([x_val, x_val]),
            next(iter(y)): jnp.array([y_val, 1-y_val]),
        })

        end = time.time()

        estimate = jnp.squeeze(est)[0] / (jnp.squeeze(est)[0] + jnp.squeeze(est)[1])
        errors.append(jnp.abs(estimate - target))
        times.append(end-middle_2 + middle_1 - start)









    return "Estimate error: " + str(jnp.mean(jnp.array(errors))) + ", Time Taken: " + str(jnp.mean(jnp.array(times)))

if __name__ == "__main__":

    # Below are the tests contained in the Empirical Evaluation in the paper.
    # The datasets and target values are generated using prepare_bns.py
    # To reproduce the results, uncomment the appropriate block.

    print(run_naive_benchmark(dataset = "asia",
                              target=0.8065,
                  x = {"bronc"},
                  y = {"dysp"},
                  z = {'smoke', 'lung', 'asia', 'tub'},
                  x_val = 0,
                  y_val = 0,))
    print(run_benchmark(dataset = "asia",
                  target = 0.8065,
                  x = {"bronc"},
                  y = {"dysp"},
                  z = {'smoke', 'lung', 'asia', 'tub'},
                  x_val = 0,
                  y_val = 0,
                  runs=10
                  )
             )


    # print(run_naive_benchmark(dataset = "andes",
    #                           target=0.1432,
    #               x = {"SNode_154"},
    #               y = {"SNode_155"},
    #               z = {'DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20', 'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4', 'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8', 'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49', 'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52', 'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15', 'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56', 'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33', 'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36', 'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38', 'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55', 'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4', 'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6', 'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46', 'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26', 'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71', 'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12', 'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55', 'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100', 'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62', 'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109', 'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112', 'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117', 'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72', 'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75', 'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131', 'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135'},
    #               x_val = 1,
    #               y_val = 1,))
    # print(run_benchmark(dataset = "andes",
    #                     target=0.1432,
    #                     x={"SNode_154"},
    #                     y={"SNode_155"},
    #                     z={'DISPLACEM0', 'RApp1', 'SNode_3', 'GIVEN_1', 'RApp2', 'SNode_8', 'SNode_16', 'SNode_20',
    #                        'NEED1', 'SNode_21', 'GRAV2', 'SNode_24', 'VALUE3', 'SNode_15', 'SNode_25', 'SLIDING4',
    #                        'SNode_11', 'SNode_26', 'CONSTANT5', 'SNode_47', 'VELOCITY7', 'KNOWN6', 'RApp3', 'KNOWN8',
    #                        'RApp4', 'SNode_27', 'GOAL_2', 'GOAL_48', 'COMPO16', 'TRY12', 'TRY11', 'SNode_5', 'GOAL_49',
    #                        'SNode_6', 'GOAL_50', 'CHOOSE19', 'SNode_17', 'SNode_51', 'SYSTEM18', 'SNode_52',
    #                        'KINEMATI17', 'GOAL_53', 'IDENTIFY10', 'SNode_28', 'IDENTIFY9', 'TRY13', 'TRY14', 'TRY15',
    #                        'SNode_29', 'VAR20', 'SNode_31', 'SNode_10', 'SNode_33', 'GIVEN21', 'SNode_34', 'GOAL_56',
    #                        'APPLY32', 'GOAL_57', 'CHOOSE35', 'SNode_7', 'SNode_59', 'MAXIMIZE34', 'SNode_60', 'AXIS33',
    #                        'GOAL_61', 'WRITE31', 'GOAL_62', 'WRITE30', 'GOAL_63', 'RESOLVE37', 'SNode_64', 'NEED36',
    #                        'SNode_9', 'SNode_41', 'SNode_42', 'SNode_43', 'IDENTIFY39', 'GOAL_66', 'RESOLVE38',
    #                        'SNode_67', 'SNode_54', 'IDENTIFY41', 'GOAL_69', 'RESOLVE40', 'SNode_70', 'SNode_55',
    #                        'IDENTIFY43', 'GOAL_72', 'RESOLVE42', 'SNode_73', 'SNode_74', 'KINE29', 'SNode_4',
    #                        'SNode_75', 'VECTOR44', 'GOAL_79', 'EQUATION28', 'VECTOR27', 'RApp5', 'GOAL_80', 'RApp6',
    #                        'GOAL_81', 'TRY25', 'TRY24', 'GOAL_83', 'GOAL_84', 'CHOOSE47', 'SNode_86', 'SYSTEM46',
    #                        'SNode_156', 'NEWTONS45', 'GOAL_98', 'DEFINE23', 'SNode_37', 'IDENTIFY22', 'TRY26',
    #                        'SNode_38', 'SNode_40', 'SNode_44', 'SNode_46', 'SNode_65', 'NULL48', 'SNode_68', 'SNode_71',
    #                        'GOAL_87', 'FIND49', 'SNode_88', 'NORMAL50', 'NORMAL52', 'INCLINE51', 'SNode_91', 'SNode_12',
    #                        'SNode_13', 'STRAT_90', 'HORIZ53', 'BUGGY54', 'SNode_92', 'SNode_93', 'IDENTIFY55',
    #                        'SNode_94', 'WEIGHT56', 'SNode_95', 'WEIGHT57', 'SNode_97', 'GOAL_99', 'FIND58', 'SNode_100',
    #                        'IDENTIFY59', 'SNode_102', 'FORCE60', 'GOAL_103', 'APPLY61', 'GOAL_104', 'CHOOSE62',
    #                        'SNode_106', 'SNode_152', 'GOAL_107', 'WRITE63', 'GOAL_108', 'WRITE64', 'GOAL_109',
    #                        'GOAL_110', 'GOAL65', 'GOAL_111', 'GOAL66', 'NEED67', 'RApp7', 'RApp8', 'SNode_112',
    #                        'GOAL_113', 'GOAL68', 'GOAL_114', 'SNode_115', 'SNode_116', 'VECTOR69', 'SNode_117',
    #                        'SNode_118', 'VECTOR70', 'SNode_119', 'EQUAL71', 'SNode_120', 'GOAL_121', 'GOAL72',
    #                        'SNode_122', 'SNode_123', 'VECTOR73', 'SNode_124', 'NEWTONS74', 'SNode_125', 'SUM75',
    #                        'GOAL_126', 'GOAL_127', 'RApp9', 'RApp10', 'SNode_128', 'GOAL_129', 'GOAL_130', 'SNode_131',
    #                        'SNode_132', 'SNode_133', 'SNode_134', 'SNode_135'},
    #                     x_val=1,
    #                     y_val=1,
    #               runs=10
    #               )
    #       )


    # print(run_naive_benchmark(dataset = "win95pts",
    #                           target=0.0521,
    #               x = {"REPEAT"},
    #               y = {"Problem2"},
    #               z = {'AppOK', 'DataFile', 'AppData', 'DskLocal', 'PrtThread', 'EMFOK', 'PrtSpool', 'GDIIN', 'PrtDriver', 'DrvSet', 'DrvOK', 'GDIOUT', 'PrtSel', 'PrtDataOut', 'PrtPath', 'NtwrkCnfg', 'PTROFFLINE', 'NetOK', 'PrtCbl', 'PrtPort', 'CblPrtHrdwrOK', 'LclOK', 'PrtMpTPth', 'DS_NTOK', 'DS_LCLOK', 'NetPrint', 'DSApplctn', 'PC2PRT', 'PrtOn', 'PrtPaper', 'PrtMem', 'PrtTimeOut', 'FllCrrptdBffr', 'TnrSpply', 'PrtData', 'Problem1', 'AppDtGnTm', 'PrntPrcssTm', 'DeskPrntSpd', 'PgOrnttnOK', 'PrntngArOK', 'CmpltPgPrntd', 'GrphcsRltdDrvrSttngs', 'EPSGrphc', 'NnPSGrphc', 'PSGRAPHIC', 'PrtPScript', 'Problem4', 'FntInstlltn', 'PrntrAccptsTrtyp', 'TTOK', 'ScrnFntNtPrntrFnt', 'NnTTOK', 'TrTypFnts', 'Problem5', 'LclGrbld', 'NtGrbld', 'GrbldOtpt', 'HrglssDrtnAftrPrnt',},
    #               x_val = 1,
    #               y_val = 1,))
    # print(run_benchmark(dataset = "win95pts",
    #               target = 0.0521,
    #                     x={"REPEAT"},
    #                     y={"Problem2"},
    #                     z={'AppOK', 'DataFile', 'AppData', 'DskLocal', 'PrtThread', 'EMFOK', 'PrtSpool', 'GDIIN',
    #                        'PrtDriver', 'DrvSet', 'DrvOK', 'GDIOUT', 'PrtSel', 'PrtDataOut', 'PrtPath', 'NtwrkCnfg',
    #                        'PTROFFLINE', 'NetOK', 'PrtCbl', 'PrtPort', 'CblPrtHrdwrOK', 'LclOK', 'PrtMpTPth', 'DS_NTOK',
    #                        'DS_LCLOK', 'NetPrint', 'DSApplctn', 'PC2PRT', 'PrtOn', 'PrtPaper', 'PrtMem', 'PrtTimeOut',
    #                        'FllCrrptdBffr', 'TnrSpply', 'PrtData', 'Problem1', 'AppDtGnTm', 'PrntPrcssTm',
    #                        'DeskPrntSpd', 'PgOrnttnOK', 'PrntngArOK', 'CmpltPgPrntd', 'GrphcsRltdDrvrSttngs',
    #                        'EPSGrphc', 'NnPSGrphc', 'PSGRAPHIC', 'PrtPScript', 'Problem4', 'FntInstlltn',
    #                        'PrntrAccptsTrtyp', 'TTOK', 'ScrnFntNtPrntrFnt', 'NnTTOK', 'TrTypFnts', 'Problem5',
    #                        'LclGrbld', 'NtGrbld', 'GrbldOtpt', 'HrglssDrtnAftrPrnt', },
    #                     x_val=1,
    #                     y_val=1,
    #               runs=10
    #               )
    #          )


    # print(run_naive_benchmark(dataset = "sachs",
    #                           target=0.2869,
    #               x = {"PKA"},
    #               y = {"Raf"},
    #               z = {'PKC','Plcg', 'PIP2', 'PIP3'},
    #               x_val = 0,
    #               y_val = 0,))
    # print(run_benchmark(dataset = "sachs",
    #               target = 0.2869,
    #               x = {"PKA"},
    #               y = {"Raf"},
    #               z = {'PKC','Plcg', 'PIP2', 'PIP3'},
    #               x_val = 0,
    #               y_val = 0,
    #               runs=10
    #               )
    #          )

    # print(run_naive_benchmark(dataset="child",
    #                           target=0.3164,
    #                           x={"Sick"},
    #                           y={"XrayReport"},
    #                           z={'BirthAsphyxia', 'Disease', 'DuctFlow', 'CardiacMixing', 'HypDistrib', 'LungParench', 'HypoxiaInO2', 'LowerBodyO2', 'RUQO2', 'CO2', 'CO2Report', 'LungFlow', 'ChestXray'},
    #                           x_val=0,
    #                           y_val=0, ))
    # print(run_benchmark(dataset="child",
    #                     target=0.3164,
    #                     x={"Sick"},
    #                     y={"XrayReport"},
    #                     z={'BirthAsphyxia', 'Disease', 'DuctFlow', 'CardiacMixing', 'HypDistrib', 'LungParench', 'HypoxiaInO2', 'LowerBodyO2', 'RUQO2', 'CO2', 'CO2Report', 'LungFlow', 'ChestXray'},
    #                     x_val=0,
    #                     y_val=0,
    #                     runs=10
    #                     )
    #       )









