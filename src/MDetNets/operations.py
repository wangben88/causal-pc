from src.MDetNets.MDVtree import Vtree, VtreeLabel, MDVtree
from collections import defaultdict
import uuid
from src.MDetNets.MDNet import LazyMDNet
from src.MDetNets.weights import GeneralizedSparseWeights
import numpy as np

def lazy_marginalize(vtree, marg_set):
    marg_set = vtree.scope.intersection(marg_set)
    output_scope = vtree.scope.difference(marg_set)

    output_vtree = type(vtree)(scope=output_scope)

    # Iteratively marginalize nodes in reverse topological order and update the output vtree
    layers = vtree.make_layers()
    regions_output_children = defaultdict(list)  # for each region in vtree, record its children in the output vtree
    for layer in reversed(layers):
        for region in layer:
            output_region_dict = vtree.lazy_marg_compute_region_dict(region, marg_set)
            output_region_dict["output_region_children"] = regions_output_children[region]
            output_region = output_vtree.lazy_marg_execute_update(output_region_dict)

            regions_output_children[vtree.parent[region]].append(output_region)
            if vtree.root == region:
                output_vtree.set_root(output_region)

    return output_vtree


def lazy_product(vtree1, vtree2):

    output_scope = vtree1.scope.union(vtree2.scope)

    output_vtree = type(vtree1)(scope=output_scope)

    def product_recur(region1, region2):
        assert not ((region1 is None) and (region2 is None))

        region_dict1, region_dict2 = vtree1.lazy_prod_compute_region_dict(region1), vtree2.lazy_prod_compute_region_dict(region2)
        output_region_dict = output_vtree.lazy_prod_compute_output_region_dict(region_dict1, region_dict2)
        # Execute child update
        if not output_region_dict["is_leaf"]:  # if output_region isn't a leaf
            output_left_child = product_recur(*output_region_dict["left_children"])
            output_right_child = product_recur(*output_region_dict["right_children"])
            output_region_dict["output_left_child"] = output_left_child
            output_region_dict["output_right_child"] = output_right_child

        output_region = output_vtree.lazy_prod_execute_update(output_region_dict)
        return output_region



    root1 = vtree1.root
    root2 = vtree2.root

    output_root = product_recur(root1, root2)
    output_vtree.set_root(output_root)


    return output_vtree


def conditional(vtree, cond_set):
    """Implements the conditional of a LazyMDNet on a conditioning set cond_set.

    Recursively traverses the LazyMDNet from the root. If the cond_set contains the mdset of the vtree node, then we
    set weights to 1 ("cond" operation). Otherwise, if cond_set is disjoint from the scope of the vtree node, we
    normalize the weights (and coefficients) using the values obtained from a forward pass. If neither of these hold,
    then the circuit is not MD with respect to the conditioning set and so we raise an Exception.

    :param vtree: the LazyMDNet to be conditioned
    :param cond_set: the variables to condition on
    :return: conditional LazyMDNet
    """

    node_values = vtree.forward(inputs={}, return_all=True)
    node_values = {region: node_values[region].squeeze(axis=0) for region in node_values} # remove batch dimension

    output_vtree = type(vtree)(scope=vtree.scope)

    def conditional_recur(region, active):
        if vtree.region_labels[region].is_marginal_deterministic(cond_set):
            mode = "cond"
        elif not cond_set.intersection(vtree.region_scopes[region]):
            mode = "norm"
        else:
            raise ValueError("cannot compute conditional; circuit is not marginal deterministic wrt conditioning set")

        if not active:
            mode = None

        output_region_dict = vtree.cond_compute_region_dict_tmp(region=region, mode=mode,
                                                            norm_consts = node_values[region],
                                                                cond_set = cond_set,
                                                                node_values=node_values)

        children_active = active and (mode != "norm")  # if norm, make it inactive

        if not output_region_dict["leaf"]:
            output_left_child = conditional_recur(output_region_dict["left_child"], active=children_active)
            output_right_child = conditional_recur(output_region_dict["right_child"], active=children_active)
            output_region_dict["output_left_child"] = output_left_child
            output_region_dict["output_right_child"] = output_right_child

        output_region = output_vtree.cond_execute_update(output_region_dict)

        return output_region

    output_root = conditional_recur(vtree.root, active=True)
    output_vtree.set_root(output_root)

    return output_vtree




