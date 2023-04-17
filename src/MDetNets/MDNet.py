from src.MDetNets.MDVtree import MDVtree, VtreeLabel
from src.MDetNets.leaves import FactorizedBinaryLeafOpRegion, FactorizedBinaryDenseLeafRegion#LeafRegion, MargLeafRegion, ProductLeafRegion
from src.MDetNets.weights import GeneralizedSparseWeights, GeneralizedDenseWeights, GeneralizedDenseLinearWeights
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
from functools import partial
from jax.tree_util import register_pytree_node_class
import uuid
import functools
from src.MDetNets.utils import random_scope_partition




# Registering class as PyTree allows us to mark the structure as static for jitting. We assume that the structure
# does not change once set (otherwise we might run into problems, c.f.
# https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
@register_pytree_node_class
class LazyMDNet(MDVtree):
    """Class representing a MDNet, which is a MDVtree augmented with sum nodes inside each region.

    Lazy version, meaning that operations such as marginalization and product are performed without changing the structure
    of the MDNet.

    Currently, we must restrict to binary vtrees and leaf regions over single variables.

    Attributes:
        - _region_shapes: dict mapping region name to the shape of the sum nodes in the region
        - _region_weights: dict mapping region name to the GeneralizedSparseWeight associated with that region (if
        not a leaf region)
        - _leaf_regions: dict mapping region name to the LeafRegion associated with that region (if a leaf
        region).

    """
    def __init__(self, scope):
        super().__init__(scope)
        self._region_shapes = {} # region -> tuple
        self._region_weights = {} # region -> GeneralizedSparseWeights
        self._leaf_regions = {} # region -> LeafRegion

    @property
    def region_shapes(self):
        return self._region_shapes

    @property
    def region_weights(self):
        return self._region_weights

    @property
    def leaf_regions(self):
        return self._leaf_regions

    def create_vtree_region(self, scope, leaf,
                            name=None, label=VtreeLabel(VtreeLabel.universal_set),
                            region_shape=(10,)):
        """Creates a new lazy mdnet region with given scope, name and label, and adds it to the md-vtree.

        :param scope: iterable of elements defining the scope function for this md-vtree region
        :param leaf: true if the region is a leaf
        :param name: name of the new region (optional)
        :param label: label for the new region (universal set by default)
        :param region_shape: shape of the region to be created
        :return: newly created region name
        """
        new_region = super().create_vtree_region(scope, leaf=leaf, name=name, label=label)
        if leaf:
            self._leaf_regions[new_region] = None
        else:
            self._region_weights[new_region] = None

        self._region_shapes[new_region] = region_shape

        return new_region


    def generate_random_weights(self, key):
        """Generates a random GeneralizedSparseWeights for the LazyMDNet.

        For each region with children, generates weights by randomly generating  a weight and a combination (indicating which sum nodes from the child regions are
            connected to it). Both the weights and combinations are represented as arrays over the product nodes.
                stored in the SparseWeights object
        For 2), the randomization occurs by generating all combinations of the sum nodes in child regions, and then
        randomly shuffling to ensure the combinations occur in a random order. Further, a random normal weight is
        assigned,

        :param key: JAX PRNG key
        """

        for region in self.regions:
            if len(self.region_shapes[region]) != 1:
                raise RuntimeError("can only generate random weights for one circuit")  # can generate only when there is one circuit
            region_size = self.region_shapes[region][0]
            if not self.leaves[region]:
                self._region_weights[region] = GeneralizedSparseWeights.random_initialize(shape_params={
                    "region_size": region_size,
                    "children_region_sizes": [self.region_shapes[child][0] for child in self.children[region]]
                },
                    key=key
                )
            else:
                # Generate random leaf regions with the correct shape
                self._leaf_regions[region] = FactorizedBinaryLeafOpRegion.random_initialize(shape_params={
                    "region_size": region_size,
                    "vars": self.region_scopes[region]
                },
                    key=key
                )

    def tree_flatten(self):
        children = (self._region_weights, self._leaf_regions)
        aux_data = vars(self).copy() # all other attributes are static
        aux_data.pop('_region_weights')
        aux_data.pop('_leaf_regions')
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mdnet = cls(scope=aux_data["_scope"])
        mdnet._region_weights = children[0]
        mdnet._leaf_regions = children[1]
        for key in aux_data:
            setattr(mdnet, key, aux_data[key])
        return mdnet


    ######### OPERATIONS #############

    def lazy_marg_compute_region_dict(self, region, marg_set):
        """Computes information relating to the marginalized version of the given region, in the output vtree.

        We compute the new GeneralizedSparseWeight (if not leaf) and LeafRegion (if leaf) after marginalization.

        :param region: name of region in vtree to be marginalized
        :param marg_set: set of variables to be marginalized out
        :return: dictionary containing information for performing the update to the output vtree.
        """
        output_region_dict = super().lazy_marg_compute_region_dict(region, marg_set)

        if self.leaves[region]:
            if self.region_scopes[region].issubset(marg_set):  # if the leaf region is to be marginalized
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).marginalize(self.leaf_regions[region], marg_set)
            else:
                output_region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            output_region_dict["region_weights"] = self.region_weights[region]

        output_region_dict["create_args"]["region_shape"] = self.region_shapes[region]

        return output_region_dict

    def lazy_marg_execute_update(self, output_region_dict):
        """Given information relating to the marginalized version of the input region (obtained using
        lazy_marg_compute_region_dict), executes the update in this vtree.

        For MDVtrees, also initializes the weights/leaf details of the new region.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """
        output_region = super().lazy_marg_execute_update(output_region_dict)

        if output_region_dict["leaf"]:
            self.leaf_regions[output_region] = output_region_dict["leaf_region"]
        else:
            self.region_weights[output_region] = output_region_dict["region_weights"]

        return output_region

    def lazy_prod_compute_region_dict(self, region):
        """Compute information necessary for computing the product of the given region with another region.

        If region is None, imitates a leaf region with empty scope, emptysetlabel and a single trivial leaf node.

        :param region: name of region in vtree for which a product is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """
        region_dict = super().lazy_prod_compute_region_dict(region)

        if region is None:
            # "dummy region"
            region_dict["region_shape"] = (1,)
        else:
            region_dict["region_shape"] = self.region_shapes[region]

        if region_dict["is_leaf"]:
            if region is None:
                region_dict["leaf_region"] = FactorizedBinaryLeafOpRegion.trivial_leaf_region(n_dims=len(region_dict["region_shape"]))
            else:
                region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            region_dict["region_weights"] = self.region_weights[region]

        return region_dict

    def lazy_prod_compute_output_region_dict(self, region_dict1, region_dict2):
        """Compute information necessary for computing the product of two regions, where we have information from each
        of the individual reasons from lazy_prod_compute_region_dict.

        For LazyMDNets, we track the new region_shape as well as weight/leaf details.

        :param region_dict1, region_dict2: dicts with information on the individual reginos
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = super().lazy_prod_compute_output_region_dict(region_dict1, region_dict2)

        output_region_dict["create_args"]["region_shape"] = region_dict1["region_shape"] + region_dict2["region_shape"]

        if output_region_dict["is_leaf"]:
            output_region_dict["leaf_region"] = FactorizedBinaryLeafOpRegion.product(
                region_dict1['leaf_region'],
                region_dict2['leaf_region']
            )
        else:
            csv = region_dict1["circuit_scope"].intersection(region_dict2["circuit_scope"])

            # check if the scopes of the children of the regions match
            if (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (
                    region_dict1["left_child_scope"].intersection(csv) == region_dict2["left_child_scope"].intersection(
                csv) and
                    region_dict1["right_child_scope"].intersection(csv) == region_dict2[
                        "right_child_scope"].intersection(csv)):
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    region_dict1['region_weights'],
                    region_dict2['region_weights']
                )
            elif (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (region_dict1["left_child_scope"].intersection(csv) == region_dict2[
                "right_child_scope"].intersection(csv) and
                         region_dict1["right_child_scope"].intersection(csv) == region_dict2[
                             "left_child_scope"].intersection(csv)):
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    region_dict1['region_weights'],
                    region_dict2['region_weights'].swap_children()
                )
            # check if one region's scope matches the scope of a child of the other region
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "left_child_scope"].intersection(csv):
                # dummy weight
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    GeneralizedSparseWeights.identity_with_child_shape(
                        region_dict1["region_shape"]),
                    region_dict2['region_weights']
                )
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "right_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    GeneralizedSparseWeights.identity_with_child_shape(
                        region_dict1["region_shape"]).swap_children(),
                    region_dict2['region_weights']
                )
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "left_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    region_dict1['region_weights'],
                    GeneralizedSparseWeights.identity_with_child_shape(
                        region_dict2["region_shape"])
                )
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "right_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = GeneralizedSparseWeights.product(
                    region_dict1['region_weights'],
                    GeneralizedSparseWeights.identity_with_child_shape(
                        region_dict2["region_shape"]).swap_children()
                )

        return output_region_dict


    def lazy_prod_execute_update(self, output_region_dict):
        """Removes a md-vtree region with given name and detaches it from its parents/children.

        :param name: name of vtree region to be removed
        :return: newly created region name
        """
        output_region = super().lazy_prod_execute_update(output_region_dict)

        if output_region_dict["is_leaf"]:
            self._leaf_regions[output_region] = output_region_dict["leaf_region"]
        else:
            self._region_weights[output_region] = output_region_dict["region_weights"]

        return output_region


    def cond_compute_region_dict(self, region, mode = None, norm_consts = None):
        """Compute information necessary for computing the conditional of a region.

        :param region: name of region in vtree for which a conditional is to be computed
        :param node: either "norm", "cond" or None. these correspond to different operations. "norm" divides the
            coefficients of the leaf/sum region by norm_consts. "cond" sets all coefficients and weights (that are
            non-zero, i.e. in the support) to 1. None simply performs the identity operation.
        :param norm_consts: value to divide by
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = super().cond_compute_region_dict(region)

        if output_region_dict["leaf"]:
            if mode == "cond":
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).uniformize(self.leaf_regions[region])
            elif mode == "norm":
                if norm_consts is None:
                    raise ValueError("for mode \"norm\" must provide norm_consts")
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).multiply(self.leaf_regions[region],
                                                                                          1/norm_consts)  # divide
            else:
                output_region_dict["leaf_region"] = self.leaf_regions[region]

        else:
            if mode == "cond":
                output_region_dict["region_weights"] = type(self.region_weights[region]).uniformize(self.region_weights[region])
            elif mode == "norm":
                if norm_consts is None:
                    raise ValueError("for mode \"norm\" must provide norm_consts")
                output_region_dict["region_weights"] = type(self.region_weights[region]).multiply(self.region_weights[region],
                                                                                           1/norm_consts)
            else:
                output_region_dict["region_weights"] = self.region_weights[region]


        output_region_dict["create_args"]["region_shape"] = self.region_shapes[region]

        return output_region_dict


    def cond_execute_update(self, output_region_dict):
        """Given information relating to the conditional region (obtained using cond_compute_region_dict),
        executes the update in this vtree.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """
        output_region = super().cond_execute_update(output_region_dict)

        # set equal to previous weights
        if output_region_dict["leaf"]:
            self._leaf_regions[output_region] = output_region_dict["leaf_region"]
        else:
            self._region_weights[output_region] = output_region_dict["region_weights"]

        return output_region

    ####################### FORWARD ######################################


    #@jit
    #@partial(jit, static_argnums=[2, ])
    def forward_log(self, inputs, return_all=False):
        """Jitted function which computes a forward pass through the MDNetStructure.

        :param inputs: dictionary mapping the variables to their input values
        :param all_weights: dictionary mapping non-leaf regions to their weights
        :param all_child_combinations: dictionary mapping non-leaf regions to the combinations of their product nodes
        :return: the output of the root region
        """
        layers = self.make_layers()
        values = {}
        #batch_size = inputs["batch_size"] #list(inputs.values())[0].shape[0]
        for layer in reversed(layers):
            for region in layer:
                # leaf region
                if not self.children[region]:

                    assert len(self.region_scopes[region]) == 0 or len(self.region_scopes[region]) == 1
                    values[region] = self.leaf_regions[region].forward_log(inputs)
                # non leaf region
                else:
                    if len(self.children[region]) != 2:
                        raise NotImplementedError("only binary vtrees currently supported")
                    left_child, right_child = self.children[region][0], self.children[region][1]

                    # dimension (batch_size, k_(l, 1), ..., k_(l, n)) where n is the number of circuits that have been multiplied
                    # similar (batch_size, k_(r, 1), ..., k_(r, n)).
                    left_child_values, right_child_values = values[left_child], values[right_child]
                    values[region] = self.region_weights[region].forward_log([left_child_values, right_child_values])

        if return_all:
            return values
        else:
            return values[self.root]

    def forward(self, inputs, return_all=False):
        result = self.forward_log(inputs, return_all=return_all)
        if return_all:
            return {region: jnp.exp(result[region]) for region in result}

        else:
            return jnp.exp(result)

@register_pytree_node_class
class LazyDenseMDNet(LazyMDNet):
    def __init__(self, scope):
        super().__init__(scope)
        self._group_sizes={} # region -> tuple

    @property
    def group_sizes(self):
        return self._group_sizes

    def create_vtree_region(self, scope, leaf,
                            name=None, label=VtreeLabel(VtreeLabel.universal_set),
                            region_shape=(10,), group_size=(2,)):
        """Creates a new lazy mdnet region with given scope, name and label, and adds it to the md-vtree.

        :param scope: iterable of elements defining the scope function for this md-vtree region
        :param leaf: true if the region is a leaf
        :param name: name of the new region (optional)
        :param label: label for the new region (universal set by default)
        :param region_shape: shape of the region to be created
        :param group_size: group sizes for each dimension of region_shape
        :return: newly created region name
        """
        new_region = super().create_vtree_region(scope, leaf=leaf, name=name, label=label, region_shape=region_shape)

        self._group_sizes[new_region] = group_size

        return new_region

    def generate_random_binary_vtree(self, region_shape=(10,), group_size=(2,), leaf_region_shape=(10,), leaf_group_size=(2,)):
        """Generates a random, balanced binary vtree, by recursively and randomly splitting each scope into near-halves.

        :return: list of layers following the generated structure
        """
        if self.root is None:
            root_region = self.create_vtree_region(self.scope, leaf=False, region_shape = (1,), group_size = (1,))
            self.set_root(root_region)
        layers = [[self.root,]]
        while layers[-1]:
            new_layer = []
            for region_name in layers[-1]:
                region_scope = self.region_scopes[region_name]
                if len(region_scope) > 1:
                    scope_partition = random_scope_partition(region_scope, 2)
                    children_names = []
                    for sub_scope in scope_partition:
                        if len(sub_scope) == 1:
                            # leaf
                            child_name = self.create_vtree_region(scope=sub_scope, name=sub_scope, leaf=True,
                                                                  region_shape=leaf_region_shape, group_size=leaf_group_size)
                        else:
                            child_name = self.create_vtree_region(scope=sub_scope, name=sub_scope, leaf=False,
                                                                  region_shape=region_shape, group_size=group_size)
                        children_names.append(child_name)
                    self.update_children(region_name, children_names)  # connect new regions
                    new_layer += scope_partition
            layers.append(new_layer)

        return layers[:-1]

    def generate_staged_binary_vtree(self, var_stages, region_shape=(10,), group_size=(2,), leaf_region_shape=(10,), leaf_group_size=(2,)):
        """Temporary code to generate a staged (i.e. PSDD like, but deterministic left branches) PSDD
        Should replace with proper code that generates the necessary vtree automatically.
        """
        if self.root is None:
            root_region = self.create_vtree_region(self.scope, leaf=False, region_shape = (1,), group_size = (1,),
                                                   label=VtreeLabel(var_stages[0]))
            self.set_root(root_region)

        def stage_recur(region_name, required_left_scope_idx):
            region_scope = self.region_scopes[region_name]
            if len(region_scope) > 1:
                if required_left_scope_idx is None:
                    scope_partition = random_scope_partition(region_scope, 2)
                    children_names = []
                    for sub_scope in scope_partition:
                        if len(sub_scope) == 1:
                            # leaf
                            child_name = self.create_vtree_region(scope=sub_scope, name=sub_scope, leaf=True,
                                                                  region_shape=leaf_region_shape,
                                                                  group_size=leaf_group_size,
                                                                  label=VtreeLabel(sub_scope))
                        else:
                            child_name = self.create_vtree_region(scope=sub_scope, name=sub_scope, leaf=False,
                                                                  region_shape=region_shape, group_size=group_size,
                                                                  label=VtreeLabel(sub_scope))
                        children_names.append(child_name)
                        stage_recur(child_name, None)
                    self.update_children(region_name, children_names)  # connect new regions
                else:
                    required_left_scope = var_stages[required_left_scope_idx]
                    right_scope = region_scope.difference(required_left_scope)

                    # LEFT CHILD
                    if len(required_left_scope) == 1:
                        left_child_name = self.create_vtree_region(scope=required_left_scope, name=required_left_scope, leaf=True,
                                                                   label=VtreeLabel(required_left_scope), # left child is det
                                                                  region_shape=leaf_region_shape,
                                                                  group_size=leaf_group_size)
                    else:
                        left_child_name = self.create_vtree_region(scope=required_left_scope, name=required_left_scope, leaf=False,
                                                                   label=VtreeLabel(required_left_scope), # left child is det
                                                                  region_shape=region_shape,
                                                                  group_size=group_size)
                    stage_recur(left_child_name, None)

                    next_required_idx = required_left_scope_idx + 1
                    if next_required_idx < len(var_stages):
                        right_label = VtreeLabel(var_stages[next_required_idx])
                    else:
                        right_label = VtreeLabel(right_scope)
                        next_required_idx = None
                    if len(right_scope) == 1:
                        right_child_name = self.create_vtree_region(scope=right_scope, name=right_scope, leaf=True,
                                                                   label=right_label, # left child is det
                                                                  region_shape=leaf_region_shape,
                                                                  group_size=leaf_group_size)
                    else:
                        right_child_name = self.create_vtree_region(scope=right_scope, name=right_scope, leaf=False,
                                                                   label=right_label,
                                                                  region_shape=region_shape,
                                                                  group_size=group_size)
                    stage_recur(right_child_name, next_required_idx)

                    children_names = [left_child_name, right_child_name]

                self.update_children(region_name, children_names)

        stage_recur(self.root, 0)


    def generate_random_weights(self, key, max_products=100, use_linear=False):
        """ TBC

        :param key: JAX PRNG key
        """
        for region in self.regions:
            if len(self.region_shapes[region]) != 1:
                raise RuntimeError("can only generate random weights for one circuit")  # can generate only when there is one circuit
            region_size = self.region_shapes[region][0]
            key, subkey = random.split(key)
            if not self.leaves[region]:
                if len(self.children[region]) != 2:
                    raise RuntimeError("dense lazymdnet only works for binary vtrees")
                left_child = self.children[region][0]
                right_child = self.children[region][1]
                if use_linear:
                    WeightClass = GeneralizedDenseLinearWeights
                else:
                    WeightClass = GeneralizedDenseWeights
                self._region_weights[region] = WeightClass.random_initialize(shape_params={
                    "region_size": region_size,
                    "group_size": self.group_sizes[region][0],
                    "left_child_region_size": self.region_shapes[left_child][0],
                    "right_child_region_size": self.region_shapes[right_child][0],
                    "num_products": min(max_products, self.region_shapes[left_child][0] * self.region_shapes[right_child][0]),
                    "left_child_group_size": self.group_sizes[left_child][0],
                    "right_child_group_size": self.group_sizes[right_child][0]
                },
                    key=subkey
                )
            else:
                # Generate random leaf regions with the correct shape
                self._leaf_regions[region] = FactorizedBinaryDenseLeafRegion.random_initialize(shape_params={
                    "vars": self.region_scopes[region],
                    "region_size": region_size,
                    "group_size": self.group_sizes[region][0]
                },
                    key=subkey
                )

    def generate_random_det_weights(self, key, max_products=100, use_linear=False, respect_labels=False):
        """ Ensures determinism

        :param key: JAX PRNG key
        """
        for region in self.regions:
            if len(self.region_shapes[region]) != 1:
                raise RuntimeError("can only generate random weights for one circuit")  # can generate only when there is one circuit
            region_size = self.region_shapes[region][0]
            if not self.leaves[region]:
                key, subkey = random.split(key)
                if len(self.children[region]) != 2:
                    raise RuntimeError("dense lazymdnet only works for binary vtrees")
                left_child = self.children[region][0]
                right_child = self.children[region][1]

                if use_linear:
                    WeightClass = GeneralizedDenseLinearWeights
                else:
                    WeightClass = GeneralizedDenseWeights

                if respect_labels:
                    region_label = self.region_labels[region]
                    left_child_label, right_child_label = self.region_labels[left_child], self.region_labels[right_child]
                    if region_label == VtreeLabel.union(left_child_label, right_child_label):
                        str_det = None
                    elif region_label == left_child_label:
                        str_det = "left"
                    elif region_label == right_child_label:
                        str_det = "right"
                    else:
                        raise ValueError("cannot generate weights as md-vtree is not optimal")
                else:
                    str_det = None

                self._region_weights[region] = WeightClass.random_det_initialize(shape_params={
                    "region_size": region_size,
                    "group_size": self.group_sizes[region][0],
                    "left_child_region_size": self.region_shapes[left_child][0],
                    "right_child_region_size": self.region_shapes[right_child][0],
                    "num_products": min(max_products, self.region_shapes[left_child][0] * self.region_shapes[right_child][0]),
                    "left_child_group_size": self.group_sizes[left_child][0],
                    "right_child_group_size": self.group_sizes[right_child][0]
                },
                    key=subkey,
                    str_det=str_det
                )
            else:
                # Generate random leaf regions with the correct shape
                self._leaf_regions[region] = FactorizedBinaryDenseLeafRegion.det_initialize(shape_params={
                    "vars": self.region_scopes[region],
                    "region_size": region_size,
                    "group_size": self.group_sizes[region][0]
                }
                )



    def forward_log(self, inputs, return_all=False, return_counts=False):
        """Return counts (number of data entries for each node within THE PRODUCT GROUP THAT IS THE CHILD OF THE
        GIVEN REGION) also.
        """
        layers = self.make_layers()
        values = {}
        product_counts = {}
        for layer in reversed(layers):
            for region in layer:
                # leaf region
                if not self.children[region]:

                    assert len(self.region_scopes[region]) == 0 or len(self.region_scopes[region]) == 1
                    values[region] = self.leaf_regions[region].forward_log(inputs)
                # non leaf region
                else:
                    if len(self.children[region]) != 2:
                        raise NotImplementedError("only binary vtrees currently supported")
                    left_child, right_child = self.children[region][0], self.children[region][1]

                    # dimension (batch_size, k_(l, 1), ..., k_(l, n)) where n is the number of circuits that have been multiplied
                    # similar (batch_size, k_(r, 1), ..., k_(r, n)).
                    left_child_values, right_child_values = values[left_child], values[right_child]
                    if return_counts:
                        values[region], product_counts[region] = self.region_weights[region].forward_log([left_child_values, right_child_values], return_counts=True)
                    else:
                        values[region] = self.region_weights[region].forward_log([left_child_values, right_child_values], return_counts=False)

        if return_all:
            if return_counts:
                return values, product_counts
            else:
                return values
        else:
            if return_counts:
                return values[self.root], product_counts
            else:
                return values[self.root]

    def forward(self, inputs, return_all=False, return_counts=False):
        if return_counts:
            values, product_counts = self.forward_log(inputs, return_all, return_counts)
            if return_all:
                return {region: jnp.exp(values[region]) for region in values}, product_counts
            else:
                return jnp.exp(values), product_counts
        else:
            values = self.forward_log(inputs, return_all, return_counts)
            if return_all:
                return {region: jnp.exp(values[region]) for region in values}
            else:
                return jnp.exp(values)




    def update_em_det(self, product_counts):
        """outputs_batched is a list (over batches) of dicts, where each dict maps from region to the value of that
        region.
        """
        for region in self.regions:
            if not self.leaves[region]:
                self.region_weights[region].update_em_det(product_counts[region])



    ########################################################
    # OPERATIONS FOR DenseMdNet (i.e. taking group in to account)

    def lazy_marg_compute_region_dict(self, region, marg_set):
        """Computes information relating to the marginalized version of the given region, in the output vtree.

        We compute the new GeneralizedSparseWeight (if not leaf) and LeafRegion (if leaf) after marginalization.

        :param region: name of region in vtree to be marginalized
        :param marg_set: set of variables to be marginalized out
        :return: dictionary containing information for performing the update to the output vtree.
        """
        output_region_dict = MDVtree.lazy_marg_compute_region_dict(self,region, marg_set)

        if self.leaves[region]:
            if self.region_scopes[region].issubset(marg_set):  # if the leaf region is to be marginalized
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).marginalize(
                    self.leaf_regions[region], marg_set)
                # output_region_dict["leaf_region"] = MargLeafRegion(self.leaf_regions[region])
            else:
                output_region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            output_region_dict["region_weights"] = self.region_weights[region]

        output_region_dict["create_args"]["region_shape"] = self.region_shapes[region]
        output_region_dict["create_args"]["group_size"] = self.group_sizes[region]

        return output_region_dict

    def lazy_prod_compute_region_dict(self, region):
        """Compute information necessary for computing the product of the given region with another region.

        If region is None, imitates a leaf region with empty scope, emptysetlabel and a single trivial leaf node.

        :param region: name of region in vtree for which a product is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """
        region_dict = MDVtree.lazy_prod_compute_region_dict(self, region)

        if region is None:
            # "dummy region"
            assert False
            region_dict["region_shape"] = (1,)
        else:
            region_dict["region_shape"] = self.region_shapes[region]
            region_dict["group_size"] = self.group_sizes[region]

        if region_dict["is_leaf"]:
            if region is None:
                region_dict["leaf_region"] = FactorizedBinaryLeafOpRegion.trivial_leaf_region(n_dims=len(region_dict["region_shape"]))
                #region_dict["leaf_region"] = LeafRegion.trivial_leaf_region()
            else:
                region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            region_dict["region_weights"] = self.region_weights[region]

        return region_dict

    def lazy_prod_compute_output_region_dict(self, region_dict1, region_dict2):
        """Compute information necessary for computing the product of two regions, where we have information from each
        of the individual reasons from lazy_prod_compute_region_dict.

        For LazyMDNets, we track the new region_shape as well as weight/leaf details.

        :param region_dict1, region_dict2: dicts with information on the individual reginos
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = MDVtree.lazy_prod_compute_output_region_dict(self, region_dict1, region_dict2)

        output_region_dict["create_args"]["region_shape"] = region_dict1["region_shape"] + region_dict2["region_shape"]
        output_region_dict["create_args"]["group_size"] = region_dict1["group_size"] + region_dict2["region_shape"]

        if output_region_dict["is_leaf"]:
            output_region_dict["leaf_region"] = type(region_dict1['leaf_region']).product(
                region_dict1['leaf_region'],
                region_dict2['leaf_region']
            )
        else:
            csv = region_dict1["circuit_scope"].intersection(region_dict2["circuit_scope"])

            # check if the scopes of the children of the regions match
            if (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (
                    region_dict1["left_child_scope"].intersection(csv) == region_dict2["left_child_scope"].intersection(
                csv) and
                    region_dict1["right_child_scope"].intersection(csv) == region_dict2[
                        "right_child_scope"].intersection(csv)):
                output_region_dict["region_weights"] = type(region_dict1['region_weights']).product(
                    region_dict1['region_weights'],
                    region_dict2['region_weights']
                )
            elif (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (region_dict1["left_child_scope"].intersection(csv) == region_dict2[
                "right_child_scope"].intersection(csv) and
                         region_dict1["right_child_scope"].intersection(csv) == region_dict2[
                             "left_child_scope"].intersection(csv)):
                output_region_dict["region_weights"] = type(region_dict1['region_weights']).product(
                    region_dict1['region_weights'],
                    region_dict2['region_weights'].swap_children()
                )
            # check if one region's scope matches the scope of a child of the other region
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "left_child_scope"].intersection(csv):
                # dummy weight
                output_region_dict["region_weights"] = type(region_dict2['region_weights']).product(
                    type(region_dict2['region_weights']).identity_with_child_shape(
                        region_dict1["region_shape"]),
                    region_dict2['region_weights']
                )
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "right_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = type(region_dict2['region_weights']).product(
                    type(region_dict2['region_weights']).identity_with_child_shape(
                        region_dict1["region_shape"]).swap_children(),
                    region_dict2['region_weights']
                )
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "left_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = type(region_dict1['region_weights']).product(
                    region_dict1['region_weights'],
                    type(region_dict1['region_weights']).identity_with_child_shape(
                        region_dict2["region_shape"])
                )
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "right_child_scope"].intersection(csv):
                output_region_dict["region_weights"] = type(region_dict1['region_weights']).product(
                    region_dict1['region_weights'],
                    type(region_dict1['region_weights']).identity_with_child_shape(
                        region_dict2["region_shape"]).swap_children()
                )

        return output_region_dict



    def cond_compute_region_dict(self, region, mode = None, norm_consts = None):
        """Compute information necessary for computing the conditional of a region.

        :param region: name of region in vtree for which a conditional is to be computed
        :param node: either "norm", "cond" or None. these correspond to different operations. "norm" divides the
            coefficients of the leaf/sum region by norm_consts. "cond" sets all coefficients and weights (that are
            non-zero, i.e. in the support) to 1. None simply performs the identity operation.
        :param norm_consts: value to divide by
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = MDVtree.cond_compute_region_dict(self, region)

        if output_region_dict["leaf"]:
            if mode == "cond":
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).uniformize(self.leaf_regions[region])
            elif mode == "norm":
                if norm_consts is None:
                    raise ValueError("for mode \"norm\" must provide norm_consts")
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).multiply(self.leaf_regions[region],
                                                                                          1/norm_consts)  # divide
            else:
                output_region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            if mode == "cond":
                output_region_dict["region_weights"] = type(self.region_weights[region]).uniformize(self.region_weights[region])
            elif mode == "norm":
                if norm_consts is None:
                    raise ValueError("for mode \"norm\" must provide norm_consts")
                output_region_dict["region_weights"] = type(self.region_weights[region]).multiply(self.region_weights[region],
                                                                                           1/norm_consts)
            else:
                output_region_dict["region_weights"] = self.region_weights[region]

        output_region_dict["create_args"]["region_shape"] = self.region_shapes[region]
        output_region_dict["create_args"]["group_size"] = self.group_sizes[region]

        return output_region_dict


    def cond_compute_region_dict_tmp(self, region, mode = None, norm_consts = None, cond_set=None,
                                     node_values=None):

        output_region_dict = MDVtree.cond_compute_region_dict(self, region)

        if output_region_dict["leaf"]:
            if mode == "cond":
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).uniformize(
                    self.leaf_regions[region])
            elif mode == "norm":
                if norm_consts is None:
                    raise ValueError("for mode \"norm\" must provide norm_consts")
                output_region_dict["leaf_region"] = type(self.leaf_regions[region]).multiply(self.leaf_regions[region],
                                                                                             1 / norm_consts)  # divide
            else:
                output_region_dict["leaf_region"] = self.leaf_regions[region]
        else:
            left_child, right_child = self.children[region][0], self.children[region][1]
            region_label = self.region_labels[region]
            left_child_label, right_child_label = self.region_labels[left_child], self.region_labels[right_child]
            if self.region_labels[region].is_marginal_deterministic(cond_set):
                if region_label == VtreeLabel.union(left_child_label, right_child_label):
                    # proceed as usual for cond
                    output_region_dict["region_weights"] = type(self.region_weights[region]).uniformize(
                        self.region_weights[region])
                elif region_label == left_child_label:
                    if not cond_set.intersection(self.region_scopes[right_child]):
                        output_region_dict["region_weights"] = type(self.region_weights[region]).uniformize_left_with_right_child_values(
                            self.region_weights[region], right_child_log_values = jnp.log(node_values[right_child]))
                    else:
                        output_region_dict["region_weights"] = type(
                            self.region_weights[region]).uniformize_left(
                            self.region_weights[region])

                else:
                    raise NotImplementedError
            elif not cond_set.intersection(self.region_scopes[region]):
                output_region_dict["region_weights"] = self.region_weights[region]
                # do nothing

        output_region_dict["create_args"]["region_shape"] = self.region_shapes[region]
        output_region_dict["create_args"]["group_size"] = self.group_sizes[region]

        return output_region_dict

    def cond_execute_update(self, output_region_dict):
        """Given information relating to the conditional region (obtained using cond_compute_region_dict),
        executes the update in this vtree.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """
        output_region = MDVtree.cond_execute_update(self, output_region_dict)

        # set equal to previous weights
        if output_region_dict["leaf"]:
            self._leaf_regions[output_region] = output_region_dict["leaf_region"]
        else:
            self._region_weights[output_region] = output_region_dict["region_weights"]

        return output_region