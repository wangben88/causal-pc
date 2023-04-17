from jax import random
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jax import vmap
import warnings
from src.MDetNets.utils import tensordot_logdomain, interleave
from jax import random

@register_pytree_node_class
class FactorizedBinaryLeafOpRegion():
    """Leaf region representing a factorized (unnormalized) distribution over binary variables.

    All operations are implemented in a purely functional manner; that is, they do not modify self in any way.

    Attributes:
        - region_shape: tuple indicating the shape of the leaf region
        - vars: list of variables that the region covers
        - params: dict, mapping from variable name to an array of shape (2, region_shape)
            the first dimension is there for probs_0 vs probs_1,
            also, have "coefficients" which is of shape (region_shape) and represents marginalization coefficients
            which are applied regardless of the input.
    """
    def __init__(self, region_shape, vars, params):
        self.region_shape = region_shape
        self.vars = vars  # set
        self.params = params  # dict

    @staticmethod
    def random_initialize(shape_params, key):
        """Provides a random initialization of the leaf region, given some shape_params detailing the
        relevant dimensions.

        shape_params is a dict containing the following keys
        - "vars": variables in this leaf region
        - "region_size": size of this region
        """

        params = {"log_coefficients": jnp.zeros(shape=shape_params["region_size"])}

        for var in shape_params["vars"]:
            key, subkey = random.split(key)
            random_probs = random.uniform(key=subkey, shape=[shape_params["region_size"]])
            params[var] = jnp.stack([jnp.log(1-random_probs), jnp.log(random_probs)], axis=0)

        return FactorizedBinaryLeafOpRegion(
            region_shape=(shape_params["region_size"],),
            vars=shape_params["vars"],
            params = params
        )
    def tree_flatten(self):
        children = (self.params,)
        aux_data = {"region_shape": self.region_shape,
                    "vars": self.vars}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(params=children[0], **aux_data)

    @staticmethod
    def single_var_leaf(var, probs):
        """Returns a leaf region over a single variable with given probabilities.

        :param var: variable name
        :param probs: array of probabilities of shape region_shape
        :return: the output region
        """
        return FactorizedBinaryLeafOpRegion(
            region_shape=probs.shape,
            vars={var},
            params = {
                "log_coefficients": jnp.zeros_like(probs),
                var: jnp.stack([jnp.log(1-probs), jnp.log(probs)], axis=0)
            }
        )

    @staticmethod
    def trivial_leaf_region(n_dims):
        """Returns a "trivial" leaf region, which has no variables and consists of a single leaf node, along n
        dimensions, and with value 1 under all inputs.

        NB. The key property of this region is that when multiplied by another region, it merely expands the dimensions
        of the other region.
        As we have incorporated the expansion whenever a variable is not present in one of the regions to be producted,
        we leave the vars empty, and params empty here, besides the coefficients.

        :param n_dims: number of dimensions for the trivial region
        :return: the output region
        """
        params = {"log_coefficients": jnp.zeros(shape=[1]*n_dims)}
        return FactorizedBinaryLeafOpRegion(
            region_shape=(1,)*n_dims,
            vars=set(),
            params=params
        )

    def forward_log(self, batched_input):
        """Given batched values for each of the variables, computes the output value for each of the leaves.

        Now, by default, if a value is not provided for some variable, then we assume it is marginalized out.

        :param batched_input: dict mapping variable to array of shape (batch_size,)
        :return: output, which is array of shape (batch_size, self.region_shape)
        """
        if len(batched_input) > 0:
            batch_size = next(iter(batched_input.values())).shape[0]
        else:
            batch_size = 1
        batched_log_output_variable_list = []
        for var in self.vars:
            log_probs_0 = self.params[var][0, ...]
            log_probs_1 = self.params[var][1, ...]
            #
            if var in batched_input:
                # Again, no tensordot for log domain so we do it manually.
                batched_log_output_var = jnp.logaddexp(
                    tensordot_logdomain(jnp.log(batched_input[var]), log_probs_1),
                    tensordot_logdomain(jnp.log(1-batched_input[var]), log_probs_0)
                )
                # batched_output_var = jnp.tensordot(batched_input[var], probs_1, axes=0) \
                #                      + jnp.tensordot(1-batched_input[var], probs_0, axes=0)  # (batch_size, ) . region_shape -> (batch_size, region_shape)
                batched_log_output_variable_list.append(batched_log_output_var)
            else:
                batched_log_output_var = jnp.stack([jnp.logaddexp(log_probs_0, log_probs_1)] * batch_size, axis=0) # region_shape -> (batch_size, region_shape)
                batched_log_output_variable_list.append(batched_log_output_var)

        #batched_output_variable_list[0].shape[0]

        batched_log_output_variable_list.append(jnp.stack([self.params["log_coefficients"]] * batch_size))
        #print([bv.shape for bv in batched_output_variable_list])
        batched_output = jnp.sum(jnp.stack(batched_log_output_variable_list, axis=0), axis=0)  # prod over variables -> sum for log

        return batched_output

    def forward(self, batched_input):
        return jnp.exp(self.forward_log(batched_input))

    def multiply(self, value):
        """Generates a new GeneralizedSparseWeights which multiplies the coefficients by the given value.

        :param value: array of shape region_shape to be multiplied
        :return: multiplied GeneralizedSparseWeights
        """
        assert value.shape == self.region_shape

        new_vars = self.vars.copy()  # copy of list
        new_params = self.params.copy()  # copy of dict
        new_params["log_coefficients"] = new_params["log_coefficients"] + jnp.log(value)  # changing entry of copied dict, does not affect old dict

        return FactorizedBinaryLeafOpRegion(region_shape=self.region_shape,
                                            vars = new_vars,
                                            params=new_params)


    def uniformize(self):
        """Generates a new leaf region where coefficients are set to 1 and all non-zero probs are set to 1.

        :return: uniformized GeneralizedSparseWeights
        """
        new_params = {"log_coefficients": jnp.zeros_like(self.params["log_coefficients"])}

        for var in self.vars:
            region_log_probs_0 = self.params[var][0, ...]
            region_log_probs_1 = self.params[var][1, ...]

            uniformized_region_log_probs_0 = jnp.log(jnp.logical_not(jnp.isneginf(region_log_probs_0)).astype(jnp.float32))
            uniformized_region_log_probs_1 = jnp.log(jnp.logical_not(jnp.isneginf(region_log_probs_1)).astype(jnp.float32))

            new_params[var] = jnp.stack([uniformized_region_log_probs_0, uniformized_region_log_probs_1], axis=0)

        return FactorizedBinaryLeafOpRegion(region_shape=self.region_shape,
                                            vars=self.vars,
                                            params=new_params)

    @staticmethod
    def marginalize(region, marg_set):
        """Marginalizes a leaf region.

        Marginalization value gets absorbed into the coefficient (this will not be one if the region is not normalized,
        e.g. after a product).

        :param marg_set: set of variables to be marginalized out
        :return: the marginalized region.
        """
        # marginalizes by removing the marginalized variables, so that they don't get calculated in forward
        new_vars = region.vars.difference(marg_set)
        new_params = {var: region.params[var] for var in new_vars}

        new_marg_vars_log_coeff_list = [jnp.logaddexp(region.params[var][0, ...], region.params[var][1, ...]) for var in region.vars.intersection(marg_set)] # probs_0 + probs_1
        new_marg_vars_log_coeff_list.append(region.params["log_coefficients"])
        new_coeffs = jnp.sum(jnp.stack(new_marg_vars_log_coeff_list, axis=0),axis=0)
        new_params["log_coefficients"] = new_coeffs

        return FactorizedBinaryLeafOpRegion(region_shape=region.region_shape,vars=new_vars, params=new_params)

    @staticmethod
    def product(region_1, region_2):
        """Product of two leaf regions.

        If the variables have distinct scope, then for any variables which they do not share, the dimensions are
        expanded in order to ensure that all variables have the same param shape.

        :param region_1, region_2: regions to be producted
        :return: the product region
        """
        new_vars = region_1.vars.union(region_2.vars)
        new_params = dict()


        #no tensordot in log-domain
        new_params["log_coefficients"] = tensordot_logdomain(region_1.params["log_coefficients"], region_2.params["log_coefficients"])
        # new_params["coefficients"] = jnp.tensordot(region_1.params["coefficients"],
        #                                            region_2.params["coefficients"],
        #                                            axes=0)
        for var in new_vars:
            if var in region_1.vars:
                region_1_probs_0 = region_1.params[var][0, ...]
                region_1_probs_1 = region_1.params[var][1, ...]
            else:
                region_1_probs_0 = jnp.zeros(shape=region_1.region_shape)
                region_1_probs_1 = jnp.zeros(shape=region_1.region_shape)

            if var in region_2.vars:
                region_2_probs_0 = region_2.params[var][0, ...]
                region_2_probs_1 = region_2.params[var][1, ...]
            else:
                region_2_probs_0 = jnp.zeros(shape=region_2.region_shape)
                region_2_probs_1 = jnp.zeros(shape=region_2.region_shape)

            new_params[var] = jnp.stack(
                [tensordot_logdomain(region_1_probs_0, region_2_probs_0),
                 tensordot_logdomain(region_1_probs_1, region_2_probs_1)],
                axis=0
            )

            assert len(new_params[var].shape) == 1 + len(region_1.region_shape) + len(region_2.region_shape)  # +1 for the 0,1 probs dimension

        return FactorizedBinaryLeafOpRegion(region_shape=region_1.region_shape + region_2.region_shape,
                                            vars=new_vars,
                                            params=new_params)

@register_pytree_node_class
class FactorizedBinaryDenseLeafRegion:
    """Implements factorized leaves agai, but this time with group dimensions.

    """
    def __init__(self, region_shape, group_sizes, vars, params):
        self.region_shape = region_shape
        self.group_sizes = group_sizes
        self.vars = vars
        self.params = params

    def tree_flatten(self):
        children = (self.params,)
        aux_data = {"region_shape": self.region_shape,
                    "group_sizes": self.group_sizes,
                    "vars": self.vars}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(params=children[0], **aux_data)

    @staticmethod
    def random_initialize(shape_params, key):
        """Provides a random initialization of the leaf region, given some shape_params detailing the
        relevant dimensions.

        shape_params is a dict containing the following keys
        - "vars": variables in this leaf region
        - "region_size": size of this region
        - "group_size": group size for this region
        """

        params = {"log_coefficients": jnp.zeros(shape=(shape_params["region_size"], shape_params["group_size"]))}

        for var in shape_params["vars"]:
            key, subkey = random.split(key)
            random_probs = random.uniform(key=subkey, shape=(shape_params["region_size"], shape_params["group_size"]))
            params[var] = jnp.stack([jnp.log(1 - random_probs), jnp.log(random_probs)], axis=0)

        return FactorizedBinaryDenseLeafRegion(
            region_shape=(shape_params["region_size"],),
            group_sizes=(shape_params["group_size"],),
            vars=shape_params["vars"],
            params=params
        )

    @staticmethod
    def det_initialize(shape_params):
        """Provides a deterministic initialization of the leaf region, i.e. the nodes in a group range over the values
        of the variables with probability 1.

        shape_params is a dict containing the following keys
        - "vars": variables in this leaf region
        - "region_size": size of this region
        - "group_size": group size for this region
        """

        params = {"log_coefficients": jnp.zeros(shape=(shape_params["region_size"], shape_params["group_size"]))}

        if len(shape_params["vars"]) > 1:
            raise NotImplementedError("Deterministic initialization only implemented for single var leafs for now")
        if shape_params["group_size"] != 2:
            raise ValueError("Only group sizes of 2 supported for deterministic leaves")

        var = next(iter(shape_params["vars"]))
        params[var] = jnp.log(jnp.stack(
            [
                jnp.stack([jnp.zeros(shape=(shape_params["region_size"],)), jnp.ones(shape=(shape_params["region_size"],))], axis=1),
                jnp.stack([jnp.ones(shape=(shape_params["region_size"],)), jnp.zeros(shape=(shape_params["region_size"],))],
                      axis=1)
            ],
            axis=0
        ))
        #print(var)

        return FactorizedBinaryDenseLeafRegion(
            region_shape=(shape_params["region_size"],),
            group_sizes=(shape_params["group_size"],),
            vars=shape_params["vars"],
            params=params
        )

    def forward_log(self, batched_input):
        """Given batched values for each of the variables, computes the output value for each of the leaves.

        Now, by default, if a value is not provided for some variable, then we assume it is marginalized out.

        NOTE: no changes needed from the sparse version

        :param batched_input: dict mapping variable to array of shape (batch_size,)
        :return: output, which is array of shape (batch_size, self.region_shape)
        """
        if len(batched_input) > 0:
            batch_size = next(iter(batched_input.values())).shape[0]
        else:
            batch_size = 1
        batched_log_output_variable_list = []
        for var in self.vars:
            log_probs_0 = self.params[var][0, ...]
            log_probs_1 = self.params[var][1, ...]
            #
            if var in batched_input:
                # Again, no tensordot for log domain so we do it manually.
                batched_log_output_var = jnp.logaddexp(
                    tensordot_logdomain(jnp.log(batched_input[var]), log_probs_1),
                    tensordot_logdomain(jnp.log(1-batched_input[var]), log_probs_0)
                )
                # batched_output_var = jnp.tensordot(batched_input[var], probs_1, axes=0) \
                #                      + jnp.tensordot(1-batched_input[var], probs_0, axes=0)  # (batch_size, ) . region_shape -> (batch_size, region_shape)
                batched_log_output_variable_list.append(batched_log_output_var)
            else:
                batched_log_output_var = jnp.stack([jnp.logaddexp(log_probs_0, log_probs_1)] * batch_size, axis=0) # region_shape -> (batch_size, region_shape)
                batched_log_output_variable_list.append(batched_log_output_var)

        #batched_output_variable_list[0].shape[0]


        batched_log_output_variable_list.append(jnp.stack([self.params["log_coefficients"]] * batch_size))
        #print([bv.shape for bv in batched_output_variable_list])
        batched_output = jnp.sum(jnp.stack(batched_log_output_variable_list, axis=0), axis=0)  # prod over variables -> sum for log
        batched_output = batched_output + jnp.expand_dims(self.params["log_coefficients"], axis=0)


        return batched_output

    def forward(self, batched_input):
        return jnp.exp(self.forward_log(batched_input))


    def update_em(self, grads_batched):
        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
            - k: group within region
            - g: node within group
            - p: ranges over product groups (N_L, N_R)
            - g_l: node within N_L
            - g_r: node within N_R
        """
        if len(self.region_shape) > 1:
            raise NotImplementedError("EM update for leaves only implemented for single circuits")
        if not jnp.all(jnp.isclose(self.params["log_coefficients"], 0.0)):
            raise NotImplementedError("EM update for leaves only implemented for log_coefficients equal to 0")
        if not len(self.vars) == 1:
            raise NotImplementedError("EM update for leaves only implemented for single variable leaves")


        for var in self.vars:
            grads_arr = jnp.sum(jnp.stack([grads.params[var] for grads in grads_batched]),
                                axis=0)
            grads_arr_norm = jnp.sum(grads_arr, axis=(0,), keepdims=True)
            self.params[var] = jnp.log(grads_arr+1e-8) - (jnp.log(grads_arr_norm+1e-8))


    ###### OPERATIONS ###################

    # no changes, except initialization of new object
    def multiply(self, value):
        """Generates a new GeneralizedSparseWeights which multiplies the coefficients by the given value.

        :param value: array of shape region_shape to be multiplied
        :return: multiplied GeneralizedSparseWeights
        """
        assert value.shape == self.region_shape

        new_vars = self.vars.copy()  # copy of list
        new_params = self.params.copy()  # copy of dict
        new_params["log_coefficients"] = new_params["log_coefficients"] + jnp.log(value)  # changing entry of copied dict, does not affect old dict

        return FactorizedBinaryDenseLeafRegion(region_shape=self.region_shape,
                                               group_sizes=self.group_sizes,
                                            vars = new_vars,
                                            params=new_params)

    # no changes, except initialization of new object
    def uniformize(self):
        """Generates a new leaf region where coefficients are set to 1 and all non-zero probs are set to 1.

        :return: uniformized GeneralizedSparseWeights
        """
        new_params = {"log_coefficients": jnp.zeros_like(self.params["log_coefficients"])}

        for var in self.vars:
            region_log_probs_0 = self.params[var][0, ...]
            region_log_probs_1 = self.params[var][1, ...]

            uniformized_region_log_probs_0 = jnp.log(jnp.logical_not(jnp.isneginf(region_log_probs_0)).astype(jnp.float32))
            uniformized_region_log_probs_1 = jnp.log(jnp.logical_not(jnp.isneginf(region_log_probs_1)).astype(jnp.float32))

            new_params[var] = jnp.stack([uniformized_region_log_probs_0, uniformized_region_log_probs_1], axis=0)

        return FactorizedBinaryDenseLeafRegion(region_shape=self.region_shape,
                                               group_sizes=self.group_sizes,
                                            vars=self.vars,
                                            params=new_params)

    # no changes, except initialization of new object
    @staticmethod
    def marginalize(region, marg_set):
        """Marginalizes a leaf region.

        Marginalization value gets absorbed into the coefficient (this will not be one if the region is not normalized,
        e.g. after a product).

        :param marg_set: set of variables to be marginalized out
        :return: the marginalized region.
        """
        # marginalizes by removing the marginalized variables, so that they don't get calculated in forward
        new_vars = region.vars.difference(marg_set)
        new_params = {var: region.params[var] for var in new_vars}

        new_marg_vars_log_coeff_list = [jnp.logaddexp(region.params[var][0, ...], region.params[var][1, ...]) for var in region.vars.intersection(marg_set)] # probs_0 + probs_1
        new_marg_vars_log_coeff_list.append(region.params["log_coefficients"])
        new_coeffs = jnp.sum(jnp.stack(new_marg_vars_log_coeff_list, axis=0),axis=0)
        new_params["log_coefficients"] = new_coeffs

        return FactorizedBinaryDenseLeafRegion(region_shape=region.region_shape,
                                               group_sizes=region.group_sizes,
                                               vars=new_vars, params=new_params)

    # no changes, except initialization of new object, and new check of shape
    @staticmethod
    def product(region_1, region_2):
        """Product of two leaf regions.

        If the variables have distinct scope, then for any variables which they do not share, the dimensions are
        expanded in order to ensure that all variables have the same param shape.

        :param region_1, region_2: regions to be producted
        :return: the product region
        """
        new_vars = region_1.vars.union(region_2.vars)
        new_params = dict()



        #no tensordot in log-domain
        new_params["log_coefficients"] = tensordot_logdomain(region_1.params["log_coefficients"], region_2.params["log_coefficients"])
        for var in new_vars:
            if var in region_1.vars:
                region_1_probs_0 = region_1.params[var][0, ...]
                region_1_probs_1 = region_1.params[var][1, ...]
            else:
                region_1_probs_0 = jnp.zeros(shape=interleave(region_1.region_shape, region_1.group_sizes))
                region_1_probs_1 = jnp.zeros(shape=interleave(region_1.region_shape, region_1.group_sizes))

            if var in region_2.vars:
                region_2_probs_0 = region_2.params[var][0, ...]
                region_2_probs_1 = region_2.params[var][1, ...]
            else:
                region_2_probs_0 = jnp.zeros(shape=interleave(region_1.region_shape, region_1.group_sizes))
                region_2_probs_1 = jnp.zeros(shape=interleave(region_2.region_shape, region_2.group_sizes))

            new_params[var] = jnp.stack(
                [tensordot_logdomain(region_1_probs_0, region_2_probs_0),
                 tensordot_logdomain(region_1_probs_1, region_2_probs_1)],
                axis=0
            )

            assert len(new_params[var].shape) == 1 + len(region_1.region_shape) + len(region_2.region_shape) + \
                len(region_1.group_sizes) + len(region_2.group_sizes) # +1 for the 0,1 probs dimension

        return FactorizedBinaryDenseLeafRegion(region_shape=region_1.region_shape + region_2.region_shape,
                                            group_sizes = region_1.group_sizes + region_2.group_sizes,
                                            vars=new_vars,
                                            params=new_params)
