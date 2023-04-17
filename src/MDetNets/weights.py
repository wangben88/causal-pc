from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from src.MDetNets.utils import tensordot_logdomain
import itertools
from jax import random
import numpy as np

@register_pytree_node_class
class GeneralizedSparseWeights:
    """Sparse implementation of weights. That is, we assume that there are only a limited number of product nodes for
    each region, for which we store the details here.

    For each region, we explictly record, for each product node in the region, which combinations of sum nodes from the
    region's children it is connected to. Note that the order of child regions in the combination corresponds to the
    order of the child regions in LazyMDNet.children[region]. We also record a weight for each product node.
    Each sum node in the region is then connected to one or more product nodes.

    In this initial implementation, sum nodes and product nodes are scalars (this will be changed into vectors in the
    future).

    The attributes are as follows:

    - child_combinations: the combination of children from the (2) child layers; shape (n_products, 2)
    - sum_to_product_and_weights: the product nodes that each sum node is connected to, and the corresponding weights;
      dict with keys "products" and "log_weights", each a list (length num_sums) of arrays
    - sum_log_coefficients: (log-)coefficients for each sum node (i.e. that node is multiplied by this). Note that this is
      separate from the weights as it may no longer be a product along dimensions/circuits.

    All of these are further wrapped in a list, to represent the generalized scenario where a weight can be the product
    of multiple individual weights, when circuits are multiplied together.
    """

    def __init__(self, child_combinations_list=None, sum_to_product_and_weights_list=None,
                 sum_log_coefficients=None):
        if child_combinations_list is None:
            self.child_combinations_list = []
        else:
            self.child_combinations_list = child_combinations_list

        if sum_to_product_and_weights_list is None:
            self.sum_to_product_and_weights_list = []
            assert sum_log_coefficients is None
            self.sum_log_coefficients = sum_log_coefficients
        else:
            self.sum_to_product_and_weights_list = sum_to_product_and_weights_list
            # default coefficients of 1
            if sum_log_coefficients is None:
                region_shape = tuple(len(sum_to_products_and_weights["products"]) for sum_to_products_and_weights in sum_to_product_and_weights_list)
                self.sum_log_coefficients = jnp.zeros(shape=region_shape)
            else:
                self.sum_log_coefficients = sum_log_coefficients


    @staticmethod
    def random_initialize(shape_params, key):
        """Provides a random initialization of the GeneralizedSparseWeight, given some shape_params detailing the
        relevant dimensions.

        shape_params is a dict containing the following keys
        - "region_size": region size of the parent region
        - "children_region_sizes": list of region sizes, for the children regions

        """
        num_children = len(shape_params["children_region_sizes"])

        num_product_nodes = np.product(np.array(shape_params["children_region_sizes"]))

        # STEP 1: Assign product nodes in a near-equal fashion to the sum nodes in the region
        sum_to_product = jnp.array_split(jnp.arange(num_product_nodes), shape_params["region_size"])

        children_region_elements = [jnp.arange(child_region_size) for child_region_size in
                                    shape_params["children_region_sizes"]]

        # STEP 2: Randomly generate the child combinations
        # jnp array with shape (num_combinations, num_children), where each all_child_combinations[i] corresponds to a
        # combination of children sum nodes.
        child_combinations = jnp.swapaxes(
            jnp.stack(jnp.meshgrid(*children_region_elements), axis=0).reshape(num_children, -1),
            0, 1
        )
        key, subkey = random.split(key)
        shuffled_child_combinations = random.permutation(key=subkey, x=child_combinations, axis=0,
                                                         independent=False)

        # STEP 3: Generate random weights
        random_log_weights = []
        for product_idxs in sum_to_product:
            key, subkey = random.split(key)
            random_log_weights.append(jnp.log(random.dirichlet(key=subkey, alpha=jnp.ones_like(product_idxs))))

        return GeneralizedSparseWeights(child_combinations_list=[shuffled_child_combinations],
            sum_to_product_and_weights_list=[{"products": sum_to_product,
                                              "log_weights": random_log_weights}])


    @staticmethod
    def identity_with_child_shape(child_region_shape):
        """Given a child region, geenrates "dummy weights" for a new region which has the child region and additionally
        a trivial region (with one sum node). Has identity weights, i.e. the new region has the same shape as the
        child region, and would return the same value under a forward pass.

        e.g. if we are given a child region with shape (2, 4, 3), then this would "create" a trivial region with shape (1,),
        and a new region with shape (2, 4, 3) which has the child and trivial regions as children.

        Note that by default, the weights will be such that the child region is the FIRST child of the new region.

        :param child_region_shape: the shape of the child region (and the new region)
        :return: weights for the new region
        """
        child_combinations_list = [jnp.stack([jnp.arange(region_dim_size, dtype=jnp.int32), jnp.zeros(region_dim_size, dtype=jnp.int32)], axis=1)
                                   for region_dim_size in child_region_shape]
        sum_to_product_and_weights_list = [
                {"products": [jnp.array([i], dtype=jnp.int32) for i in range(region_dim_size)],
                 "log_weights": [jnp.array([0.0]) for i in range(region_dim_size)]
                 }
            for region_dim_size in child_region_shape]
        sum_log_coefficients = jnp.zeros(shape=child_region_shape)

        weights = GeneralizedSparseWeights(child_combinations_list=child_combinations_list,
                                           sum_to_product_and_weights_list=sum_to_product_and_weights_list,
                                           sum_log_coefficients=sum_log_coefficients)
        return weights

    @staticmethod
    def product(weights1, weights2):
        """Generates the product of two GeneralizedSparseWeights.

        :param weights1, weights2: the weights to be multiplied (in order)
        :return: new GeneralizedSparseWeights object representing the product of the two weights.
        """
        new_sum_log_coefficients = tensordot_logdomain(weights1.sum_log_coefficients, weights2.sum_log_coefficients)

        product_weights = GeneralizedSparseWeights(child_combinations_list = weights1.child_combinations_list + weights2.child_combinations_list,
                                                   sum_to_product_and_weights_list = weights1.sum_to_product_and_weights_list + weights2.sum_to_product_and_weights_list,
                                                   sum_log_coefficients=new_sum_log_coefficients)
        return product_weights

    def multiply(self, value):
        """Generates a new GeneralizedSparseWeights which multiplies the coefficients by the given value.

        :param value: array of shape region_shape to be multiplied
        :return: multiplied GeneralizedSparseWeights
        """
        assert value.shape == self.sum_log_coefficients.shape

        new_sum_log_coefficients = self.sum_log_coefficients + jnp.log(value)

        return GeneralizedSparseWeights(child_combinations_list = self.child_combinations_list,
                                        sum_to_product_and_weights_list = self.sum_to_product_and_weights_list,
                                        sum_log_coefficients=new_sum_log_coefficients)

    def uniformize(self):
        """Generates a new GeneralizedSparseWeights where coefficients are set to 1 and all non-zero weights are
        set to 1.

        :return: uniformized GeneralizedSparseWeights
        """
        new_sum_log_coefficients = jnp.zeros_like(self.sum_log_coefficients)

        new_sum_to_product_and_weights_list = []

        for sum_to_product_and_weights in self.sum_to_product_and_weights_list:
            new_sum_to_products = sum_to_product_and_weights["products"]
            new_sum_to_log_weights = []
            for log_weights in sum_to_product_and_weights["log_weights"]:
                new_log_weights = jnp.log(jnp.logical_not(jnp.isneginf(log_weights)).astype(jnp.float32))
                new_sum_to_log_weights.append(new_log_weights)
            new_sum_to_product_and_weights = {"products": new_sum_to_products,
                                              "log_weights": new_sum_to_log_weights}
            new_sum_to_product_and_weights_list.append(new_sum_to_product_and_weights)

        return GeneralizedSparseWeights(child_combinations_list = self.child_combinations_list,
                                        sum_to_product_and_weights_list= new_sum_to_product_and_weights_list,
                                        sum_log_coefficients = new_sum_log_coefficients)


    def swap_children(self):
        """Utility function which produces a new weight which is equivalent to the current weight but with the children
        region order swapped.

        :return: new GeneralizedSparseWeights for the swapped case
        """
        swapped_child_combinations_list = []
        for child_combinations in self.child_combinations_list:
            assert child_combinations.shape[1] == 2 # only implemented for binary vtree
            swapped_child_combinations = jnp.stack([child_combinations[:, 1], child_combinations[:, 0]], axis=1)
            swapped_child_combinations_list.append(swapped_child_combinations)

        return GeneralizedSparseWeights(child_combinations_list=swapped_child_combinations_list,
                                        sum_to_product_and_weights_list=self.sum_to_product_and_weights_list,
                                        sum_log_coefficients = self.sum_log_coefficients)


    def tree_flatten(self):
        children = (self.child_combinations_list, self.sum_to_product_and_weights_list, self.sum_log_coefficients)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def forward_log(self, input_log_values):
        # Explicit computation of product nodes, since we are working with sparse.
        assert len(input_log_values) == 2   # only binary vtrees allowed currently

        # dimension (batch_size, k_(l, 1), ..., k_(l, n)) where n is the number of circuits that have been multiplied
        # similar (batch_size, k_(r, 1), ..., k_(r, n)).
        left_child_log_values, right_child_log_values = input_log_values[0], input_log_values[1]


        # need to add 1 for batch dimension
        def circuit_idx_to_dimension(circuit_idx):
            return circuit_idx + 1

        # Step 1: Index the children with the combinations
        # now dimensions (batch_size, p_1, ..., p_n) for both left and right, where p_i is the number of combinations
        # for circuit i.
        for circuit_idx, child_combinations in enumerate(self.child_combinations_list):
            assert child_combinations.shape[1] == 2
            child_combinations_left, child_combinations_right = child_combinations[:, 0], child_combinations[:, 1]
            left_child_log_values = jnp.take(left_child_log_values, child_combinations_left,
                                         axis=circuit_idx_to_dimension(circuit_idx))
            right_child_log_values = jnp.take(right_child_log_values, child_combinations_right,
                                          axis=circuit_idx_to_dimension(circuit_idx))

        # Step 2: Combine the children with a product
        combinations_log_values = left_child_log_values + right_child_log_values

        # Step 3: Multiply with weights and sum out into region sum nodes
        region_log_values = combinations_log_values
        num_circuits = len(self.sum_to_product_and_weights_list)
        # For each circuit, i.e. each array/dimension of combinations.
        for circuit_idx, sum_to_product_and_weights in enumerate(self.sum_to_product_and_weights_list):
            assert region_log_values.ndim == num_circuits + 1
            broadcast_shape = [-1 if axis == circuit_idx_to_dimension(circuit_idx) else 1 for axis in
                               range(region_log_values.ndim)]
            #region_values = region_values * weights.reshape(broadcast_shape)

            # Step 3: "Sum out" this dimension, such that the size of the dimension goes from num_combinations to num_outputs

            region_log_values = jnp.stack(
                [logsumexp(
                    jnp.take(
                        region_log_values,
                        products,
                        axis=circuit_idx_to_dimension(circuit_idx)
                    )
                    +
                    log_weights.reshape(broadcast_shape)
                    ,
                    axis=circuit_idx_to_dimension(circuit_idx)
                )
                    for products, log_weights in zip(sum_to_product_and_weights["products"], sum_to_product_and_weights["log_weights"])
                ],
                axis=circuit_idx_to_dimension(circuit_idx)
            )

        # Step 3b: Multiply with coefficients, which has dimension (k_1, ..., k_n)
        region_log_values = region_log_values + jnp.expand_dims(self.sum_log_coefficients, axis=0)

        # Final shape of region_values is (batch_size, k_1, ..., k_n)

        return region_log_values


@register_pytree_node_class
class GeneralizedDenseWeights():
    """Differs from the sparse implementation in two ways.

    1) Each node is now a vector group. In practice, this means that we expand the dimensions so that there are 2 dimensions
    for each circuit, one for the group within the region, and one for the node within the group. These are organized as
    k_1, g_1, ..., k_n, g_n where k is the group index and g is the unit within the group.
    2) We change to a dense implementation where each sum group is connected to all product groups. Alternatives are to
    connect only to a single product group, or to explicitly emunerate the connections as in the sparse implementation.
    Currently, hwoever, we do not generate all possible product groups.
    """

    def __init__(self, child_combinations_list=None, sum_to_log_weights_list=None,
                 sum_log_coefficients=None):
        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
            - k: group within region
            - g: node within group
            - p: ranges over product groups (N_L, N_R)
            - g_l: node within N_L
            - g_r: node within N_R
        """

        if child_combinations_list is None:
            self.child_combinations_list = []
        else:
            self.child_combinations_list = child_combinations_list

        if sum_to_log_weights_list is None:
            self.sum_to_log_weights_list = []
            assert sum_log_coefficients is None
            self.sum_log_coefficients = sum_log_coefficients
        else:
            self.sum_to_log_weights_list = sum_to_log_weights_list
            # default coefficients of 1
            if sum_log_coefficients is None:
                # (k_1, g_1, k_2, g_2, ..., k_n, g_n)
                region_shape = tuple(sum_to_log_weights.shape[i]
                                     for sum_to_log_weights in sum_to_log_weights_list
                                     for i in range(2))
                self.sum_log_coefficients = jnp.zeros(region_shape)
            else:
                self.sum_log_coefficients = sum_log_coefficients

    def tree_flatten(self):
        children = (self.sum_to_log_weights_list, self.sum_log_coefficients)
        aux_data = (self.child_combinations_list,)  # disable tracing of child combinations list for now
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(child_combinations_list = aux_data[0], sum_to_log_weights_list = children[0],
                   sum_log_coefficients = children[1])

    @staticmethod
    def random_initialize(shape_params, key):
        """Provides a random initialization of the GeneralizedSparseWeight, given some shape_params detailing the
        relevant dimensions.

        shape_params is a dict containing the following keys
        - "region_size": region size of the parent region
        - "group_size": group size for the parent region
        - "left_child_region_size": region size of the left child region
        - "right_child_region_size": region size of the left child region
        - "num_products": number of products to randomly generate
        - "left_child_group_size": group size for the left child region
        - "right_child_group_size": group size for the right child region
        """
        region_size = shape_params["region_size"]
        group_size = shape_params["group_size"]
        left_child_region_size = shape_params["left_child_region_size"]
        right_child_region_size = shape_params["right_child_region_size"]
        num_products = shape_params["num_products"]
        left_child_group_size = shape_params["left_child_group_size"]
        right_child_group_size = shape_params["right_child_group_size"]

        # Step 1: Select products randomly
        key, subkey = random.split(key)
        left_combinations = random.randint(key=subkey, shape=(num_products, ), minval=0, maxval=left_child_region_size)
        right_combinations = random.randint(key=subkey, shape=(num_products,), minval=0, maxval=right_child_region_size)

        child_combinations_list = [jnp.stack([left_combinations, right_combinations], axis=1)]

        # Step 2: Generate random weights
        key, subkey = random.split(key)
        sum_to_log_weights = jnp.log(random.dirichlet(key=subkey,
                                              alpha=jnp.ones(
                                                  shape=(num_products * left_child_group_size * right_child_group_size,)
                                              ),
                                              shape=(region_size, group_size)
                                              ))
        sum_to_log_weights = jnp.reshape(sum_to_log_weights, newshape=(region_size,
                                                                       group_size,
                                                                       num_products,
                                                                       left_child_group_size,
                                                                       right_child_group_size))
        sum_to_log_weights_list = [sum_to_log_weights]

        return GeneralizedDenseWeights(child_combinations_list=child_combinations_list,
                                       sum_to_log_weights_list=sum_to_log_weights_list)

    @staticmethod
    def random_det_initialize(shape_params, key, str_det=None):
        """Provides a random DETERMINISTIC initialization of the GeneralizedSparseWeight. This means that each product
        is assigned to a single sum node.

        shape_params is a dict containing the following keys
        - "region_size": region size of the parent region
        - "group_size": group size for the parent region
        - "left_child_region_size": region size of the left child region
        - "right_child_region_size": region size of the left child region
        - "num_products": number of products to randomly generate
        - "left_child_group_size": group size for the left child region
        - "right_child_group_size": group size for the right child region
        """

        region_size = shape_params["region_size"]
        group_size = shape_params["group_size"]
        left_child_region_size = shape_params["left_child_region_size"]
        right_child_region_size = shape_params["right_child_region_size"]
        num_products = shape_params["num_products"]
        left_child_group_size = shape_params["left_child_group_size"]
        right_child_group_size = shape_params["right_child_group_size"]

        if shape_params["region_size"] > 1 or shape_params["num_products"] > 1 or shape_params["left_child_region_size"] > 1\
            or shape_params["right_child_region_size"] > 1:
            raise NotImplementedError("Deterministic weight initialization only implemented for region sizes of 1")





        # Step 1: Select products randomly
        key, subkey = random.split(key)

        child_combinations_list = [jnp.stack([jnp.array([0], dtype=jnp.int32), jnp.array([0], dtype=jnp.int32)], axis=1)] # the only product

        # Step 2: Generate random weights
        # For deterministic, we need to split the weights.

        key, subkey = random.split(key)
        if str_det is None:
            assignments = random.randint(key=subkey, shape=(left_child_group_size, right_child_group_size), minval=0,
                                         maxval=group_size)
            rand_log_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                num_weights = jnp.sum(mask)
                if num_weights > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[mask].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(num_weights,)))
                        # weights_arr[mask]/jnp.sum(weights_arr[mask])
                    )  # + 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             )  # + 1e-09
                rand_log_weights = jnp.log(rand_weights)
                rand_log_weights = jnp.expand_dims(rand_log_weights, axis=0)  # products dimension
                rand_log_weights_list.append(rand_log_weights)
        elif str_det == 'left':
            assignments = random.randint(key=subkey, shape=(left_child_group_size,), minval=0,
                                         maxval=group_size)
            rand_log_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                if jnp.sum(mask) > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[mask, :].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(jnp.sum(mask), right_child_group_size)))
                        # weights_arr[mask]/jnp.sum(weights_arr[mask])
                    )  # + 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             )  # + 1e-09
                rand_log_weights = jnp.log(rand_weights)
                rand_log_weights = jnp.expand_dims(rand_log_weights, axis=0)  # products dimension
                rand_log_weights_list.append(rand_log_weights)
        elif str_det == "right":
            assignments = random.randint(key=subkey, shape=(right_child_group_size,), minval=0,
                                         maxval=group_size)
            rand_log_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                if jnp.sum(mask) > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[:, mask].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(left_child_group_size, jnp.sum(mask))))
                        # weights_arr[mask]/jnp.sum(weights_arr[mask])
                    )  # + 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             )  # + 1e-09
                rand_log_weights = jnp.log(rand_weights)
                rand_log_weights = jnp.expand_dims(rand_log_weights, axis=0)  # products dimension
                rand_log_weights_list.append(rand_log_weights)

        group_rand_log_weights = jnp.stack(rand_log_weights_list, axis=0)
        region_rand_log_weights = jnp.expand_dims(group_rand_log_weights, axis=0)

        sum_to_log_weights = region_rand_log_weights
        sum_to_log_weights_list = [sum_to_log_weights]

        return GeneralizedDenseWeights(child_combinations_list=child_combinations_list,
                                       sum_to_log_weights_list=sum_to_log_weights_list)


    def forward_log(self, input_log_values, return_counts=False):
        # Step 0: Extract the left and right inputs.
        assert len(input_log_values) == 2

        # dimension (batch_size, K_(l, 1), G_(l, 1), ..., K_(l, n), G_(l, n)), sim for r.
        left_child_log_values, right_child_log_values = input_log_values[0], input_log_values[1]
        batch_size = left_child_log_values.shape[0]

        num_circuits = len(self.child_combinations_list)

        # need to add 1 for batch dimension, multiply by 2 to account for the group dimension
        def circuit_idx_to_region_dimension(circuit_idx):
            return 2*circuit_idx + 1
        def circuit_idx_to_group_dimension(circuit_idx):
            return 2*circuit_idx + 2

        # Step 1: Index the children with the combinations
        # Afterwards, dimension (batch_size, P_1, G_(l, 1), ..., P_n, G_(l, n))
        # For circuit i.
        for circuit_idx, child_combinations in enumerate(self.child_combinations_list):
            assert child_combinations.shape[1] == 2
            child_combinations_left, child_combinations_right = child_combinations[:, 0], child_combinations[:, 1]
            left_child_log_values = jnp.take(left_child_log_values, child_combinations_left,
                                             axis=circuit_idx_to_region_dimension(circuit_idx))
            right_child_log_values = jnp.take(right_child_log_values, child_combinations_right,
                                              axis=circuit_idx_to_region_dimension(circuit_idx))

        # Step 2: Combine the children with a product (sum in log-domain) . Note that we need to sum along the p dimensions, and expand
        # along the g dimensions.
        # Shape (batch_size, P_1, G_(l, 1), G_(r, 1), ... P_n, G_(l, n), G_(r, n))
        combinations_log_values = jnp.reshape(left_child_log_values,
                                              newshape=(batch_size,) + tuple(itertools.chain.from_iterable(
                                                  (left_child_log_values.shape[circuit_idx_to_region_dimension(circuit_idx)],
                                                   left_child_log_values.shape[circuit_idx_to_group_dimension(circuit_idx)],
                                                   1)
                                                  for circuit_idx in range(num_circuits)
                                              ))
                                              ) \
                                  + \
                                  jnp.reshape(right_child_log_values,
                                              newshape=(batch_size,) + tuple(itertools.chain.from_iterable(
                                                  (left_child_log_values.shape[circuit_idx_to_region_dimension(circuit_idx)],
                                                   1,
                                                   right_child_log_values.shape[circuit_idx_to_group_dimension(circuit_idx)])
                                                  for circuit_idx in range(num_circuits)
                                              ))
                                              )

        # Step 2b: Get counts (note that we don't have to keep track of which data entries they come from at the moment,
        # as we assume a single group for all regions, so each group covers all data entries)

        product_counts = jnp.sum(jnp.logical_not(jnp.isneginf(combinations_log_values)),
                                 axis=0)  # sum over batch entries

        # Step 3: Multiply with weights and sum out into region sum nodes
        region_log_values = combinations_log_values
        for circuit_idx, sum_to_log_weights in enumerate(self.sum_to_log_weights_list):
            assert region_log_values.ndim == 1 + 2 * (circuit_idx) + 3 * (num_circuits - circuit_idx)
            broadcast_shape_log_weights = (1,) + (1, 1) * circuit_idx + sum_to_log_weights.shape + (1,) * 3 * (num_circuits - circuit_idx - 1)
            broadcast_shape_log_values = region_log_values.shape[ : 1+2*(circuit_idx)] + (1, 1) + region_log_values.shape[1+2*(circuit_idx):]

            region_log_values = logsumexp(
                jnp.reshape(region_log_values, newshape=broadcast_shape_log_values) + \
                jnp.reshape(sum_to_log_weights, newshape=broadcast_shape_log_weights),
                axis=(1+2*(circuit_idx)+2, 1+2*(circuit_idx)+3, 1+2*(circuit_idx)+4)  # p_i, g_(l, i), g_(r, i)
            )


        # Final result has shape (batch_size, K_1, G_1, ..., K_n, G_n)

        # Step 3b: Multiply with coefficients, which has dimension (K_1, G_1, ..., K_n, G_n)
        region_log_values = region_log_values + jnp.expand_dims(self.sum_log_coefficients, axis=0)

        if return_counts:
            return region_log_values, product_counts
        else:
            return region_log_values


    def update_em(self, grads_batched):
        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
            - k: group within region
            - g: node within group
            - p: ranges over product groups (N_L, N_R)
            - g_l: node within N_L
            - g_r: node within N_R
        """
        if len(self.sum_to_log_weights_list) > 1:
            raise NotImplementedError("EM update for weights only implemented for single circuits")
        if not jnp.all(jnp.isclose(self.sum_log_coefficients, 0.0)):
            raise NotImplementedError("EM update for weights only implemented for log_coefficients equal to 0")

        grads_arr = jnp.sum(jnp.stack([grads.sum_to_log_weights_list[0] for grads in grads_batched]),
                            axis=0)  # d/dlog_(w_ij) = d/dw_ij dw_ij/dlog_(w_ij) = w_ij d/dw_ij = n_ij

        print(grads_arr)


        grads_arr_norm = jnp.sum(grads_arr, axis=(2, 3, 4), keepdims=True)

        self.sum_to_log_weights_list[0] = jnp.log(grads_arr+1e-8) - (jnp.log(grads_arr_norm+1e-8))  # correction to ensure if tot count is 0 (unattached sum nodes), we still get a correct result



    def update_em_det(self, product_counts, alpha=0.1):
        """product_counts has shape (P, G_L, G_R) where P = 1

        :param alpha: Laplace smoothing parameter.
        """
        if len(self.sum_to_log_weights_list) > 1:
            raise NotImplementedError("EM update for log weights only implemented for single circuits")
        if not jnp.all(jnp.isclose(self.sum_log_coefficients, 0.0)):
            raise NotImplementedError("EM update for log weights only implemented for log coefficients equal to 0")

        nij = jnp.logical_not(jnp.isneginf(self.sum_to_log_weights_list[0])) * jnp.expand_dims(product_counts, axis=(0, 1))

        nij = nij + jnp.logical_not(jnp.isneginf(self.sum_to_log_weights_list[0])) * alpha

        nij_norm = jnp.sum(nij, axis=(2, 3, 4), keepdims=True)

        log_weights_with_nans = jnp.log(nij) - jnp.log(nij_norm)

        self.sum_to_log_weights_list[0] = log_weights_with_nans.at[jnp.isnan(log_weights_with_nans)].set(-jnp.inf)



    ############### OPERATIONS #########################

    @staticmethod
    def product(weights1, weights2):
        """Generates the product of two GeneralizedSparseWeights.

        :param weights1, weights2: the weights to be multiplied (in order)
        :return: new GeneralizedSparseWeights object representing the product of the two weights.
        """

        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
                    - k: group within region
                    - g: node within group
                    - p: ranges over product groups (N_L, N_R)
                    - g_l: node within N_L
                    - g_r: node within N_R
                """


        # Unfortunately there is no analogue to tensordot in the log-domain (i.e. sum rather than product), so we must
        # do this manually.
        new_sum_log_coefficients = tensordot_logdomain(weights1.sum_log_coefficients, weights2.sum_log_coefficients)

        product_weights = GeneralizedDenseWeights(
            child_combinations_list=weights1.child_combinations_list + weights2.child_combinations_list,
            sum_to_log_weights_list=weights1.sum_to_log_weights_list + weights2.sum_to_log_weights_list,
            sum_log_coefficients=new_sum_log_coefficients)
        return product_weights

    def multiply(self, value):
        """Generates a new GeneralizedSparseWeights which multiplies the coefficients by the given value.

        :param value: array of shape region_shape to be multiplied
        :return: multiplied GeneralizedSparseWeights
        """
        assert value.shape == self.sum_log_coefficients.shape

        new_sum_log_coefficients = self.sum_log_coefficients + jnp.log(value)

        return GeneralizedDenseWeights(child_combinations_list=self.child_combinations_list,
                                       sum_to_log_weights_list=self.sum_to_log_weights_list,
                                        sum_log_coefficients=new_sum_log_coefficients)

    def uniformize(self):
        """Generates a new GeneralizedSparseWeights where coefficients are set to 1 and all non-zero weights are
        set to 1.

        :return: uniformized GeneralizedSparseWeights
        """
        new_sum_log_coefficients = jnp.zeros_like(self.sum_log_coefficients)

        new_sum_to_log_weights_list = []

        for sum_to_log_weights in self.sum_to_log_weights_list:
            new_sum_to_log_weights = jnp.log(jnp.logical_not(jnp.isneginf(sum_to_log_weights)).astype(jnp.float32))
            new_sum_to_log_weights_list.append(new_sum_to_log_weights)

        return GeneralizedDenseWeights(child_combinations_list=self.child_combinations_list,
                                        sum_to_log_weights_list=new_sum_to_log_weights_list,
                                        sum_log_coefficients=new_sum_log_coefficients)

    def uniformize_left(self):
        """ Normalize each "row", i.e. for each given left child
        """
        new_sum_log_coefficients = jnp.zeros_like(self.sum_log_coefficients)

        new_sum_to_log_weights_list = []

        for sum_to_log_weights in self.sum_to_log_weights_list:
            new_sum_to_log_weights = sum_to_log_weights - logsumexp(sum_to_log_weights, axis=4, keepdims=True)
            new_sum_to_log_weights = new_sum_to_log_weights.at[jnp.isnan(new_sum_to_log_weights)].set(-jnp.inf)
            new_sum_to_log_weights_list.append(new_sum_to_log_weights)

        return GeneralizedDenseWeights(child_combinations_list=self.child_combinations_list,
                                       sum_to_log_weights_list=new_sum_to_log_weights_list,
                                       sum_log_coefficients=new_sum_log_coefficients)

    def uniformize_left_with_right_child_values(self, right_child_log_values):
        """ Normalize each "row", i.e. for each given left child
        """
        assert len(self.sum_to_log_weights_list) == 1

        new_sum_log_coefficients = jnp.zeros_like(self.sum_log_coefficients)

        new_sum_to_log_weights_list = []

        for sum_to_log_weights in self.sum_to_log_weights_list:
            assert len(right_child_log_values.shape) == 2
            assert right_child_log_values.shape[0] == 1
            total = logsumexp(sum_to_log_weights + jnp.expand_dims(right_child_log_values, axis=(0, 1, 2)), axis=4, keepdims=True)
            new_sum_to_log_weights = sum_to_log_weights - total
            new_sum_to_log_weights = new_sum_to_log_weights.at[jnp.isnan(new_sum_to_log_weights)].set(-jnp.inf)
            new_sum_to_log_weights_list.append(new_sum_to_log_weights)

        return GeneralizedDenseWeights(child_combinations_list=self.child_combinations_list,
                                       sum_to_log_weights_list=new_sum_to_log_weights_list,
                                       sum_log_coefficients=new_sum_log_coefficients)

    def swap_children(self):
        """Utility function which produces a new weight which is equivalent to the current weight but with the children
        region order swapped.

        :return: new GeneralizedSparseWeights for the swapped case
        """
        swapped_child_combinations_list = []
        for child_combinations in self.child_combinations_list:
            assert child_combinations.shape[1] == 2  # only implemented for binary vtree
            swapped_child_combinations = jnp.stack([child_combinations[:, 1], child_combinations[:, 0]], axis=1)
            swapped_child_combinations_list.append(swapped_child_combinations)

        return GeneralizedDenseWeights(child_combinations_list=swapped_child_combinations_list,
                                        sum_to_log_weights_list=self.sum_to_log_weights_list,
                                        sum_log_coefficients=self.sum_log_coefficients)







@register_pytree_node_class
class GeneralizedDenseLinearWeights():
    # Basically the same as denseweights, but with weights stored in the linear domain
    def __init__(self, child_combinations_list=None, sum_to_weights_list=None,
                 sum_coefficients=None):
        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
            - k: group within region
            - g: node within group
            - p: ranges over product groups (N_L, N_R)
            - g_l: node within N_L
            - g_r: node within N_R
        """

        if child_combinations_list is None:
            self.child_combinations_list = []
        else:
            self.child_combinations_list = child_combinations_list

        if sum_to_weights_list is None:
            self.sum_to_weights_list = []
            assert sum_coefficients is None
            self.sum_coefficients = sum_coefficients
        else:
            self.sum_to_weights_list = sum_to_weights_list
            # default coefficients of 1
            if sum_coefficients is None:
                # (k_1, g_1, k_2, g_2, ..., k_n, g_n)
                region_shape = tuple(sum_to_weights.shape[i]
                                     for sum_to_weights in sum_to_weights_list
                                     for i in range(2))
                self.sum_coefficients = jnp.ones(region_shape)
            else:
                self.sum_coefficients = sum_coefficients

    def tree_flatten(self):
        children = (self.sum_to_weights_list, self.sum_coefficients)
        aux_data = (self.child_combinations_list,)  # disable tracing of child combinations list for now
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(child_combinations_list=aux_data[0], sum_to_weights_list=children[0],
                   sum_coefficients=children[1])

    @staticmethod
    def random_initialize(shape_params, key):
        """Provides a random initialization of the GeneralizedSparseWeight, given some shape_params detailing the
        relevant dimensions.

        shape_params is a dict containing the following keys
        - "region_size": region size of the parent region
        - "group_size": group size for the parent region
        - "left_child_region_size": region size of the left child region
        - "right_child_region_size": region size of the left child region
        - "num_products": number of products to randomly generate
        - "left_child_group_size": group size for the left child region
        - "right_child_group_size": group size for the right child region
        """
        region_size = shape_params["region_size"]
        group_size = shape_params["group_size"]
        left_child_region_size = shape_params["left_child_region_size"]
        right_child_region_size = shape_params["right_child_region_size"]
        num_products = shape_params["num_products"]
        left_child_group_size = shape_params["left_child_group_size"]
        right_child_group_size = shape_params["right_child_group_size"]

        # Step 1: Select products randomly
        key, subkey = random.split(key)
        left_combinations = random.randint(key=subkey, shape=(num_products,), minval=0, maxval=left_child_region_size)
        right_combinations = random.randint(key=subkey, shape=(num_products,), minval=0, maxval=right_child_region_size)

        child_combinations_list = [jnp.stack([left_combinations, right_combinations], axis=1)]

        # Step 2: Generate random weights
        key, subkey = random.split(key)
        sum_to_weights = random.dirichlet(key=subkey,
                                                      alpha=jnp.ones(
                                                          shape=(
                                                          num_products * left_child_group_size * right_child_group_size,)
                                                      ),
                                                      shape=(region_size, group_size)
                                                      )
        sum_to_weights = jnp.reshape(sum_to_weights, newshape=(region_size,
                                                                       group_size,
                                                                       num_products,
                                                                       left_child_group_size,
                                                                       right_child_group_size))
        sum_to_weights_list = [sum_to_weights]

        return GeneralizedDenseLinearWeights(child_combinations_list=child_combinations_list,
                                       sum_to_weights_list=sum_to_weights_list)

    @staticmethod
    def random_det_initialize(shape_params, key, str_det=None):
        """Provides a random DETERMINISTIC initialization of the GeneralizedSparseWeight. This means that each product
        is assigned to a single sum node.

        shape_params is a dict containing the following keys
        - "region_size": region size of the parent region
        - "group_size": group size for the parent region
        - "left_child_region_size": region size of the left child region
        - "right_child_region_size": region size of the left child region
        - "num_products": number of products to randomly generate
        - "left_child_group_size": group size for the left child region
        - "right_child_group_size": group size for the right child region
        """

        region_size = shape_params["region_size"]
        group_size = shape_params["group_size"]
        left_child_region_size = shape_params["left_child_region_size"]
        right_child_region_size = shape_params["right_child_region_size"]
        num_products = shape_params["num_products"]
        left_child_group_size = shape_params["left_child_group_size"]
        right_child_group_size = shape_params["right_child_group_size"]

        if shape_params["region_size"] > 1 or shape_params["num_products"] > 1 or shape_params[
            "left_child_region_size"] > 1 \
                or shape_params["right_child_region_size"] > 1:
            raise NotImplementedError("Deterministic weight initialization only implemented for region sizes of 1")

        # Step 1: Select products randomly
        key, subkey = random.split(key)

        child_combinations_list = [
            jnp.stack([jnp.array([0], dtype=jnp.int32), jnp.array([0], dtype=jnp.int32)], axis=1)]  # the only product

        # Step 2: Generate random weights
        # For deterministic, we need to split the weights.
        key, subkey = random.split(key)
        if str_det is None:
            assignments = random.randint(key=subkey, shape=(left_child_group_size, right_child_group_size), minval=0,
                                         maxval=group_size)

            rand_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                num_weights = jnp.sum(mask)
                if num_weights > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[mask].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(num_weights,)))#weights_arr[mask]/jnp.sum(weights_arr[mask])
                    ) #+ 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ) #+ 1e-09
                rand_weights = jnp.expand_dims(rand_weights, axis=0)  # products dimension
                rand_weights_list.append(rand_weights)
        elif str_det == 'left':
            assignments = random.randint(key=subkey, shape=(left_child_group_size,), minval=0,
                                         maxval=group_size)
            rand_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                if jnp.sum(mask) > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[mask, :].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(jnp.sum(mask), right_child_group_size)))
                        # weights_arr[mask]/jnp.sum(weights_arr[mask])
                    )  # + 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             )  # + 1e-09
                rand_weights = jnp.expand_dims(rand_weights, axis=0)  # products dimension
                rand_weights_list.append(rand_weights)
        elif str_det == "right":
            assignments = random.randint(key=subkey, shape=(right_child_group_size,), minval=0,
                                         maxval=group_size)
            rand_weights_list = []
            for group_index in range(group_size):
                key, subkey = random.split(key)
                mask = (assignments == group_index)
                if jnp.sum(mask) > 0:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             ).at[:, mask].set(
                        random.dirichlet(key=subkey, alpha=jnp.ones(shape=(left_child_group_size, jnp.sum(mask))))
                        # weights_arr[mask]/jnp.sum(weights_arr[mask])
                    )  # + 1e-09
                else:
                    rand_weights = jnp.zeros(shape=(left_child_group_size, right_child_group_size,)
                                             )  # + 1e-09
                rand_weights = jnp.expand_dims(rand_weights, axis=0)  # products dimension
                rand_weights_list.append(rand_weights)

        group_rand_weights = jnp.stack(rand_weights_list, axis=0)
        region_rand_weights = jnp.expand_dims(group_rand_weights, axis=0)





        sum_to_weights = region_rand_weights
        sum_to_weights_list = [sum_to_weights]

        return GeneralizedDenseLinearWeights(child_combinations_list=child_combinations_list,
                                       sum_to_weights_list=sum_to_weights_list)

    def forward_log(self, input_log_values, return_counts=False):
        # Step 0: Extract the left and right inputs.
        assert len(input_log_values) == 2

        # dimension (batch_size, K_(l, 1), G_(l, 1), ..., K_(l, n), G_(l, n)), sim for r.
        left_child_log_values, right_child_log_values = input_log_values[0], input_log_values[1]
        batch_size = left_child_log_values.shape[0]

        num_circuits = len(self.child_combinations_list)

        # need to add 1 for batch dimension, multiply by 2 to account for the group dimension
        def circuit_idx_to_region_dimension(circuit_idx):
            return 2 * circuit_idx + 1

        def circuit_idx_to_group_dimension(circuit_idx):
            return 2 * circuit_idx + 2

        # Step 1: Index the children with the combinations
        # Afterwards, dimension (batch_size, P_1, G_(l, 1), ..., P_n, G_(l, n))
        # For circuit i.
        for circuit_idx, child_combinations in enumerate(self.child_combinations_list):
            assert child_combinations.shape[1] == 2
            child_combinations_left, child_combinations_right = child_combinations[:, 0], child_combinations[:, 1]
            left_child_log_values = jnp.take(left_child_log_values, child_combinations_left,
                                             axis=circuit_idx_to_region_dimension(circuit_idx))
            right_child_log_values = jnp.take(right_child_log_values, child_combinations_right,
                                              axis=circuit_idx_to_region_dimension(circuit_idx))

        # Step 2: Combine the children with a product (sum in log-domain) . Note that we need to sum along the p dimensions, and expand
        # along the g dimensions.
        # Shape (batch_size, P_1, G_(l, 1), G_(r, 1), ... P_n, G_(l, n), G_(r, n))
        combinations_log_values = jnp.reshape(left_child_log_values,
                                              newshape=(batch_size,) + tuple(itertools.chain.from_iterable(
                                                  (left_child_log_values.shape[
                                                       circuit_idx_to_region_dimension(circuit_idx)],
                                                   left_child_log_values.shape[
                                                       circuit_idx_to_group_dimension(circuit_idx)],
                                                   1)
                                                  for circuit_idx in range(num_circuits)
                                              ))
                                              ) \
                                  + \
                                  jnp.reshape(right_child_log_values,
                                              newshape=(batch_size,) + tuple(itertools.chain.from_iterable(
                                                  (left_child_log_values.shape[
                                                       circuit_idx_to_region_dimension(circuit_idx)],
                                                   1,
                                                   right_child_log_values.shape[
                                                       circuit_idx_to_group_dimension(circuit_idx)])
                                                  for circuit_idx in range(num_circuits)
                                              ))
                                              )

        # Step 2b: Get counts (note that we don't have to keep track of which data entries they come from at the moment,
        # as we assume a single group for all regions, so each group covers all data entries)

        product_counts = jnp.sum(jnp.logical_not(jnp.isneginf(combinations_log_values)), axis=0)  # sum over batch entries

        # Step 3: Multiply with weights and sum out into region sum nodes
        region_log_values = combinations_log_values
        for circuit_idx, sum_to_weights in enumerate(self.sum_to_weights_list):
            # When circuit_idx = i - 1, we have:
            # sum_to_log_weights shape = (K_i, G_i, P_i, G_(l, i), G_(r, i))
            # region_log_values shape = (batch_size, K_1, G_1, ..., K_(i-1), G_(i-1), P_i, G_(l, i), G_(r, i), P_(i + 1), G_(l, i+1), G_(r, i+1), ...)
            # and want to broadcast to shape
            # (batch_size, K_1, G_1, ..., K_(i-1), G_(i-1), K_i, G_i, P_i, G_(l, i), G_(r, i), P_(i + 1), G_(l, i+1), G_(r, i+1), ...)
            assert region_log_values.ndim == 1 + 2 * (circuit_idx) + 3 * (num_circuits - circuit_idx)
            broadcast_shape_weights = (1,) + (1, 1) * circuit_idx + sum_to_weights.shape + (1,) * 3 * (
                        num_circuits - circuit_idx - 1)
            broadcast_shape_values = region_log_values.shape[: 1 + 2 * (circuit_idx)] + (
            1, 1) + region_log_values.shape[1 + 2 * (circuit_idx):]

            reshaped_weights = jnp.reshape(sum_to_weights, newshape=broadcast_shape_weights)
            reshaped_region_log_values = jnp.reshape(region_log_values, newshape=broadcast_shape_values)

            # max along summed out axes
            max_region_log_values_keepdims = jnp.max(reshaped_region_log_values,
                                            axis=(1 + 2 * (circuit_idx) + 2, 1 + 2 * (circuit_idx) + 3, 1 + 2 * (circuit_idx) + 4),
                                            keepdims=True)
            max_region_log_values = jnp.max(reshaped_region_log_values,
                                                     axis=(1 + 2 * (circuit_idx) + 2, 1 + 2 * (circuit_idx) + 3,
                                                           1 + 2 * (circuit_idx) + 4),
                                                     )


            region_log_values = max_region_log_values + \
                jnp.log(
                    jnp.sum(
                        reshaped_weights * jnp.exp(reshaped_region_log_values - max_region_log_values_keepdims),
                        axis=(1 + 2 * (circuit_idx) + 2, 1 + 2 * (circuit_idx) + 3, 1 + 2 * (circuit_idx) + 4)
                    )
                )

        # Final result has shape (batch_size, K_1, G_1, ..., K_n, G_n)

        # Step 3b: Multiply with coefficients, which has dimension (K_1, G_1, ..., K_n, G_n)
        region_log_values = region_log_values * jnp.expand_dims(self.sum_coefficients, axis=0)

        if return_counts:
            return region_log_values, product_counts
        else:
            return region_log_values


    def update_em(self, grads_batched):
        """Each sum_to_log_weights is a jnp array with the dimension (k, g, p, g_l, g_r):
            - k: group within region
            - g: node within group
            - p: ranges over product groups (N_L, N_R)
            - g_l: node within N_L
            - g_r: node within N_R
        """
        if len(self.sum_to_weights_list) > 1:
            raise NotImplementedError("EM update for weights only implemented for single circuits")
        if not jnp.all(jnp.isclose(self.sum_coefficients, 1.0)):
            raise NotImplementedError("EM update for weights only implemented for coefficients equal to 1")

        grads_arr = jnp.sum(jnp.stack([grads.sum_to_weights_list[0] for grads in grads_batched]),
                            axis=0)  # d/dlog_(w_ij) = d/dw_ij dw_ij/dlog_(w_ij) = w_ij d/dw_ij = n_ij


        nij = grads_arr * self.sum_to_weights_list[0] + 1e-9


        nij_norm = jnp.sum(nij, axis=(2, 3, 4), keepdims=True)

        self.sum_to_weights_list[0] = nij/(nij_norm + 1e-9)


    def update_em_det(self, product_counts, alpha=0.1):
        """product_counts has shape (P, G_L, G_R) where P = 1

        :param alpha: Laplace smoothing parameter.
        """
        if len(self.sum_to_weights_list) > 1:
            raise NotImplementedError("EM update for weights only implemented for single circuits")
        if not jnp.all(jnp.isclose(self.sum_coefficients, 1.0)):
            raise NotImplementedError("EM update for weights only implemented for coefficients equal to 1")

        nij = (self.sum_to_weights_list[0] > 0) * jnp.expand_dims(product_counts, axis=(0, 1))

        nij = nij + alpha

        nij_norm = jnp.sum(nij, axis=(2, 3, 4), keepdims=True)

        self.sum_to_weights_list[0] = nij/nij_norm


