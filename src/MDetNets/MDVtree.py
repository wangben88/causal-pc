import random
from src.MDetNets.utils import random_scope_partition
from infinite_sets import everything
from functools import reduce

import uuid

class VtreeLabel:
    """Class implementing labels for vtree regions.

    Supports the required operations for vtree labels. The label is implemented as a set, with a special a special
    class attribute for the universal set.
    """

    universal_set = everything()

    def __init__(self, label_set):
        self.label_set = label_set # either a set (can be empty), or the value universal_set (indicating infty/universal set)

    def __eq__(self, other_label):
        return self.label_set == other_label.label_set

    @staticmethod
    def union(*labels):
        """Given a list of Vtree Labels, returns their union (e.g. used in product algorithm)

        :param labels: list of vtree labels
        :return: a vtree label representing the union
        """
        #print([label.label_set for label in labels])
        return VtreeLabel(reduce(lambda x, y: x | y, [label.label_set for label in labels]))

    @staticmethod
    def intersection(*labels):
        """Given a list of Vtree Labels, returns their intersection. NB is this really needed?

        :param labels: list of vtree labels
        :return: a vtree label representing the intersection
        """
        return VtreeLabel(reduce(lambda x, y: x & y, [label.label_set for label in labels]))

    def issubset(self, other_label):
        """Determines if the label is a subset (stronger condition) of another label."""
        return self.label_set <= other_label.label_set

    def issuperset(self, other_label):
        """Determines if the label is a superset (weaker condition) of another label."""
        return self.label_set >= other_label.label_set

    def is_contained_in_scope(self, scope):
        """Determines if the label is contained within a given scope (and thus valid)."""
        if self.label_set == VtreeLabel.universal_set:
            return True
        return self.label_set <= scope

    def overlaps_with_mdset(self, md_set):
        """Determines if the label overlaps (has non-empty intersection) with a given set."""
        return len(self.label_set & md_set) > 0

    def is_marginal_deterministic(self, md_set):
        """Determines if sum units in the PC respecting this labelling are MD wrt a given set

        :param md_set: the set Q which we are testing marginal determinism with respect to
        :return: Boolean indicating whether MD holds
        """
        return self.label_set <= md_set

    def __repr__(self):
        return self.label_set.__repr__()


class Vtree:
    """Defines a Vtree, which is a rooted tree where each node (henceforth referred to as a "region") is associated with
       a scope, and the children of a region have scopes partitioning the scope of the region. Each region is identified
       by its unique name.
    """

    def __init__(self, scope):
        """Initialize empty Vtree by providing its scope.

        :param scope: scope of the vtree (i.e. of its root region)
        """
        scope_set = frozenset(scope)
        if len(scope) != len(scope_set):
            raise ValueError("scope of vtree node/region cannot contain duplicate entries")
        self._scope = scope_set

        self._regions = []  # list of region names
        self._region_scopes = {}  # mapping from region name to its scope
        self._parent = {}  # mapping from region name to its parent region namme (None if root)
        self._children = {}  # mapping from region name to a list of its children region names
        self._leaves = {}  # mapping from region name to bool indicating whether the region is a leaf (i.e. has no
        # children)

        self._root = None  # name of the root region; this is not set initially

    @property
    def regions(self):
        return self._regions

    @property
    def scope(self):
        return self._scope

    @property
    def region_scopes(self):
        return self._region_scopes

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def leaves(self):
        return self._leaves

    @property
    def root(self):
        return self._root

    def set_root(self, root_region):
        """Set the specified region as the root region of the Vtree.

        :param root_region: name of region to be set as root
        """
        if root_region not in self.regions:
            raise ValueError("vtree region not found")
        elif self.region_scopes[root_region] != self.scope:
            raise ValueError("proposed root region does not have same scope as vtree")
        self._root = root_region


    def create_vtree_region(self, scope, leaf, name = None):
        """Creates a new vtree region with given scope and name and adds it to the Vtree.

        :param scope: iterable of elements defining the scope function for this vtree region
        :param leaf: true if the region is a leaf
        :param name: name of new vtree region
        :return: newly created region name
        """
        scope_set = frozenset(scope)
        # allow duplicate empty scopes; this is to allow lazy operations
        # if scope_set and scope_set in self.region_scopes.values():
        #     raise ValueError("vtree node/region with same scope already exists")
        if not scope_set.issubset(self.scope):
            raise ValueError("scope of vtree node/region must be contained in the scope of the entire vtree")
        if len(scope) != len(scope_set):
            raise ValueError("scope of vtree node/region cannot contain duplicate entries")

        if name is None:
            new_region = scope_set  # use scope as default name
        elif name in self.regions:
            raise ValueError("vtree node/region already exists")
        else:
            new_region = name

        self._regions.append(new_region)
        self._region_scopes[new_region] = scope_set
        self._parent[new_region] = None
        self._children[new_region] = []

        self._leaves[new_region] = leaf
        return new_region

    def _remove_vtree_region(self, name):
        """Removes a vtree region with given name and detaches it from its parents/children.
        (Only for internal use, as this will destroy the structure, e.g. scope partitioning in general)

        :param name: name of vtree node to remove
        """
        if self.parent[name] is not None:
            self._children[self.parent[name]].remove(name)
        self._parent.pop(name)
        for child_name in self.children:
            self._parent[child_name] = None
        self._children.pop(name)

        self._leaves.pop(name)

        self._region_scopes.pop(name)
        self._regions.remove(name)


    def update_children(self, parent_name, children_names):
        """Updates the children of the given vtree region.

        :param parent_name: name of the parent region to be updated
        :param children_names: List of names of the vtree regions forming the children
        """
        if parent_name not in self.regions or not set(children_names).issubset(self.regions):
            raise IndexError("vtree node/region not found")

        parent_scope = self.region_scopes[parent_name]
        children_scopes = [self.region_scopes[child_name] for child_name in children_names]

        # Check that scopes of children and parent are valid
        children_joined_scope = [var for child_scope in children_scopes for var in child_scope]
        children_total_scope = frozenset(children_joined_scope)
        if len(children_joined_scope) != len(children_total_scope) \
                or children_total_scope != parent_scope:
            raise ValueError("scopes of children do not partition scope of parent vtree node/region")

        # Add the new children to the parent
        self._children[parent_name] = [child_name for child_name in children_names]
        self._parent.update([(child_name, parent_name) for child_name in children_names])


    def make_layers(self):
        """Creates list of layers, where each layer consists of a list of vtree region names. The list has the property
        that any vtree region has its parent in the previous layer.

        :return: list of layers
        """
        if self.root is None:
            raise RuntimeError("cannot make layers as root not set for vtree")
        layers = [[self.root,]]
        while layers[-1]:
            layers.append([child_name for region_name in layers[-1] for child_name in self.children[region_name]])

        # last element is []
        return layers[:-1]

    def generate_random_binary_vtree(self):
        """Generates a random, balanced binary vtree, by recursively and randomly splitting each scope into near-halves.

        :return: list of layers following the generated structure
        """
        if self.root is None:
            root_region = self.create_vtree_region(self.scope, leaf=False)
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
                        child_name = self.create_vtree_region(scope=sub_scope, name=sub_scope, leaf=(len(sub_scope) == 1))
                        children_names.append(child_name)
                    self.update_children(region_name, children_names)  # connect new regions
                    new_layer += scope_partition
            layers.append(new_layer)

        return layers[:-1]

    ############ OPERATIONS ########################

    def lazy_marg_compute_region_dict(self, region, marg_set):
        """Computes information relating to the marginalized version of the given region, in the output vtree.

        If region is None, treats it as a "dummy region" with no scope, a single sum node, and a label with empty scop

        :param region: name of region in vtree to be marginalized
        :param marg_set: set of variables to be marginalized out
        :return: dictionary containing information for performing the update to the output vtree.
        """

        output_scope = self.region_scopes[region].difference(marg_set)

        return {"region": region,
                "leaf": self.leaves[region],
                "output_scope": output_scope,
                "create_args": {"scope": output_scope, "name": uuid.uuid4(), "leaf": self.leaves[region]}
        }

    def lazy_marg_execute_update(self, output_region_dict):
        """Given information relating to the marginalized version of the input region (obtained using
        lazy_marg_compute_region_dict), executes the update in this vtree.

        Lazy version, meaning that we do not remove regions with empty scope. As a result, we need to generate unique
        names which are not scopes (because there can be regions with the same scope now, either empty scope or a
        region with one child), as well as allow duplicate scopes in create_vtree_region.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """

        output_region = self.create_vtree_region(**output_region_dict["create_args"])

        if len(output_region_dict["output_region_children"]) > 0:
            self.update_children(output_region, output_region_dict["output_region_children"])

        return output_region

    def lazy_prod_compute_region_dict(self, region):
        """Compute information necessary for computing the product of the given region with another region.

        If region is None, imitates a leaf region with empty scope.

        :param region: name of region in vtree for which a product is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """

        if region is None:
            region_dict = {
                "region": region,
                "circuit_scope": self.scope,
                "scope": frozenset([]),
                "is_leaf": True
            }
        else:
            region_dict = {
                "region": region,
                "circuit_scope": self.scope,
                "scope": self.region_scopes[region],
                "is_leaf": self.leaves[region]
            }

        if not region_dict["is_leaf"]:
            assert len(self.children[region]) == 2
            region_dict["left_child"], region_dict["right_child"] = self.children[region][0], self.children[region][1]
            region_dict["left_child_scope"], region_dict["right_child_scope"] = self.region_scopes[
                                                                                    region_dict["left_child"]], \
                                                                                self.region_scopes[
                                                                                    region_dict["right_child"]]

        return region_dict

    def lazy_prod_compute_output_region_dict(self, region_dict1, region_dict2):
        """Compute information necessary for computing the product of two regions, where we have information from each
        of the individual reasons from lazy_prod_compute_region_dict.

        :param region_dict1, region_dict2: dicts with information on the individual reginos
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = {}

        output_region_dict["output_region_scope"] = region_dict1["scope"].union(region_dict2["scope"])
        output_region_dict["is_leaf"] = region_dict1['is_leaf'] and region_dict2['is_leaf']
        output_region_dict["name"] = uuid.uuid4()

        output_region_dict["create_args"] = {"scope": output_region_dict["output_region_scope"],
                                             "leaf": output_region_dict["is_leaf"],
                                             "name": output_region_dict["name"]}

        csv = region_dict1["circuit_scope"].intersection(region_dict2["circuit_scope"])  # circuit shared variables




        if not output_region_dict["is_leaf"]:
            # check if the scopes of the children of the regions match
            if (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (
                    region_dict1["left_child_scope"].intersection(csv) == region_dict2["left_child_scope"].intersection(
                csv) and
                    region_dict1["right_child_scope"].intersection(csv) == region_dict2["right_child_scope"].intersection(
                csv)):
                output_region_dict["left_children"] = (region_dict1["left_child"], region_dict2["left_child"])
                output_region_dict["right_children"] = (region_dict1["right_child"], region_dict2["right_child"])
            elif (not region_dict1["is_leaf"] and not region_dict2["is_leaf"]) \
                    and (
                    region_dict1["left_child_scope"].intersection(csv) == region_dict2["right_child_scope"].intersection(
                csv) and
                    region_dict1["right_child_scope"].intersection(csv) == region_dict2["left_child_scope"].intersection(
                csv)):
                output_region_dict["left_children"] = (region_dict1["left_child"], region_dict2["right_child"])
                output_region_dict["right_children"] = (region_dict1["right_child"], region_dict2["left_child"])
            # check if one region's scope matches the scope of a child of the other region
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "left_child_scope"].intersection(csv):
                output_region_dict["left_children"] = (region_dict1["region"], region_dict2["left_child"])
                output_region_dict["right_children"] = (None, region_dict2["right_child"])
            elif not region_dict2["is_leaf"] and region_dict1["scope"].intersection(csv) == region_dict2[
                "right_child_scope"].intersection(csv):
                output_region_dict["left_children"] = (None, region_dict2["left_child"])
                output_region_dict["right_children"] = (region_dict1["region"], region_dict2["right_child"])
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "left_child_scope"].intersection(csv):
                output_region_dict["left_children"] = (region_dict1["left_child"], region_dict2["region"])
                output_region_dict["right_children"] = (region_dict1["right_child"], None)
            elif not region_dict1["is_leaf"] and region_dict2["scope"].intersection(csv) == region_dict1[
                "right_child_scope"].intersection(csv):
                output_region_dict["left_children"] = (region_dict1["left_child"], None)
                output_region_dict["right_children"] = (region_dict1["right_child"], region_dict2["region"])
            else:
                if region_dict1["is_leaf"] or region_dict2["is_leaf"]:
                    print(region_dict1, region_dict2)
                    raise ValueError("not compatible; leaf cannot be decomposed")
                else:
                    print(region_dict1, region_dict2)
                    raise ValueError("not compatible; regions cannot be decomposed")

        return output_region_dict



    def lazy_prod_execute_update(self, output_region_dict):
        """Given information relating to the product region (obtained using lazy_prod_compute_output_region_dict),
        executes the update in this vtree.

        Lazy version, meaning that we do not remove regions with empty scope. As a result, we need to generate unique
        names which are not scopes (because there can be regions with the same scope now, either empty scope or a
        region with one child), as well as allow duplicate scopes in create_vtree_region.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """

        output_region = self.create_vtree_region(**output_region_dict["create_args"])

        if not output_region_dict["is_leaf"]:
            self.update_children(output_region, [output_region_dict["output_left_child"], output_region_dict["output_right_child"]])

        return output_region

    def cond_compute_region_dict(self, region):
        """Compute information necessary for computing the conditional of a region.

        :param region: name of region in vtree for which a conditional is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """
        scope = self.region_scopes[region]


        output_region_dict = {"region": region,
                "leaf": self.leaves[region],
                "output_scope": self.region_scopes[region],
                "create_args": {"scope": self.region_scopes[region], "name": uuid.uuid4(), "leaf": self.leaves[region]},
        }

        if not output_region_dict["leaf"]:
            assert len(self.children[region]) == 2
            output_region_dict["left_child"] = self.children[region][0]
            output_region_dict["right_child"] = self.children[region][1]

        return output_region_dict

    def cond_execute_update(self, output_region_dict):
        """Given information relating to the conditional region (obtained using cond_compute_region_dict),
        executes the update in this vtree.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """
        output_region = self.create_vtree_region(**output_region_dict["create_args"])

        if not output_region_dict["leaf"]:
            self.update_children(output_region, [output_region_dict["output_left_child"], output_region_dict["output_right_child"]])

        return output_region


class MDVtree(Vtree):
    def __init__(self, scope):
        """Defines a MD-Vtree, including its root node/region (we use the nomenclature "region" henceforth") and a
        list of region names, together with their scopes and labels.

        By default, if a name is not given, the scope of a md-vtree region is used as its name.

        :param scope: scope of the md-vtree (i.e. of its root region)
        :param root_name: name for the root region
        """
        super().__init__(scope = scope)
        self._region_labels = {}

    @property
    def region_labels(self):
        return self._region_labels

    def update_label(self, region, label):
        """Given a region name, updates its label in the vtree.

        :param region: name of region to be updated
        :param label: new label for the region
        """
        if not label.is_contained_in_scope(self.region_scopes[region]):
            raise ValueError("Label must be contained within the scope of the vtree node/region")
        self._region_labels[region] = label

    def create_vtree_region(self, scope, leaf, name = None, label = VtreeLabel(VtreeLabel.universal_set)):
        """Creates a new md-vtree region with given scope, name and label, and adds it to the md-vtree.

        :param scope: iterable of elements defining the scope function for this md-vtree region
        :param leaf: true if region is a leaf
        :param name: name of the new region (optional)
        :param label: label for the new region (universal set by default)
        :return: newly created md-vtree region name
        """
        new_region = super().create_vtree_region(scope=scope, leaf=leaf, name=name)
        self.update_label(new_region, label)
        return new_region

    def _remove_vtree_region(self, name):
        """Removes a md-vtree region with given name and detaches it from its parents/children.

        :param name: name of vtree region to be removed
        :return: newly created region name
        """
        super()._remove_vtree_region(name)
        self.region_labels.pop(name)



    ########### OPERATIONS #################################


    def lazy_marg_compute_region_dict(self, region, marg_set):
        """Computes information relating to the marginalized version of the given region, in the output vtree.

        For MDVtrees, additionally computes the label after marginalization.

        :param region: name of region in vtree to be marginalized
        :param marg_set: set of variables to be marginalized out
        :return: dictionary containing information for performing the update to the output vtree.
        """

        output_region_dict = super().lazy_marg_compute_region_dict(region, marg_set)

        if self.region_labels[region].overlaps_with_mdset(marg_set):
            output_region_label = VtreeLabel(VtreeLabel.universal_set)
        else:
            output_region_label = self.region_labels[region]

        output_region_dict["create_args"]["label"] = output_region_label

        return output_region_dict

    def lazy_marg_execute_update(self, output_region_dict):
        """Given information relating to the marginalized version of the input region (obtained using
        lazy_marg_compute_region_dict), executes the update in this vtree.

        For MDVtrees, also initializes the label of the new region.

        :param output_region_dict: dictionary containing information for performing the update to the vtree.
        :return: name of the newly created region in this vtree
        """

        output_region = super().lazy_marg_execute_update(output_region_dict)

        return output_region

    def lazy_prod_compute_region_dict(self, region):
        """Compute information necessary for computing the product of the given region with another region.

        If region is None, imitates a leaf region with empty scope and emptyset label.

        :param region: name of region in vtree for which a product is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """
        region_dict = super().lazy_prod_compute_region_dict(region)

        if region is None:
            # "dummy region"
            region_dict["label"] = VtreeLabel(frozenset([]))
        else:
            region_dict["label"] = self.region_labels[region]

        return region_dict

    def lazy_prod_compute_output_region_dict(self, region_dict1, region_dict2):
        """Compute information necessary for computing the product of two regions, where we have information from each
        of the individual reasons from lazy_prod_compute_region_dict.

        :param region_dict1, region_dict2: dicts with information on the individual reginos
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = super().lazy_prod_compute_output_region_dict(region_dict1, region_dict2)

        output_region_dict["create_args"]["label"] = VtreeLabel.union(region_dict1["label"], region_dict2["label"])

        return output_region_dict

    def cond_compute_region_dict(self, region):
        """Compute information necessary for computing the conditional of a region.

        :param region: name of region in vtree for which a conditional is to be computed
        :return: dictionary containing information for performing the update to the output vtree
        """
        output_region_dict = super().cond_compute_region_dict(region)

        output_region_label = self.region_labels[region]

        output_region_dict["create_args"]["label"] = output_region_label

        return output_region_dict