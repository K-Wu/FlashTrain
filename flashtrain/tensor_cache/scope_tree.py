from dataclasses import dataclass
from typing import Optional
from ..logger import logger


@dataclass(frozen=True)
class ActivationContext:
    ctx_id: int
    # sequence_id: int
    # The following two store the site of the caller of torch.utils.checkpoint
    # caller_filename: str
    # caller_lineno: int


@dataclass(frozen=True)
class SavedActivationContext:
    seq_id: int


@dataclass(frozen=True)
class ModuleReentrantContext:
    module_id: int
    reenter_count: int


class ScopeTreeNode:
    scope: ModuleReentrantContext | SavedActivationContext
    parent: Optional["ScopeTreeNode"]
    children: list["ScopeTreeNode"]

    def __init__(
        self,
        scope: ModuleReentrantContext | SavedActivationContext,
        parent: Optional["ScopeTreeNode"],
    ):
        self.scope = scope
        self.parent = parent
        self.children = []


def find_node_index_in(nodes: list[ScopeTreeNode], node: ScopeTreeNode):
    for idx, n in enumerate(nodes):
        if n.scope == node.scope:
            return idx
    raise ValueError(f"Cannot find f{node.scope} in the list")


class ScopeTree:
    """
    This is a tree with TransformerLayer in the first level (roots), and MLP and attention blocks in the second level.
    When there is ActivationContext, aside from the above three tracked type of modules, their parent ActivationContext scope is also added to the tree.
    So ultimately there may be more than two levels.
    """

    roots: list[ScopeTreeNode]
    nodes: dict[ModuleReentrantContext | SavedActivationContext, ScopeTreeNode]

    def __init__(self):
        self.roots = []
        self.nodes = dict()

    def is_leaf(
        self, module: ModuleReentrantContext | SavedActivationContext
    ) -> bool:
        if module in self.nodes:
            return len(self.nodes[module].children) == 0
        raise ValueError(f"Cannot find {module} in the tree")

    def add_bottom_to_root_path(
        self, scopes: list[ModuleReentrantContext | SavedActivationContext]
    ):
        current_root = None
        current_children_list = self.roots
        # Nodes are in the order from the bottom to the top
        for scope in reversed(scopes):
            if scope not in self.nodes:
                new_node = ScopeTreeNode(scope, current_root)
                current_root = new_node
                self.nodes[scope] = new_node
                current_children_list.append(new_node)
            current_root = self.nodes[scope]
            current_children_list = self.nodes[scope].children

    def print_nodes(self):
        scopes = []
        for node in self.walk_nodes():
            scopes.append(node.scope)
        logger.critical(f"Nodes in post-order: {scopes}")

    def get_previous_nodes(self, node: ScopeTreeNode) -> list[ScopeTreeNode]:
        """Get all the nodes happen before the specified nodes.
        At each level, return all the nodes before the X-order parent (including X=0 which is itself).
        For example, if the specified node is the MLP block in the second Transformer layer,
        return the first transformer layer (or activation context) and the attention block in the second Transformer layer (or activation context)
        """
        scope: ModuleReentrantContext | SavedActivationContext = node.scope

        current_node = self.nodes[scope]
        previous_nodes = []
        while current_node is not None:
            if current_node.parent is not None:
                earlier_siblings = current_node.parent.children[
                    : find_node_index_in(
                        current_node.parent.children, current_node
                    )
                ]
            else:
                earlier_siblings = self.roots[
                    : find_node_index_in(self.roots, current_node)
                ]
            previous_nodes = (
                earlier_siblings + previous_nodes
            )  # Put higher level nodes first
            current_node = current_node.parent

        return previous_nodes

    def walk_nodes_(self, node: ScopeTreeNode):
        # Post-order traversal
        for child in node.children:
            yield from self.walk_nodes_(child)
        yield node

    def walk_nodes(self):
        for root in self.roots:
            yield from self.walk_nodes_(root)


if __name__ == "__main__":
    #    1        2
    #   / \     / \
    #  3   4   5   6
    # / \ /    /  /
    # 7 8 9  10 11
    tree = ScopeTree()
    tree.add_bottom_to_root_path(
        [
            ModuleReentrantContext(7, 1),
            ModuleReentrantContext(3, 1),
            ModuleReentrantContext(1, 1),
        ]
    )
    tree.add_bottom_to_root_path(
        [
            ModuleReentrantContext(8, 1),
            ModuleReentrantContext(3, 1),
            ModuleReentrantContext(1, 1),
        ]
    )
    tree.add_bottom_to_root_path(
        [
            ModuleReentrantContext(9, 1),
            ModuleReentrantContext(4, 1),
            ModuleReentrantContext(1, 1),
        ]
    )
    tree.add_bottom_to_root_path(
        [
            ModuleReentrantContext(10, 1),
            ModuleReentrantContext(5, 1),
            ModuleReentrantContext(2, 1),
        ]
    )
    tree.add_bottom_to_root_path(
        [
            ModuleReentrantContext(11, 1),
            ModuleReentrantContext(6, 1),
            ModuleReentrantContext(2, 1),
        ]
    )

    for node in tree.walk_nodes():
        print(node.scope)
    # 7, 8, 3, 9, 4, 1, 10, 5, 11, 6, 2
