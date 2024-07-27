from dataclasses import dataclass
from typing import Optional


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


class ProfilerTreeNode:
    scope: ModuleReentrantContext | SavedActivationContext
    parent: Optional["ProfilerTreeNode"]
    children: list["ProfilerTreeNode"]

    def __init__(
        self,
        scope: ModuleReentrantContext | SavedActivationContext,
        parent: Optional["ProfilerTreeNode"],
    ):
        self.scope = scope
        self.parent = parent
        self.children = []


class ProfilerTree:
    """
    This is a tree with TransformerLayer in the first level (roots), and MLP and attention blocks in the second level.
    When there is ActivationContext, aside from the above three tracked type of modules, their parent ActivationContext scope is also added to the tree.
    So ultimately there may be more than two levels.
    """

    roots: list[ProfilerTreeNode]
    nodes: dict[
        ModuleReentrantContext | SavedActivationContext, ProfilerTreeNode
    ]

    def __init__(self):
        self.roots = []
        self.nodes = dict()

    def add_bottom_to_root_path(
        self, scopes: list[ModuleReentrantContext | SavedActivationContext]
    ):
        current_root = None
        current_children_list = self.roots
        # Nodes are in the order from the bottom to the top
        for scope in reversed(scopes):
            if scope not in self.nodes:
                new_node = ProfilerTreeNode(scope, current_root)
                current_root = new_node
                self.nodes[scope] = new_node
                current_children_list.append(new_node)
            current_children_list = self.nodes[scope].children

    def get_previous_nodes(
        self, scope: ModuleReentrantContext | SavedActivationContext
    ) -> list[ProfilerTreeNode]:
        """Get all the nodes happen before the specified nodes.
        At each level, return all the nodes before the X-order parent (including X=0 which is itself).
        For example, if the specified node is the MLP block in the second Transformer layer,
        return the first transformer layer (or activation context) and the attention block in the second Transformer layer (or activation context)
        """

        current_node = self.nodes[scope]
        previous_nodes = []
        while current_node is not None:
            if current_node.parent is not None:
                earlier_siblings = current_node.parent.children[
                    : current_node.parent.children.index(current_node)
                ]
            else:
                earlier_siblings = self.roots[: self.roots.index(current_node)]
            previous_nodes = (
                earlier_siblings + previous_nodes
            )  # Put higher level nodes first
            current_node = current_node.parent

        return previous_nodes
