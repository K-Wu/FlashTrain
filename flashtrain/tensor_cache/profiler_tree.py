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
class ModuleReentrantContext:
    module_id: int
    reenter_count: int


class ProfilerTreeNode:
    scope: ModuleReentrantContext | ActivationContext
    parent: Optional["ProfilerTreeNode"]
    children: list["ProfilerTreeNode"]

    def __init__(
        self,
        scope: ModuleReentrantContext | ActivationContext,
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

    roots: list[ProfilerTreeNode] = []
    nodes: dict[
        ModuleReentrantContext | ActivationContext, ProfilerTreeNode
    ] = {}

    def __init__(self):
        self.roots = []
        self.nodes = dict()

    def add_bottom_to_root_path(
        self, scopes: list[ModuleReentrantContext | ActivationContext]
    ):
        current_children_list = self.roots
        # Nodes are in the order from the bottom to the top
        for scope in reversed(scopes):
            if scope not in self.nodes:
                new_node = ProfilerTreeNode(scope, None)
                self.nodes[scope] = new_node
                current_children_list.append(new_node)
            current_children_list = self.nodes[scope].children
