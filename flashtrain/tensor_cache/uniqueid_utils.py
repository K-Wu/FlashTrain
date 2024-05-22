from weakref import WeakKeyDictionary
from collections import defaultdict
from uuid import uuid4


class UniqueIdMap(WeakKeyDictionary):
    def __init__(self, dict=None):
        super().__init__(self)
        # replace data with a defaultdict to generate uuids
        self.data = defaultdict(uuid4)
        if dict is not None:
            self.update(dict)


uniqueidmap = UniqueIdMap()


def uniqueid(obj):
    # Adapted from https://stackoverflow.com/questions/52096582/how-unique-is-pythons-id
    # uniqueid() won't cause GPU operations when applied to a tensor. Reference: https://stackoverflow.com/questions/52096582/how-unique-is-pythons-id
    # https://github.com/pytorch/pytorch/issues/2569

    """Produce a unique-throughout-program-life-time integer id for the object.

    Object must me *hashable*. Id is a UUID and should be unique
    across Python invocations.

    """
    return uniqueidmap[obj].int
