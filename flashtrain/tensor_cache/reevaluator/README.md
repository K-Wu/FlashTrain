


Reevaluation == checkpoint + supply the recomputed output to the next layer during backward propagation. 

The reevaluator implementation basically is a rearrangement of the logic in the activation checkpoint function.

Step 1. The forward propagation logic is the same as activation checkpoint function. However, in the forward propagation, we also need to register the recomputation function in the tensor cache: when the unpack hook is triggered for the output of the checkpointed layer, we need to do the recomputation.

Step 2. The original static method backward() is partitioned into two part, the recomputation, and the real backward propagation. The recomputation is triggered by the unpack hook, and the real backward propagation is defined as the new static method backward().