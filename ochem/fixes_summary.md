# GNN Model Shape Mismatch Fixes

## Problem Description

Several GNN models were failing during inference with the error:
```
ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis. 
Received: input_shape=[(None, None, 128), (None, 2)]
```

This occurred because the models were trying to concatenate:
- **Node features** (ragged tensor with shape `(None, None, 128)`)
- **Graph descriptors** (regular tensor with shape `(None, 2)`)

## Root Cause

The issue was in the graph state fusion logic where models were trying to concatenate graph descriptors with node features **before** pooling to graph level. This creates a shape mismatch because:
- Node features are ragged tensors (variable number of nodes per graph)
- Graph descriptors are regular tensors (one descriptor per graph)

## Solution

Move the graph state fusion to happen **after** pooling to graph level, when both tensors have compatible shapes.

## Fixed Models

### 1. ContrastiveGNN (`kgcnn/literature/ContrastiveGNN/_make.py`)
**Issue**: Line 583-584 - Concatenating ragged pooled nodes with graph embedding
**Fix**: Added tensor conversion before concatenation
```python
# Before
pooled_nodes = PoolingNodes(pooling_method="sum")(n)
n = ks.layers.Concatenate()([pooled_nodes, graph_embedding])

# After  
pooled_nodes = PoolingNodes(pooling_method="sum")(n)
pooled_nodes = ks.layers.Lambda(
    lambda x: tf.RaggedTensor.to_tensor(x) if hasattr(x, 'to_tensor') else x
)(pooled_nodes)
n = ks.layers.Concatenate()([pooled_nodes, graph_embedding])
```

### 2. DMPNNAttention (`kgcnn/literature/DMPNNAttention/_make.py`)
**Issue**: Line 147-148 - Concatenating node features with graph embedding
**Fix**: Moved graph state fusion after pooling
```python
# Before
if use_graph_state and graph_embedding is not None:
    n = ks.layers.Concatenate()([n, graph_embedding])

# After
if output_embedding == "graph":
    out = PoolingNodesDMPNNAttention(**pooling_args)(n)
    if use_graph_state and graph_embedding is not None:
        out = ks.layers.Concatenate()([out, graph_embedding])
```

### 3. GRPE (`kgcnn/literature/GRPE/_make.py`)
**Issue**: Line 100-101 - Concatenating node features with graph embedding
**Fix**: Moved graph state fusion after pooling
```python
# Before
if use_graph_state and graph_embedding is not None:
    n = ks.layers.Concatenate()([n, graph_embedding])

# After
if output_embedding == "graph":
    out = PoolingNodesGRPE(pooling_method="sum")(n)
    if use_graph_state and graph_embedding is not None:
        out = ks.layers.Concatenate()([out, graph_embedding])
```

### 4. TransformerGAT (`kgcnn/literature/TransformerGAT/_make.py`)
**Issue**: Line 95-96 - Concatenating node features with graph embedding
**Fix**: Moved graph state fusion after pooling
```python
# Before
if use_graph_state and graph_embedding is not None:
    n = ks.layers.Concatenate()([n, graph_embedding])

# After
if output_embedding == "graph":
    out = PoolingNodesTransformerGAT(pooling_method="sum")(n)
    if use_graph_state and graph_embedding is not None:
        out = ks.layers.Concatenate()([out, graph_embedding])
```

### 5. EGAT (`kgcnn/literature/EGAT/_make.py`)
**Issue**: Line 100-101 - Concatenating node features with graph embedding
**Fix**: Moved graph state fusion after pooling
```python
# Before
if use_graph_state and graph_embedding is not None:
    n = ks.layers.Concatenate()([n, graph_embedding])

# After
if output_embedding == "graph":
    out = PoolingNodesEGAT(pooling_method="sum")(n)
    if use_graph_state and graph_embedding is not None:
        out = ks.layers.Concatenate()([out, graph_embedding])
```

### 6. ExpC (`kgcnn/literature/ExpC/_make.py`)
**Issue**: Line 95-96 - Concatenating node features with graph embedding
**Fix**: Moved graph state fusion after pooling
```python
# Before
if use_graph_state and graph_embedding is not None:
    n = ks.layers.Concatenate()([n, graph_embedding])

# After
if output_embedding == "graph":
    out = PoolingNodesExpC(pooling_method="sum")(n)
    if use_graph_state and graph_embedding is not None:
        out = ks.layers.Concatenate()([out, graph_embedding])
```

## Models That Were Already Correct

The following models already handled graph state fusion correctly (after pooling):
- **GIN** (`kgcnn/literature/GIN/_make.py`)
- **GAT** (`kgcnn/literature/GAT/_make.py`)
- **AttentiveFP** (`kgcnn/literature/AttentiveFP/_make.py`)
- **PNA** (`kgcnn/literature/PNA/_make.py`)
- **RGCN** (`kgcnn/literature/RGCN/_make.py`)

## Testing

A test script `test_contrastive_gin_fix.py` has been created to verify that the ContrastiveGIN model works with the fix.

## Expected Impact

These fixes should resolve the inference failures for:
- ContrastiveGIN (test inference)
- ContrastiveGAT (test inference) 
- ContrastiveGATv2 (test inference)
- DMPNNAttention (test inference)
- rGINE (test inference)
- GRPE (test inference)

The models should now be able to properly handle graph descriptors during inference without shape mismatch errors. 