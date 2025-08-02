# KGCNN-Keras-Unlocked: Graph Neural Networks for Molecular Property Prediction

This repository contains an extended implementation of KGCNN (Keras Graph Convolutional Neural Networks) with unified descriptor handling and support for 3D molecular representations. All models have been enhanced with continuous float descriptor integration using the unified input system from `kgcnn/utils/input_utils.py`.

## Implemented Models

### 2D Molecular Graph Models

#### **GAT (Graph Attention Network)**
- **Architecture**: Attention-based message passing with multi-head attention mechanisms
- **Literature**: Veličković et al. (2018) - "Graph Attention Networks"
- **Main Features**: Uses attention weights to dynamically determine the importance of neighboring nodes during message aggregation. Implements multi-head attention for capturing different types of relationships in molecular graphs.

#### **GATv2 (Graph Attention Network v2)**
- **Architecture**: Improved attention mechanism with learnable attention coefficients
- **Literature**: Brody et al. (2021) - "How Attentive are Graph Attention Networks?"
- **Main Features**: Addresses the limitations of GAT by making attention coefficients learnable, providing more expressive attention mechanisms for molecular property prediction.

#### **GCN (Graph Convolutional Network)**
- **Architecture**: Spectral graph convolution with Chebyshev polynomial approximation
- **Literature**: Kipf & Welling (2016) - "Semi-Supervised Classification with Graph Convolutional Networks"
- **Main Features**: Applies spectral graph convolutions to aggregate information from neighboring nodes, providing a foundation for many subsequent graph neural network architectures.

#### **GIN (Graph Isomorphism Network)**
- **Architecture**: Message passing with injective aggregation functions
- **Literature**: Xu et al. (2019) - "How Powerful are Graph Neural Networks?"
- **Main Features**: Designed to be as powerful as the Weisfeiler-Lehman graph isomorphism test, using injective aggregation functions to maintain discriminative power.

#### **GraphSAGE**
- **Architecture**: Inductive node embedding with neighborhood sampling
- **Literature**: Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs"
- **Main Features**: Implements inductive learning by sampling fixed-size neighborhoods and aggregating information hierarchically, enabling generalization to unseen nodes.

#### **PNA (Principal Neighbourhood Aggregation)**
- **Architecture**: Multi-scale aggregation with degree-based normalization
- **Literature**: Corso et al. (2020) - "Principal Neighbourhood Aggregation for Graph Nets"
- **Main Features**: Uses multiple aggregation functions and degree-based normalization to capture complex graph patterns, improving expressiveness over standard message passing.

#### **AddGNN (Additive Graph Neural Network)**
- **Architecture**: Additive aggregation with skip connections
- **Literature**: Chen et al. (2020) - "Additive Graph Neural Networks"
- **Main Features**: Implements additive aggregation functions with residual connections, providing better gradient flow and improved performance on molecular property prediction.

#### **DGIN (Directed Graph Isomorphism Network)**
- **Architecture**: Direction-aware message passing with edge direction encoding
- **Literature**: Bevilacqua et al. (2021) - "Equivariant Subgraph Aggregation Networks"
- **Main Features**: Extends GIN to handle directed graphs by incorporating edge direction information in the message passing process.

#### **rGIN (Random Features GIN)**
- **Architecture**: GIN with random feature augmentation
- **Literature**: Sato et al. (2020) - "Random Features Strengthen Graph Neural Networks"
- **Main Features**: Enhances GIN's expressiveness by incorporating random features during message passing, improving performance on molecular property prediction tasks.

#### **GINE (Graph Isomorphism Network with Edge Features)**
- **Architecture**: GIN extended with explicit edge feature handling
- **Literature**: Hu et al. (2020) - "Strategies for Pre-training Graph Neural Networks"
- **Main Features**: Extends GIN to explicitly handle edge features, improving representation learning for molecular graphs with rich edge information.

#### **rGINE (Random Features GINE)**
- **Architecture**: GINE with random feature augmentation
- **Literature**: Extension of rGIN and GINE
- **Main Features**: Combines the benefits of random feature augmentation with explicit edge feature handling for enhanced molecular representation learning.

### Attention-Based Models

#### **AttentiveFP**
- **Architecture**: Attention-based fingerprinting with gated recurrent units
- **Literature**: Xiong et al. (2020) - "Pushing the Boundaries of Molecular Representation for Drug Discovery"
- **Main Features**: Uses attention mechanisms to create molecular fingerprints, focusing on the most relevant substructures for property prediction.

#### **AttentiveFPPlus**
- **Architecture**: Enhanced AttentiveFP with improved attention mechanisms
- **Literature**: Extension of AttentiveFP
- **Main Features**: Improves upon AttentiveFP with better attention mechanisms and enhanced molecular representation learning.

#### **CoAttentiveFP (Co-Attention AttentiveFP)**
- **Architecture**: Co-attention mechanism for node-edge interactions
- **Literature**: Extension of AttentiveFP with co-attention
- **Main Features**: Implements co-attention between nodes and edges, capturing complex interactions in molecular graphs.

#### **TransformerGAT**
- **Architecture**: Transformer-based graph attention with positional encoding
- **Literature**: Extension of GAT with transformer architecture
- **Main Features**: Combines transformer architecture with graph attention, providing powerful sequence modeling capabilities for graph-structured data.

#### **EGAT (Edge-aware Graph Attention)**
- **Architecture**: Edge-aware attention with explicit edge feature modeling
- **Literature**: Extension of GAT with edge awareness
- **Main Features**: Extends GAT to explicitly model edge features and edge-node interactions through specialized attention mechanisms.

#### **KAGAT (Knowledge-Aware Graph Attention)**
- **Architecture**: Knowledge-aware attention with domain-specific priors
- **Literature**: Extension of GAT with knowledge integration
- **Main Features**: Incorporates domain knowledge and chemical priors into the attention mechanism for improved molecular property prediction.

### Message Passing Models

#### **CMPNN (Communicating Message Passing Neural Network)**
- **Architecture**: Bidirectional message passing with communication channels
- **Literature**: Song et al. (2020) - "Communicating Message Passing Neural Networks"
- **Main Features**: Implements bidirectional message passing with explicit communication channels, enabling better information flow between nodes and edges.

#### **CMPNNPlus**
- **Architecture**: Enhanced CMPNN with improved communication mechanisms
- **Literature**: Extension of CMPNN
- **Main Features**: Improves upon CMPNN with better communication mechanisms and enhanced molecular representation learning.

#### **DMPNN (Directed Message Passing Neural Network)**
- **Architecture**: Direction-aware message passing with edge direction encoding
- **Literature**: Yang et al. (2019) - "Analyzing Learned Molecular Representations for Property Prediction"
- **Main Features**: Extends message passing to handle directed graphs by incorporating edge direction information in the message aggregation process.

#### **DMPNNAttention**
- **Architecture**: DMPNN with attention-based message aggregation
- **Literature**: Extension of DMPNN with attention
- **Main Features**: Combines DMPNN's direction awareness with attention mechanisms for improved molecular property prediction.

#### **NMPN (Neural Message Passing for Quantum Chemistry)**
- **Architecture**: Message passing designed for quantum chemistry applications
- **Literature**: Gilmer et al. (2017) - "Neural Message Passing for Quantum Chemistry"
- **Main Features**: Specifically designed for quantum chemistry applications, using message passing to predict molecular properties from atomic coordinates and features.

### 3D Molecular Models

#### **PAiNN (PaiNN: Equivariant Message Passing)**
- **Architecture**: Equivariant message passing with spherical harmonics
- **Literature**: Schütt et al. (2020) - "Equivariant message passing for the prediction of tensorial properties and molecular spectra"
- **Main Features**: Implements E(3)-equivariant message passing using spherical harmonics, enabling accurate prediction of tensorial molecular properties while preserving rotational and translational symmetries.

#### **Schnet (SchNet)**
- **Architecture**: Continuous-filter convolutional layers for 3D molecular representation
- **Literature**: Schütt et al. (2017) - "SchNet – A deep learning architecture for molecules and materials"
- **Main Features**: Uses continuous-filter convolutional layers to model quantum interactions between atoms, providing accurate predictions of molecular properties from 3D coordinates.

#### **DimeNetPP (DimeNet++)**
- **Architecture**: Directional message passing with spherical harmonics
- **Literature**: Klicpera et al. (2020) - "Directional Message Passing for Molecular Graphs"
- **Main Features**: Extends DimeNet with improved directional message passing using spherical harmonics, enabling better modeling of 3D molecular interactions and geometric relationships.

#### **HamNet (Hamiltonian Neural Network)**
- **Architecture**: Conformation-guided molecular representation with Hamiltonian dynamics
- **Literature**: Li et al. (2021) - "HamNet: Conformation-Guided Molecular Representation with Hamiltonian Neural Networks"
- **Main Features**: Uses Hamiltonian neural networks to learn conformation-aware molecular representations, incorporating physical constraints and geometric relationships.

#### **MoGAT (Multi-order Graph Attention Network)**
- **Architecture**: Multi-order attention for water solubility prediction
- **Literature**: Lee et al. (2023) - "Multi-order graph attention network for water solubility prediction and interpretation"
- **Main Features**: Implements multi-order attention mechanisms to capture complex molecular interactions at different scales, specifically designed for water solubility prediction.

#### **EGNN (E(n) Equivariant Graph Neural Networks)**
- **Architecture**: E(n)-equivariant graph neural networks with geometric vector perceptrons
- **Literature**: Satorras et al. (2021) - "E(n) Equivariant Graph Neural Networks"
- **Main Features**: Implements E(n)-equivariant graph neural networks that preserve rotational, translational, and permutational symmetries, enabling accurate 3D molecular modeling.

### Ensemble and Multi-Expert Models

#### **MultiGraphMoE (Multi-Graph Mixture of Experts)**
- **Architecture**: Mixture of experts with multiple graph neural network experts
- **Literature**: Extension of MoE for graph neural networks
- **Main Features**: Uses multiple specialized graph neural network experts with a gating mechanism to automatically select the most appropriate expert for each input, improving overall performance.

#### **MultiChem**
- **Architecture**: Multi-chemical representation learning with ensemble methods
- **Literature**: Extension for multi-chemical property prediction
- **Main Features**: Implements ensemble methods and multi-chemical representation learning to handle diverse molecular properties and chemical spaces.

#### **GraphGPS (Graph Generalization with Permutation Symmetry)**
- **Architecture**: Generalization framework with permutation symmetry preservation
- **Literature**: Rampášek et al. (2022) - "Recipe for a General, Powerful, Scalable Graph Transformer"
- **Main Features**: Provides a framework for building powerful and scalable graph transformers while preserving permutation symmetries, enabling generalization across different graph structures.

### Contrastive Learning Models

#### **ContrastiveGIN**
- **Architecture**: Contrastive learning with GIN backbone
- **Literature**: Extension of GIN with contrastive learning
- **Main Features**: Implements contrastive learning on top of GIN to learn better molecular representations by contrasting positive and negative pairs.

#### **ContrastiveGAT**
- **Architecture**: Contrastive learning with GAT backbone
- **Literature**: Extension of GAT with contrastive learning
- **Main Features**: Combines GAT's attention mechanisms with contrastive learning for improved molecular representation learning.

#### **ContrastiveGATv2**
- **Architecture**: Contrastive learning with GATv2 backbone
- **Literature**: Extension of GATv2 with contrastive learning
- **Main Features**: Extends GATv2 with contrastive learning capabilities for enhanced molecular property prediction.

#### **ContrastiveDMPNN**
- **Architecture**: Contrastive learning with DMPNN backbone
- **Literature**: Extension of DMPNN with contrastive learning
- **Main Features**: Implements contrastive learning on top of DMPNN to improve molecular representation learning through positive-negative pair comparisons.

#### **ContrastiveAttFP**
- **Architecture**: Contrastive learning with AttentiveFP backbone
- **Literature**: Extension of AttentiveFP with contrastive learning
- **Main Features**: Combines AttentiveFP's attention-based fingerprinting with contrastive learning for enhanced molecular property prediction.

#### **ContrastiveAddGNN**
- **Architecture**: Contrastive learning with AddGNN backbone
- **Literature**: Extension of AddGNN with contrastive learning
- **Main Features**: Extends AddGNN with contrastive learning capabilities for improved molecular representation learning.

#### **ContrastiveDGIN**
- **Architecture**: Contrastive learning with DGIN backbone
- **Literature**: Extension of DGIN with contrastive learning
- **Main Features**: Implements contrastive learning on top of DGIN to enhance molecular representation learning through positive-negative pair comparisons.

#### **ContrastivePNA**
- **Architecture**: Contrastive learning with PNA backbone
- **Literature**: Extension of PNA with contrastive learning
- **Main Features**: Combines PNA's multi-scale aggregation with contrastive learning for improved molecular property prediction.

#### **ContrastiveMoE**
- **Architecture**: Contrastive learning with Mixture of Experts
- **Literature**: Extension of MoE with contrastive learning
- **Main Features**: Implements contrastive learning on top of mixture of experts architecture for enhanced molecular representation learning.

#### **ContrastiveGNN**
- **Architecture**: General contrastive learning framework for graph neural networks
- **Literature**: General framework for contrastive learning on graphs
- **Main Features**: Provides a general framework for implementing contrastive learning on various graph neural network architectures.

### Specialized Models

#### **ExpC (Explainable Chemistry)**
- **Architecture**: Explainable molecular property prediction with attention visualization
- **Literature**: Extension for explainable AI in chemistry
- **Main Features**: Implements explainable AI techniques for molecular property prediction, providing interpretable predictions and attention visualizations.

#### **GRPE (Graph Relative Positional Encoding)**
- **Architecture**: Graph transformer with relative positional encoding
- **Literature**: Extension of graph transformers with positional encoding
- **Main Features**: Implements relative positional encoding for graph transformers, improving the model's ability to capture structural relationships in molecular graphs.

#### **AWARE (Attention-based Weighted Aggregation for Representation Enhancement)**
- **Architecture**: Attention-based weighted aggregation with representation enhancement
- **Literature**: Extension for enhanced molecular representation
- **Main Features**: Uses attention-based weighted aggregation to enhance molecular representations, improving the quality of learned features for property prediction.

#### **DHTNN (Deep Hamiltonian Neural Network)**
- **Architecture**: Deep Hamiltonian neural networks for molecular dynamics
- **Literature**: Extension of Hamiltonian neural networks
- **Main Features**: Implements deep Hamiltonian neural networks for molecular dynamics simulation and property prediction, incorporating physical constraints.

#### **DHTNNPlus**
- **Architecture**: Enhanced DHTNN with improved Hamiltonian modeling
- **Literature**: Extension of DHTNN
- **Main Features**: Improves upon DHTNN with better Hamiltonian modeling and enhanced molecular dynamics simulation capabilities.

### Hybrid and Ensemble Models

#### **AddGNN-PNA**
- **Architecture**: Hybrid of AddGNN and PNA with ensemble methods
- **Literature**: Combination of AddGNN and PNA
- **Main Features**: Combines AddGNN's additive aggregation with PNA's multi-scale aggregation in an ensemble framework for improved molecular property prediction.

#### **CMPNN-PNA**
- **Architecture**: Hybrid of CMPNN and PNA with ensemble methods
- **Literature**: Combination of CMPNN and PNA
- **Main Features**: Combines CMPNN's bidirectional message passing with PNA's multi-scale aggregation for enhanced molecular representation learning.

#### **RGCN-PNA**
- **Architecture**: Hybrid of RGCN and PNA with ensemble methods
- **Literature**: Combination of RGCN and PNA
- **Main Features**: Combines RGCN's relational modeling with PNA's multi-scale aggregation for improved molecular property prediction.

#### **MoE (Mixture of Experts)**
- **Architecture**: Mixture of experts with multiple specialized models
- **Literature**: Jacobs et al. (1991) - "Adaptive Mixtures of Local Experts"
- **Main Features**: Implements a mixture of experts framework with multiple specialized graph neural network models and a gating mechanism for automatic expert selection.

#### **ConfigurableMoE**
- **Architecture**: Configurable mixture of experts with flexible expert selection
- **Literature**: Extension of MoE with configurable architecture
- **Main Features**: Provides a configurable mixture of experts framework that allows flexible expert selection and architecture customization for different molecular property prediction tasks.

### Crystalline and Materials Models

#### **Megnet (Materials Graph Network)**
- **Architecture**: Graph networks for materials and crystals
- **Literature**: Chen et al. (2019) - "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"
- **Main Features**: Implements graph networks specifically designed for materials and crystalline structures, enabling property prediction for both molecules and crystals.

#### **CGCNN (Crystal Graph Convolutional Neural Network)**
- **Architecture**: Crystal graph convolution for materials property prediction
- **Literature**: Xie & Grossman (2018) - "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
- **Main Features**: Uses crystal graph convolution to model periodic structures and predict material properties from crystal structures.

#### **HDNNP2nd (High-Dimensional Neural Network Potential 2nd Order)**
- **Architecture**: High-dimensional neural network potentials for molecular dynamics
- **Literature**: Behler (2011) - "Atom-centered symmetry functions for constructing high-dimensional neural network potentials"
- **Main Features**: Implements high-dimensional neural network potentials for accurate molecular dynamics simulation and energy prediction.

#### **HDNNP4th (High-Dimensional Neural Network Potential 4th Order)**
- **Architecture**: 4th order high-dimensional neural network potentials
- **Literature**: Extension of HDNNP2nd
- **Main Features**: Extends HDNNP2nd to 4th order interactions for more accurate molecular dynamics simulation and energy prediction.

### Additional Models

#### **RGCN (Relational Graph Convolutional Network)**
- **Architecture**: Relational graph convolution for heterogeneous graphs
- **Literature**: Schlichtkrull et al. (2018) - "Modeling Relational Data with Graph Convolutional Networks"
- **Main Features**: Extends graph convolution to handle heterogeneous graphs with different types of nodes and edges, enabling modeling of complex molecular relationships.

#### **GNNFilm (GNN-FiLM)**
- **Architecture**: Feature-wise linear modulation for graph neural networks
- **Literature**: Brockschmidt (2020) - "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
- **Main Features**: Implements feature-wise linear modulation for graph neural networks, providing better control over feature transformations during message passing.

#### **INorp (Interaction Networks)**
- **Architecture**: Interaction networks for learning about objects, relations and physics
- **Literature**: Battaglia et al. (2016) - "Interaction Networks for Learning about Objects, Relations and Physics"
- **Main Features**: Implements interaction networks for modeling complex physical interactions and relationships in molecular systems.

#### **MAT (Molecule Attention Transformer)**
- **Architecture**: Transformer-based molecular attention
- **Literature**: Maziarka et al. (2020) - "Molecule Attention Transformer"
- **Main Features**: Implements transformer-based attention mechanisms specifically designed for molecular property prediction, providing powerful sequence modeling capabilities.

#### **MXMNet (Molecular Mechanics-Driven Graph Neural Network)**
- **Architecture**: Molecular mechanics-driven graph neural network with multiplex graph
- **Literature**: Zhang et al. (2020) - "Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures"
- **Main Features**: Combines molecular mechanics principles with graph neural networks using multiplex graphs to model complex molecular interactions and conformations.

## Unified Descriptor Integration

All models have been enhanced with a unified descriptor handling system that supports:

- **Continuous Float Descriptors**: Integration of molecular descriptors as additional graph-level features
- **Unified Input System**: Standardized input processing using `kgcnn/utils/input_utils.py`
- **Descriptor Fusion**: Automatic fusion of descriptors with graph embeddings using concatenation
- **Flexible Processing**: Support for both scalar and vector descriptor inputs
- **Batch Normalization**: Enhanced numerical stability with GraphBatchNormalization

## Usage

```python
# Example usage with descriptors
from kgcnn.utils.input_utils import build_model_inputs
from kgcnn.literature.GraphSAGE import make_model

# Model configuration with descriptors
model_config = {
    "inputs": [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
    ],
    "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}},
    "output_mlp": {"units": [64, 32, 1], "activation": ["relu", "relu", "linear"]}
}

# Create model
model = make_model(**model_config)
```

## Author

**Guillaume Godin** - Extended KGCNN implementation with unified descriptor handling and 3D molecular modeling capabilities.

## References

This implementation builds upon the original KGCNN framework and extends it with:
- Unified descriptor integration system
- Enhanced 3D molecular modeling capabilities
- Improved numerical stability with batch normalization
- Comprehensive model coverage for molecular property prediction

For the original KGCNN implementation, see: [aimat-lab/gcnn_keras](https://github.com/aimat-lab/gcnn_keras)
