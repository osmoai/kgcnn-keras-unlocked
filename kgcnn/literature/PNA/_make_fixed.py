import tensorflow as tf
from tensorflow import keras as ks
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import OptionalInputEmbedding
from kgcnn.layers.pooling import PoolingEmbedding
from kgcnn.layers.mlp import MLP
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.literature.PNA._pna_conv_fixed import PNALayerFixed


def make_model_fixed(inputs, input_embedding=None, pna_args=None, depth=4, 
                    verbose=10, use_graph_state=False, output_embedding="graph",
                    output_to_tensor=True, output_mlp=None, name="PNAFixed"):
    """
    Make fixed PNA model.
    
    Args:
        inputs: Model inputs
        input_embedding: Input embedding configuration
        pna_args: PNA layer arguments
        depth: Number of PNA layers
        verbose: Verbosity level
        use_graph_state: Whether to use graph state
        output_embedding: Output embedding type
        output_to_tensor: Whether to convert output to tensor
        output_mlp: Output MLP configuration
        name: Model name
        
    Returns:
        PNA model
    """
    
    # Default arguments
    pna_args = pna_args or {}
    pna_args.setdefault('units', 128)
    pna_args.setdefault('aggregators', ['mean', 'max', 'min', 'std'])
    pna_args.setdefault('scalers', ['identity', 'amplification', 'attenuation'])
    pna_args.setdefault('use_bias', True)
    pna_args.setdefault('activation', 'relu')
    pna_args.setdefault('dropout_rate', 0.1)
    
    # Input layers
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])
    
    # Optional graph descriptors input
    graph_descriptors_input = None
    if len(inputs) > 2:
        graph_descriptors_input = ks.layers.Input(**inputs[2])
    
    # Node embedding
    n = OptionalInputEmbedding(**input_embedding.get("node", {}))([node_input, node_input])
    
    # Graph embedding (if provided)
    graph_embedding = None
    if graph_descriptors_input is not None and "graph" in input_embedding:
        graph_embedding = OptionalInputEmbedding(**input_embedding["graph"])([graph_descriptors_input, graph_descriptors_input])
    
    # PNA layers
    for i in range(depth):
        n = PNALayerFixed(**pna_args)([n, edge_index_input])
    
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingEmbedding(pooling_method="sum")(n)
        # Graph state fusion if provided
        if graph_descriptors_input is not None:
            if graph_embedding is not None:
                # Handle graph_embedding properly - it might be a list
                if isinstance(graph_embedding, list):
                    graph_embedding_tensor = graph_embedding[0]
                else:
                    graph_embedding_tensor = graph_embedding
                out = ks.layers.Concatenate()([out, graph_embedding_tensor])
            else:
                out = ks.layers.Concatenate()([out, graph_descriptors_input])
    elif output_embedding == "node":
        out = n
        # Graph state fusion if provided (for node output, broadcast descriptors)
        if graph_descriptors_input is not None:
            # Expand graph descriptors to match node dimension
            graph_descriptors_expanded = tf.expand_dims(graph_descriptors_input, axis=1)
            # Tile to match number of nodes in each graph
            num_nodes_per_graph = node_input.row_lengths()
            graph_descriptors_tiled = tf.ragged.map_flat_values(
                lambda x, n_nodes: tf.tile(x, [n_nodes, 1]),
                graph_descriptors_expanded, num_nodes_per_graph
            )
            out = ks.layers.Concatenate()([out, graph_descriptors_tiled])
    else:
        raise ValueError("Unsupported output embedding for mode %s" % output_embedding)
    
    # Output casting
    if output_to_tensor:
        if hasattr(out, 'to_tensor'):
            out = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)
    
    # Output MLP
    if output_mlp is not None:
        out = MLP(**output_mlp)(out)
    
    # Model creation
    model_inputs = [node_input, edge_index_input]
    if graph_descriptors_input is not None:
        model_inputs.append(graph_descriptors_input)
    
    model = ks.models.Model(inputs=model_inputs, outputs=out, name=name)
    
    return model 


def make_contrastive_pna_model(inputs: list = None,
                              input_embedding: dict = None,
                              depth: int = None,
                              pna_args: dict = None,
                              contrastive_args: dict = None,
                              name: str = None,
                              verbose: int = None,
                              use_graph_state: bool = False,
                              output_embedding: str = None,
                              output_to_tensor: bool = None,
                              output_mlp: dict = None
                              ):
    r"""Make Contrastive PNA model that uses our fixed PNA as base and adds contrastive learning losses.
    
    This is a simple implementation that:
    1. Uses our fixed PNA as the base model
    2. Adds contrastive learning losses on top
    3. Properly handles graph_descriptors input
    
    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`.
        input_embedding (dict): Dictionary of embedding arguments.
        depth (int): Number of PNA layers.
        pna_args (dict): Dictionary of PNA layer arguments.
        contrastive_args (dict): Dictionary of contrastive learning arguments.
        name (str): Name of the model.
        verbose (int): Level of print output.
        use_graph_state (bool): Whether to use graph descriptors.
        output_embedding (str): Output embedding type.
        output_to_tensor (bool): Whether to convert output to tensor.
        output_mlp (dict): Output MLP configuration.
        
    Returns:
        :obj:`tf.keras.models.Model`
    """
    # Default contrastive arguments
    if contrastive_args is None:
        contrastive_args = {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1
        }
    
    # Create the base PNA model using our fixed implementation
    base_model = make_model_fixed(
        inputs=inputs,
        input_embedding=input_embedding,
        depth=depth,
        pna_args=pna_args,
        name=name,
        verbose=verbose,
        use_graph_state=use_graph_state,
        output_embedding=output_embedding,
        output_to_tensor=output_to_tensor,
        output_mlp=output_mlp
    )
    
    # Add contrastive learning capabilities
    if contrastive_args.get("use_contrastive_loss", False):
        # Create a custom loss class that combines main task loss with contrastive loss
        class ContrastiveLoss(tf.keras.losses.Loss):
            def __init__(self, contrastive_args, **kwargs):
                super(ContrastiveLoss, self).__init__(**kwargs)
                self.contrastive_args = contrastive_args
                
            def call(self, y_true, y_pred, sample_weight=None):
                # Main task loss (binary crossentropy for classification)
                main_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                
                # Apply sample weights if provided
                if sample_weight is not None:
                    main_loss = main_loss * sample_weight
                
                # Simple contrastive loss based on predictions
                # For similar inputs (same class), predictions should be similar
                # For different inputs (different class), predictions should be different
                batch_size = tf.shape(y_pred)[0]
                
                # Create similarity matrix based on predictions
                pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
                similarity_matrix = tf.matmul(pred_norm, tf.transpose(pred_norm))
                
                # Create target similarity matrix based on true labels
                y_true_expanded = tf.expand_dims(y_true, 1)
                target_similarity = tf.cast(tf.equal(y_true_expanded, tf.transpose(y_true_expanded)), tf.float32)
                
                # Contrastive loss: maximize similarity for same class, minimize for different class
                temperature = self.contrastive_args.get("temperature", 0.1)
                contrastive_loss = tf.reduce_mean(
                    -target_similarity * tf.math.log(tf.nn.sigmoid(similarity_matrix / temperature) + 1e-8)
                )
                
                # Combine losses
                contrastive_weight = self.contrastive_args.get("contrastive_weight", 0.1)
                total_loss = main_loss + contrastive_weight * contrastive_loss
                
                return total_loss
        
        # Compile the model with the contrastive loss
        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=ContrastiveLoss(contrastive_args),
            metrics=['accuracy']
        )
    
    return base_model 