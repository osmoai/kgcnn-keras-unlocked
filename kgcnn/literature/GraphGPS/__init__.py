from ._make import make_graphgps_model
from ._graphgps_conv import GraphGPSConv

# Alias for compatibility with the pipeline
make_model = make_graphgps_model

__all__ = ["make_graphgps_model", "GraphGPSConv", "make_model"] 