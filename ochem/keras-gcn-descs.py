import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import time
import os
import os.path
import sys
import argparse
import json
from copy import deepcopy

# Add parent directory to Python path to ensure kgcnn modules can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kgcnn.training.schedule
import kgcnn.training.scheduler
import kgcnn.training.callbacks
from kgcnn.data.utils import save_json_file
# from kgcnn.utils.models  decrepeated starting 2.2.4
from kgcnn.model.utils import get_model_class

# Add RDKit import for SMILES validation
import rdkit.Chem as Chem

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.molecule.encoder import OneHotEncoder # remove starting 3.0.1
from kgcnn.training.hyper import HyperParameter # moved from hyper to training

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import configparser
import tarfile
from distutils.util import strtobool
# Standard imports
import pickle
# import tensorflow_addons as tfa
# from tensorflow_addons import optimizers


# Define the config
conf_name = sys.argv[1]
config = configparser.ConfigParser();
config.read(conf_name);

print("Load config file: ", conf_name);

def descriptor_callback (mg, ds):
    return np.array(ds[descs],dtype='float32')

def descriptor_callback_result0 (mg, ds):
    """Descriptor callback for Result0 only - uses desc0"""
    return np.array(ds['desc0'],dtype='float32')

def descriptor_callback_result1 (mg, ds):
    """Descriptor callback for Result1 only - uses desc1"""
    return np.array(ds['desc1'],dtype='float32')

def getConfig(section, attribute, default=""):
    try:
        return config[section][attribute];
    except:
        return default;


def SmilesOK(smi):
    try:
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        return True
    except:
        return False


# Categorical multiclass including Binary case "2"
def CCEmask(y_true, y_pred):
    y_true = tf.cast(y_true,
                     dtype=tf.float32)  # case of Classifiers # we may don't need it if we have the split between Reg/Class
    masked = tf.where(tf.math.is_nan(y_true), 0., 1.)
    return K.categorical_crossentropy(y_true * masked, y_pred * masked)


def BCEmask(y_true, y_pred):
    y_true = tf.cast(y_true,
                     dtype=tf.float32)  # case of Classifiers # we may don't need it if we have the split between Reg/Class
    masked = tf.where(tf.math.is_nan(y_true), 0., 1.)
    return K.binary_crossentropy(y_true * masked, y_pred * masked)


## multi sparse task
def RMSEmask(y_true, y_pred):
    # Compute the square error, and subsequently masked
    y_true = tf.cast(y_true, dtype=tf.float32)  # case of Classifiers
    masked = tf.where(tf.math.is_nan(y_true), 0., 1.)
    y_true_ = tf.where(tf.math.is_nan(y_true), 0., y_true)
    y_pred_ = tf.where(tf.math.is_nan(y_true), 0., y_pred)

    err = (y_true_ - y_pred_) * (y_true_ - y_pred_)

    # Select the qualifying values and mask
    sumsq = K.sum(masked * err, keepdims=True)
    num = tf.cast(K.sum(masked, keepdims=True), dtype=tf.float32)
    return K.sqrt(sumsq / (num + K.epsilon()))


class MaskedRMSE(tf.keras.metrics.Metric):
    def __init__(self, name='masked_rmse', **kwargs):
        super(MaskedRMSE, self).__init__(name=name, **kwargs)
        self.total_squared_error = self.add_weight(name='total_squared_error', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Simple masked RMSE metric for NaN handling
        y_true = tf.cast(y_true, dtype=tf.float32)
        masked = tf.where(tf.math.is_nan(y_true), 0., 1.)
        y_true_ = tf.where(tf.math.is_nan(y_true), 0., y_true)
        y_pred_ = tf.where(tf.math.is_nan(y_true), 0., y_pred)
        
        err = (y_true_ - y_pred_) * (y_true_ - y_pred_)
        sumsq = tf.reduce_sum(masked * err)
        num = tf.cast(tf.reduce_sum(masked), dtype=tf.float32)
        
        self.total_squared_error.assign_add(sumsq)
        self.total_count.assign_add(num)
    
    def result(self):
        return tf.sqrt(self.total_squared_error / (self.total_count + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.total_squared_error.assign(0.)
        self.total_count.assign(0.)


def isClassifer(graph_labels_all):
    t = np.unique(graph_labels_all)
    if len(t) == 2 and np.all(t == [0, 1]):
        return True
    else:
        return False

def splitingTrain_Val(dataset,labels,data_length, inputs=None, hypers=None, idx=0, cuts=10, output_dim=1, embsize=False, isCCE=False, seed=10):
    assert idx < cuts
    # split data using a specific seed!
    np.random.seed(seed)
    split_indices = np.uint8(np.floor(np.random.uniform(0,cuts,size=data_length)))
    print('data length: ',data_length)
    test = split_indices==idx
    train = ~test
    testidx = np.transpose(np.where(test==True))
    trainidx = np.transpose(np.where(train==True))
    # Debug: Print what inputs are being requested
    print(f"DEBUG: Requesting tensor with inputs: {[inp['name'] for inp in inputs]}")
    xtrain, ytrain = dataset[trainidx].tensor(inputs), labels[trainidx,:]
    xtest, ytest = dataset[testidx].tensor(inputs), labels[testidx,:]
    
    # Debug: Print what we actually got
    print(f"DEBUG: xtrain has {len(xtrain)} inputs")
    if hasattr(xtrain, '__iter__'):
        for i, x in enumerate(xtrain):
            print(f"DEBUG: xtrain[{i}] shape: {x.shape}, type: {type(x)}")
    
    ytrain = tf.reshape(ytrain, shape=(len(trainidx),labels.shape[1]))
    ytest = tf.reshape(ytest, shape=(len(testidx),labels.shape[1]))
    
    # DEBUG: Print label statistics
    print(f"DEBUG: ytrain shape: {ytrain.shape}")
    print(f"DEBUG: ytrain unique values: {tf.unique(tf.reshape(ytrain, [-1]))}")
    print(f"DEBUG: ytrain mean: {tf.reduce_mean(ytrain)}")
    print(f"DEBUG: ytrain std: {tf.math.reduce_std(ytrain)}")
    print(f"DEBUG: ytrain min: {tf.reduce_min(ytrain)}")
    print(f"DEBUG: ytrain max: {tf.reduce_max(ytrain)}")
    
    print(hypers)
    print(f"DEBUG: hypers type: {type(hypers)}")
    print(f"DEBUG: hypers has get_dict: {hasattr(hypers, 'get_dict')}")
    print(f"DEBUG: hypers has get: {hasattr(hypers, 'get')}")
    print(f"DEBUG: hypers dir: {[attr for attr in dir(hypers) if not attr.startswith('_')]}")

    # Handle both HyperParameter objects and regular dictionaries
    if hasattr(hypers, 'get') and callable(hypers.get):
        # It's a regular dictionary-like object
        hypers_dict = hypers
        print("DEBUG: Using hypers as dictionary")
    else:
        # It's a HyperParameter object, convert to dictionary
        try:
            # Try different methods to convert HyperParameter to dictionary
            if hasattr(hypers, 'get_dict'):
                hypers_dict = hypers.get_dict()
            elif hasattr(hypers, 'to_dict'):
                hypers_dict = hypers.to_dict()
            elif hasattr(hypers, '_hyper'):
                hypers_dict = hypers._hyper
            elif hasattr(hypers, '__dict__'):
                hypers_dict = hypers.__dict__
            else:
                hypers_dict = dict(hypers)
            print("⚠️  Converted HyperParameter to dictionary for processing")
            print(f"DEBUG: hypers_dict type: {type(hypers_dict)}")
            print(f"DEBUG: hypers_dict keys: {list(hypers_dict.keys())}")
        except Exception as e:
            print(f"⚠️  Error converting HyperParameter: {e}, using as-is")
            hypers_dict = hypers
    
    print(f"DEBUG: Final hypers_dict type: {type(hypers_dict)}")
    if hasattr(hypers_dict, 'get'):
        print(f"DEBUG: hypers_dict['model']['config'].keys(): {list(hypers_dict['model']['config'].keys())}")
    else:
        print(f"DEBUG: hypers_dict is not a dictionary-like object")
        # Try to access the HyperParameter object directly
        try:
            print(f"DEBUG: hypers['model']['config'].keys(): {list(hypers['model']['config'].keys())}")
        except Exception as e:
            print(f"DEBUG: Cannot access hypers['model']['config']: {e}")

    # CRITICAL: During inference, NEVER modify input dimensions
    if TRAIN == "False":
        print("🔒 Inference mode: Skipping input dimension updates to preserve saved model configuration")
    elif embsize:
        feature_atom_bond_mol = [xi.shape[2] for xi in xtest]
        hypers_dict['model']['config']['inputs'][0]['shape'][1] = feature_atom_bond_mol[0]
        hypers_dict['model']['config']['inputs'][1]['shape'][1] = feature_atom_bond_mol[1]
        hypers_dict['model']['config']['inputs'][2]['shape'][1] = feature_atom_bond_mol[2]
        hypers_dict['model']['config']['input_embedding']['node']['input_dim'] = feature_atom_bond_mol[0]
        hypers_dict['model']['config']['input_embedding']['edge']['input_dim'] = feature_atom_bond_mol[1]
    
    if 'last_mlp' in hypers_dict['model']['config'].keys():
        if type(hypers_dict['model']['config']['last_mlp']['units']) == list:
            hypers_dict['model']['config']['last_mlp']['units'][-1] = output_dim
        else:
            hypers_dict['model']['config']['last_mlp']['units'] = output_dim
            
    if 'output_mlp' in hypers['model']['config'].keys():
        if type(hypers_dict['model']['config']['output_mlp']['units']) == list:
            hypers_dict['model']['config']['output_mlp']['units'][-1] = output_dim
        else:
            hypers_dict['model']['config']['output_mlp']['units'] = output_dim


    isClass = isClassifer(ytrain)
    
    # SAFETY CHECK: Ensure output_mlp activation is properly initialized
    if 'output_mlp' in hypers_dict['model']["config"] and 'activation' not in hypers_dict['model']["config"]['output_mlp']:
        hypers_dict['model']["config"]['output_mlp']['activation'] = 'linear'
        print(f"⚠️  Fixed missing activation in output_mlp for {hypers_dict['model']['config']['name']}")
    
    # change activation function ! we do use sigmoid or softmax in general
    # CRITICAL RULE: For classification, sigmoid should be in the FINAL output layer
    if isClass:
        # Check if we have last_mlp (main output layer) or just output_mlp
        if 'last_mlp' in hypers_dict['model']["config"]:
            # For models with last_mlp, ensure last_mlp has linear activation
            # and output_mlp has sigmoid (final output)
            if isinstance(hypers_dict['model']["config"]['last_mlp']['activation'], list):
                hypers_dict['model']["config"]['last_mlp']['activation'][-1] = 'linear'
            else:
                hypers_dict['model']["config"]['last_mlp']['activation'] = 'linear'
            
            # Set output_mlp to sigmoid (final output layer)
            if isinstance(hypers_dict['model']["config"]['output_mlp']['activation'], list):
                hypers_dict['model']["config"]['output_mlp']['activation'][-1] = 'sigmoid'
            else:
                hypers_dict['model']["config"]['output_mlp']['activation'] = 'sigmoid'
            print(f"✅ Set last_mlp to linear and output_mlp to sigmoid for {hypers_dict['model']['config']['name']}")
        else:
            # Only output_mlp exists - modify it directly for classification
            if isCCE:
                # only for two target classes minimum
                if isinstance(hypers_dict['model']["config"]['output_mlp']['activation'], list):
                    hypers_dict['model']["config"]['output_mlp']['activation'][-1] = 'softmax'
                else:
                    hypers_dict['model']["config"]['output_mlp']['activation'] = 'softmax'
            else:
                # generally it works
                if isinstance(hypers_dict['model']["config"]['output_mlp']['activation'], list):
                    hypers_dict['model']["config"]['output_mlp']['activation'][-1] = 'sigmoid'
                else:
                    hypers_dict['model']["config"]['output_mlp']['activation'] = 'sigmoid'
    
    # FINAL SAFETY CHECK: Ensure proper sigmoid placement for classification
    if 'last_mlp' in hypers_dict['model']["config"] and 'output_mlp' in hypers_dict['model']["config"]:
        last_activation = hypers_dict['model']["config"]["last_mlp"].get('activation', 'linear')
        output_activation = hypers_dict['model']["config"]["output_mlp"].get('activation', 'linear')
        
        # Check if both have sigmoid (this should not happen)
        last_is_sigmoid = False
        output_is_sigmoid = False
        
        if isinstance(last_activation, list) and len(last_activation) > 0:
            last_is_sigmoid = last_activation[-1] in ['sigmoid', 'softmax']
        elif isinstance(last_activation, str):
            last_is_sigmoid = last_activation in ['sigmoid', 'softmax']
            
        if isinstance(output_activation, list) and len(output_activation) > 0:
            output_is_sigmoid = output_activation[-1] in ['sigmoid', 'softmax']
        elif isinstance(output_activation, str):
            output_is_sigmoid = output_activation in ['sigmoid', 'softmax']
        
        if last_is_sigmoid and output_is_sigmoid:
            print(f"🚨 WARNING: Double sigmoid detected in {hypers_dict['model']['config']['name']}!")
            print(f"🚨 This might cause issues. Consider using only one sigmoid at the final output.")
        elif not last_is_sigmoid and output_is_sigmoid:
            print(f"✅ Perfect: Sigmoid is correctly placed in final output layer for {hypers_dict['model']['config']['name']}")
        elif last_is_sigmoid and not output_is_sigmoid:
            print(f"⚠️  WARNING: Sigmoid is in last_mlp but not in output_mlp for {hypers_dict['model']['config']['name']}")
            print(f"⚠️  This might cause issues with classification output.")

    hyper = HyperParameter(hypers_dict,
                             model_name=hypers_dict['model']['config']['name'],
                             model_module=hypers_dict['model']['module_name'],
                             model_class=hypers_dict['model']['class_name'],
                             dataset_name="MoleculeNetDataset")
    
    #print('ici:',feature_atom_bond_mol,'| assigned inputs new dims:',hyper['model']["config"]['inputs'])
    
    dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

    return xtrain, ytrain, xtest, ytest, hyper, isClass

def prepData(name, labelcols, datasetname='Datamol', hyper=None, modelname=None, overwrite=False, descs=None):
    # Handle HyperParameter object access
    try:
        if hasattr(hyper, 'get') and callable(hyper.get):
            # It's a regular dictionary-like object
            hyper_dict = hyper
        else:
            # It's a HyperParameter object, convert to dictionary
            try:
                hyper_dict = hyper.get_dict() if hasattr(hyper, 'get_dict') else dict(hyper)
                print("⚠️  Converted HyperParameter to dictionary in prepData")
            except Exception as e:
                print(f"⚠️  Error converting HyperParameter in prepData: {e}, using as-is")
                hyper_dict = hyper
    except Exception as e:
        print(f"⚠️  Error handling hyper object in prepData: {e}, using as-is")
        hyper_dict = hyper
    
    is3D = False
    if modelname in ['PAiNN','DimeNetPP','HamNet','Schnet','HDNNP2nd','Megnet']:
        print('model need 3D molecules')
        if os.path.exists(name.replace('.csv','.sdf')) and not overwrite == 'True':
            print('use external 3D molecules')
            dataset = MoleculeNetDataset(file_name=name, data_directory="", dataset_name="MoleculeNetDataset")
            dataset.prepare_data(overwrite=overwrite)
            dataset.read_in_memory(label_column_name=labelcols)


            if modelname=='DimeNetPP':
                dataset.map_list(method="set_range", max_distance=4,  max_neighbours= 20)
                dataset.map_list(method="set_angle")
            elif modelname=='PAiNN':
                dataset.map_list(method="set_range", max_distance=3,  max_neighbours= 10000)
            elif modelname=='Schnet' :
                dataset.map_list(method="set_range", max_distance=4,  max_neighbours= 10000)
            elif modelname=='Megnet' :
                dataset.map_list(method="set_range", max_distance=4,  max_neighbours= 10000)
            elif modelname=='NMPN' :
                dataset.map_list(method="set_range", max_distance=3,  max_neighbours= 10000)
            elif modelname=='HDNNP2nd':
                dataset.map_list(method="set_range", max_distance=8,  max_neighbours= 10000)
                dataset.map_list(method="set_angle")


            #dataset.set_methods(hyper["data"]["dataset"]["methods"])
        else:
            is3D = True
            print('force internal 3D molecules')

            dataset = MoleculeNetDataset(file_name=name, data_directory="", dataset_name="MoleculeNetDataset")
            dataset.prepare_data(overwrite=overwrite, smiles_column_name="smiles",
                             make_conformers=is3D, add_hydrogen=is3D,
                             optimize_conformer=is3D, num_workers=10)

            dataset.read_in_memory(label_column_name=labelcols, add_hydrogen=False,
                               has_conformers=False)
            if len(descs)>0:
                print('Adding the graph_attributes from csv file!')
                # Use graph_descriptors for all models (standardized naming)
                callback_name = "graph_descriptors"
                print(f"Using callback name: {callback_name} for model: {modelname}")
                
                # Use only the appropriate callback for each model
                additional_callbacks = {callback_name: descriptor_callback}
                print(f"{modelname}: Using {callback_name} for descriptors")
                
                dataset.set_attributes(add_hydrogen=False,has_conformers=is3D, additional_callbacks=additional_callbacks)
    
            else:
                dataset.set_attributes(add_hydrogen=False,has_conformers=is3D)
        
            dataset.set_methods(hyper_dict["data"]["dataset"]["methods"])


    else:
        print('no 3D model')    
        dataset = MoleculeNetDataset(file_name=name, data_directory="", dataset_name="MoleculeNetDataset")
        
        dataset.prepare_data(overwrite=is3D, smiles_column_name="smiles",
                             make_conformers=is3D, add_hydrogen=is3D,
                             optimize_conformer=is3D, num_workers=10)

        dataset.read_in_memory(label_column_name=labelcols, add_hydrogen=False,
                               has_conformers=False)
                               
        if len(descs)>0:
                print('Adding the graph_attributes from csv file!')
                # Use graph_descriptors for all models (standardized naming)
                callback_name = "graph_descriptors"
                print(f"Using callback name: {callback_name} for model: {modelname}")
                
                # Use only the appropriate callback for each model
                additional_callbacks = {callback_name: descriptor_callback}
                print(f"{modelname}: Using {callback_name} for descriptors")
                
                dataset.set_attributes(add_hydrogen=False,has_conformers=is3D, additional_callbacks=additional_callbacks)
    
        else:
                dataset.set_attributes(add_hydrogen=False,has_conformers=is3D)
        
        dataset.set_methods(hyper_dict["data"]["dataset"]["methods"])

    # Generate edge_indices_reverse for directed models
    if modelname in ['MultiChem', 'DMPNN', 'DMPNNAttention', 'DGIN', 'EGAT', 'TransformerGAT', 'DHTNN', 'DHTNNPlus']:
        print(f"Generating edge_indices_reverse for {modelname} model...")
        import numpy as np
        for graph in dataset:
            if 'edge_indices' in graph and 'edge_indices_reverse' not in graph:
                # Create reverse edge indices
                edge_indices = np.array(graph['edge_indices'])
                edge_indices_reverse = []
                for i, edge in enumerate(edge_indices):
                    # Find the reverse edge
                    reverse_edge = np.array([edge[1], edge[0]])
                    # Find matching reverse edges
                    matches = np.where((edge_indices == reverse_edge).all(axis=1))[0]
                    if len(matches) > 0:
                        reverse_idx = matches[0]
                        edge_indices_reverse.append(reverse_idx)
                    else:
                        # If reverse edge doesn't exist, use the same edge
                        edge_indices_reverse.append(i)
                graph['edge_indices_reverse'] = np.array(edge_indices_reverse).reshape(-1, 1)

    invalid = dataset.clean(hyper_dict["model"]["config"]["inputs"])
    # I guess clean first and assert clean ok

    return dataset, invalid


target_list = ['target']
# Define standard parameters from config.cfg
TRAIN = getConfig("Task", "train_mode");
MODEL_FILE = getConfig("Task", "model_file");
TRAIN_FILE = getConfig("Task", "train_data_file");
APPLY_FILE = getConfig("Task", "apply_data_file", "train.csv");
RESULT_FILE = getConfig("Task", "result_file", "results.csv");

# Define details options from config.cfg
nbepochs = int(getConfig("Details", "nbepochs", 100));
seed = int(getConfig("Details", "seed", 101));
batch_size = int(getConfig("Details", "batch", 128));
earlystop_patience = int(getConfig("Details", "earlystop_patience", 30));
redlr_patience = int(getConfig("Details", "redlr_patience", 10));
redlr_factor = float(getConfig("Details", "redlr_factor", 0.8));
val_split = float(getConfig("Details", "reduce_lr_factor", 0.2));
cfg_learning_rate = float(getConfig("Details", "lr", 0.001));
gpu = int(getConfig("Details", "gpu", 0));

# Define network parameters
nn_embedding = int(getConfig("Details", "embedding_size", 32));
nn_lstmunits = int(getConfig("Details", "lstm_units", 16));
nn_denseunits = int(getConfig("Details", "dense_units", 16));
nn_return_proba = int(getConfig("Details", "return_proba", 0));
nn_loss = str(getConfig("Details", "nn_loss", "RMSE"));
nn_lr_start = float(getConfig("Details", "nn_lr_start", 1e-2));
output_dim = int(getConfig("Details", "output_dim", 1));
print(f"Read output_dim from config: {output_dim}")
activation = str(getConfig("Details", "activation", "linear"));
print(f"Read activation from config: {activation}")
lossdef = str(getConfig("Details", "lossdef", "RMSEmask"));
print(f"Read lossdef from config: {lossdef}")

# Map loss function names to actual functions
loss_function_map = {
    "BCEmask": BCEmask,
    "CCEmask": CCEmask,
    "RMSEmask": RMSEmask,
    "MaskedRMSE": MaskedRMSE,
    "mean_squared_error": "mean_squared_error",
    "mean_absolute_error": "mean_absolute_error",
    "binary_crossentropy": "binary_crossentropy",
    "categorical_crossentropy": "categorical_crossentropy"
}

# Get the actual loss function
if lossdef in loss_function_map:
    loss_function = loss_function_map[lossdef]
    print(f"Using loss function: {lossdef}")
else:
    loss_function = lossdef  # Use as string if not in map
    print(f"Using loss function as string: {lossdef}")

desc_dim = int(getConfig("Details", "desc_dim", 0));

# Descriptor parameters
use_descriptors = strtobool(getConfig("Details", "use_descriptors", "False"));
descriptor_columns = getConfig("Details", "descriptor_columns", "");
descs = None
if use_descriptors and descriptor_columns:
    descs = [col.strip() for col in descriptor_columns.split(',')]
    print(f"Using descriptor columns: {descs}")
    desc_dim = len(descs)

# model selection parameters for selecting which architecture to run
architecture_name = getConfig("Details", "architecture_name", 'AttFP');
overwrite = getConfig("Details", "overwrite", 'False');

print("Architecture selected:", architecture_name)

# Function to update output dimensions in model configuration
def update_output_dimensions(config_dict, architecture_name=None):
    """Update output dimensions and activation in model configuration based on config file"""
    # Check if we're in inference mode (train_mode = False)
    if TRAIN == "False":
        print("🔒 Inference mode: Skipping update_output_dimensions to preserve original model configuration")
        return config_dict
    
    print(f"Updating output dimensions for output_dim = {output_dim}")
    print(f"Updating activation to: {activation}")
    
    # If architecture_name is provided, try to read from config file first
    if architecture_name and architecture_name in config:
        arch_config = config[architecture_name]
        print(f"Reading configuration for {architecture_name} from config file")
        
        # Parse output_mlp from config file
        if 'output_mlp' in arch_config:
            try:
                # Parse the output_mlp string from config
                output_mlp_str = arch_config['output_mlp']
                # This is a simplified parser - in production you'd want to use ast.literal_eval
                # For now, we'll extract the units part
                if '"units": [' in output_mlp_str:
                    units_start = output_mlp_str.find('"units": [') + 9
                    units_end = output_mlp_str.find(']', units_start)
                    units_str = output_mlp_str[units_start:units_end]
                    units = [int(x.strip()) for x in units_str.split(',')]
                    # Update the last unit to match output_dim
                    units[-1] = output_dim
                    config_dict["output_mlp"] = {
                        "use_bias": [True, True, False],
                        "units": units,
                        "activation": ["kgcnn>leaky_relu", "selu", "linear"]
                    }
                    print(f"Updated output_mlp from config: {config_dict['output_mlp']}")
                    return config_dict
            except Exception as e:
                print(f"Failed to parse output_mlp from config: {e}")
    
    # NEW RULE: For binary classification, sigmoid MUST be in the FINAL output layer
    # This applies to ALL models consistently
    if "last_mlp" in config_dict:
        print(f"Before update - last_mlp: {config_dict['last_mlp']}")
        if isinstance(config_dict["last_mlp"].get('units', []), list) and len(config_dict["last_mlp"]['units']) > 0:
            config_dict["last_mlp"]['units'][-1] = output_dim
            print(f"Updated last_mlp units list: {config_dict['last_mlp']['units']}")
        elif isinstance(config_dict["last_mlp"].get('units', 1), int):
            config_dict["last_mlp"]['units'] = output_dim
            print(f"Updated last_mlp units: {config_dict['last_mlp']['units']}")
        
        # NEW RULE: last_mlp should have linear activation (not sigmoid)
        if isinstance(config_dict["last_mlp"].get('activation', []), list) and len(config_dict["last_mlp"]['activation']) > 0:
            config_dict["last_mlp"]['activation'][-1] = 'linear'
            print(f"✅ Set last_mlp activation to linear (sigmoid goes in final output): {config_dict['last_mlp']['activation']}")
        elif isinstance(config_dict["last_mlp"].get('activation', 'linear'), str):
            config_dict["last_mlp"]['activation'] = 'linear'
            print(f"✅ Set last_mlp activation to linear (sigmoid goes in final output): {config_dict['last_mlp']['activation']}")
        print(f"After update - last_mlp: {config_dict['last_mlp']}")
    
    if "output_mlp" in config_dict:
        print(f"Before update - output_mlp: {config_dict['output_mlp']}")
        if isinstance(config_dict["output_mlp"].get('units', []), list) and len(config_dict["output_mlp"]['units']) > 0:
            config_dict["output_mlp"]['units'][-1] = output_dim
            print(f"Updated output_mlp units list: {config_dict['output_mlp']['units']}")
        elif isinstance(config_dict["output_mlp"].get('units', 1), int):
            config_dict["output_mlp"]['units'] = output_dim
            print(f"Updated output_mlp units: {config_dict['output_mlp']['units']}")
        
        # NEW RULE: output_mlp should have sigmoid activation (final output layer)
        # This is the FINAL layer that produces classification probabilities
        
        # SAFETY CHECK: Ensure activation is properly formatted
        if 'activation' not in config_dict["output_mlp"]:
            if isinstance(config_dict["output_mlp"].get('units', []), list):
                units_list = config_dict["output_mlp"]['units']
                config_dict["output_mlp"]['activation'] = ['linear'] * len(units_list)
            else:
                config_dict["output_mlp"]['activation'] = 'linear'
            print(f"⚠️  Added missing activation in update_output_dimensions: {config_dict['output_mlp']['activation']}")
        elif isinstance(config_dict["output_mlp"].get('units', []), list) and isinstance(config_dict["output_mlp"]['activation'], str):
            # Convert string activation to list if units is a list
            units_list = config_dict["output_mlp"]['units']
            config_dict["output_mlp"]['activation'] = [config_dict["output_mlp"]['activation']] * len(units_list)
            print(f"⚠️  Converted activation to list in update_output_dimensions: {config_dict['output_mlp']['activation']}")
        
        # NEW RULE: Set the final activation to sigmoid (for binary classification)
        if isinstance(config_dict["output_mlp"].get('activation', []), list) and len(config_dict["output_mlp"]['activation']) > 0:
            config_dict["output_mlp"]['activation'][-1] = activation  # This should be 'sigmoid' for binary classification
            print(f"✅ Set output_mlp final activation to {activation} (final output layer): {config_dict['output_mlp']['activation']}")
        elif isinstance(config_dict["output_mlp"].get('activation', 'linear'), str):
            config_dict["output_mlp"]['activation'] = activation  # This should be 'sigmoid' for binary classification
            print(f"✅ Set output_mlp activation to {activation} (final output layer): {config_dict['output_mlp']['activation']}")
        print(f"After update - output_mlp: {config_dict['output_mlp']}")
    
    # FINAL SAFETY CHECK: Ensure proper sigmoid placement for classification
    if "last_mlp" in config_dict and "output_mlp" in config_dict:
        last_activation = config_dict["last_mlp"].get('activation', 'linear')
        output_activation = config_dict["output_mlp"].get('activation', 'linear')
        
        # Check activation placement
        last_is_sigmoid = False
        output_is_sigmoid = False
        
        if isinstance(last_activation, list) and len(last_activation) > 0:
            last_is_sigmoid = last_activation[-1] in ['sigmoid', 'softmax']
        elif isinstance(last_activation, str):
            last_is_sigmoid = last_activation in ['sigmoid', 'softmax']
            
        if isinstance(output_activation, list) and len(output_activation) > 0:
            output_is_sigmoid = output_activation[-1] in ['sigmoid', 'softmax']
        elif isinstance(output_activation, str):
            output_is_sigmoid = output_activation in ['sigmoid', 'softmax']
        
        if last_is_sigmoid and output_is_sigmoid:
            print("🚨 WARNING: Double sigmoid detected! This might cause issues.")
            print("🚨 Consider using only one sigmoid at the final output layer.")
        elif not last_is_sigmoid and output_is_sigmoid:
            print("✅ Perfect: Sigmoid is correctly placed in final output layer")
        elif last_is_sigmoid and not output_is_sigmoid:
            print("⚠️  WARNING: Sigmoid is in last_mlp but not in output_mlp")
            print("⚠️  This might cause issues with classification output.")
            print("⚠️  Moving sigmoid to output_mlp (final layer)...")
            # Fix: Move sigmoid to output_mlp
            if isinstance(config_dict["last_mlp"]['activation'], list):
                config_dict["last_mlp"]['activation'][-1] = 'linear'
            else:
                config_dict["last_mlp"]['activation'] = 'linear'
            if isinstance(config_dict["output_mlp"]['activation'], list):
                config_dict["output_mlp"]['activation'][-1] = activation
            else:
                config_dict["output_mlp"]['activation'] = activation
            print("✅ Fixed: Moved sigmoid to final output layer")
    
    return config_dict


if architecture_name == 'GCN':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GCN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
        "input_embedding": {"node": {"input_dim": 95, "output_dim": 100},
                            "edge": {"input_dim": 10, "output_dim": 100},
                            "graph": {"input_dim": 100, "output_dim": 64}},
        "gcn_args": {"units": 200, "use_bias": True, "activation": "relu"},
        "depth": 5, "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                       "activation": ["kgcnn>leaky_relu", "kgcnn>leaky_relu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GCN",
            "config": model_config
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 800,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 250, "epo": 800,
                        "verbose": 0
                    }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        hyper["model"]["config"]["inputs"].append(
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        )
        hyper["model"]["config"]["use_graph_state"] = True
        print(f"Added descriptor input with dimension {desc_dim} to GCN")

elif architecture_name == 'GAT':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GAT",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 100},
            "edge": {"input_dim": 8, "output_dim": 100},
            "graph": {"input_dim": 100, "output_dim": 64}},
        "attention_args": {"units": 100, "use_bias": True, "use_edge_features": True,
                           "use_final_activation": False, "has_self_loops": True},
        "pooling_nodes_args": {"pooling_method": "sum"},
        "depth": 4, "attention_heads_num": 10,
        "attention_heads_concat": False, "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                       "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GAT",
            "config": model_config
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 2, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 1, "epo": 2,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error",
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        hyper["model"]["config"]["inputs"].append(
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        )
        hyper["model"]["config"]["use_graph_state"] = True
        print(f"Added descriptor input with dimension {desc_dim} to GAT")

elif architecture_name == 'GATv2':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GATv2",
            "config": {
                "name": "GATv2",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {
                    "node": {"input_dim": 95, "output_dim": 100},
                    "edge": {"input_dim": 8, "output_dim": 100},
                    "graph": {"input_dim": 100, "output_dim": 64}},
                "attention_args": {"units": 100, "use_bias": True, "use_edge_features": True,
                                   "use_final_activation": False, "has_self_loops": True},
                "pooling_nodes_args": {"pooling_method": "sum"},
                "depth": 4, "attention_heads_num": 10,
                "attention_heads_concat": False, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 500, "validation_freq": 2, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 0.5e-03, "learning_rate_stop": 1e-05, "epo_min": 250, "epo": 500,
                        "verbose": 0}
                     }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": []
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        hyper["model"]["config"]["inputs"].append(
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        )
        hyper["model"]["config"]["use_graph_state"] = True
        print(f"Added descriptor input with dimension {desc_dim} to GATv2")

elif architecture_name == 'CMPNN':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CMPNN",
            "config": {
                "name": "CMPNN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64}},
                "node_initialize": {"units": 200, "activation": "relu"},
                "edge_initialize": {"units": 200, "activation": "relu"},
                "edge_dense": {"units": 200, "activation": "linear"},
                "node_dense": {"units": 200, "activation": "linear"},
                "edge_activation": {"activation": "relu"},
                "verbose": 10,
                "depth": 3,
                "dropout": None,
                "use_final_gru": False,
                "pooling_gru": {"units": 200},
                "pooling_kwargs": {"pooling_method": "sum"},
                "output_embedding": "graph",
                "output_mlp": {
                    "use_bias": [True, True, False], "units": [200, 100, output_dim],
                    "activation": ["kgcnn>leaky_relu", "selu", "linear"]
                }
            }
        },
        "training": {
            "fit": {"batch_size": 50, "epochs": 600, "validation_freq": 1, "verbose": 2, "callbacks": []},
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}}
                              }},
                "loss": "mean_squared_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.4"
        }
    }






# Accept both 'AttFP' and 'AttentiveFP' for AttentiveFP model
elif architecture_name in ['AttFP', 'AttentiveFP']:
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFP",
            "config": {
                "name": "AttentiveFP",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64","ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 100},
                                    "edge": {"input_dim": 5, "output_dim": 100},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "attention_args": {"units": 200},
                "depthato": 2, 
                "depthmol": 2,
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 100, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.0031622776601683794,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to AttentiveFP")
        hyper["model"]["config"]["use_graph_state"] = True
    
    # Handle descriptors for AttentiveFP - only add if not already added
    if len(descs)>0:
        print('There are Additional Descriptors/Conditions')
        if 'use_graph_state' in hyper["model"]["config"].keys():
            print(hyper["model"]["config"]['use_graph_state'])
            hyper["model"]["config"]['use_graph_state']=True
            # Check if graph_descriptors is already in inputs to avoid duplicates
            input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
            if 'graph_descriptors' not in input_names:
                hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}
        else:
            print('Model does not support use_graph_state, adding descriptor input manually')
            # Add descriptor input manually for models that don't have use_graph_state
            input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
            if 'graph_descriptors' not in input_names:
                hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            # Add graph embedding to input_embedding if it doesn't exist
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}

# CoAttentiveFP (Collaborative Attentive Fingerprint) - Enhanced AttFP with collaborative attention
elif architecture_name == 'CoAttentiveFP':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CoAttentiveFP",
            "config": {
                "name": "CoAttentiveFP",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64","ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},  # Enhanced embedding
                                    "edge": {"input_dim": 5, "output_dim": 128},   # Enhanced embedding
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "attention_args": {"units": 256, "use_collaborative": True, "collaboration_heads": 8},  # Collaborative attention
                "depthato": 3,  # Increased depth for better collaboration
                "depthmol": 3,  # Increased depth for better collaboration
                "dropout": 0.15,  # Reduced dropout for better collaboration
                "verbose": 10,
                "output_embedding": "graph",
                "use_graph_state": False,  # Default to False, will be set to True if descriptors are used
                "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 150, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 50, "epo": 150,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to CoAttentiveFP")
        hyper["model"]["config"]["use_graph_state"] = True

# AttentiveFP+ (Multi-scale Attention Fusion) - Enhanced AttFP with multi-scale attention
elif architecture_name == 'AttentiveFPPlus':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AttentiveFPPlus",
            "config": {
                "name": "AttentiveFPPlus",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32","ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64","ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "attention_args": {"units": 256, "use_multiscale": True, "scale_fusion": "weighted_sum", "attention_scales": [1, 2, 4]},  # Multi-scale attention
                "depthato": 4,  # Increased depth for multi-scale
                "depthmol": 4,  # Increased depth for multi-scale
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to AttentiveFPPlus")
        hyper["model"]["config"]["use_graph_state"] = True

# CMPNN+ (Communicative Message Passing Neural Network) - Enhanced CMPNN with multi-level communicative message passing
elif architecture_name == 'CMPNNPlus':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.CMPNNPlus",
            "config": {
                "name": "CMPNNPlus",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "pooling_args": {"pooling_method": "sum"},
                "edge_initialize": {"units": 256, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 256, "use_bias": True, "activation": "relu"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 256, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depth": 6, "dropout": {"rate": 0.15},
                "use_communicative": True, "communication_levels": 3,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, output_dim],
                               "activation": ["relu", "relu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to CMPNNPlus")
        hyper["model"]["config"]["use_graph_state"] = True

# DMPNN with Attention Readout - Enhanced DMPNN with attention-based pooling
elif architecture_name == 'DMPNNAttention':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNNAttention",
            "config": {
                "name": "DMPNNAttention",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "pooling_args": {"pooling_method": "attention", "attention_units": 128, "attention_heads": 8},
                "edge_initialize": {"units": 200, "use_bias": True, "activation": "relu"},
                "edge_dense": {"units": 200, "use_bias": True, "activation": "relu"},
                "edge_activation": {"activation": "relu"},
                "node_dense": {"units": 41, "use_bias": True, "activation": "relu"},
                "verbose": 10, "depth": 5, "dropout": {"rate": 0.1},
                "attention_units": 128, "attention_heads": 8,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["relu", "relu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 2, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 1, "epo": 2,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              },
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to DMPNNAttention")
        hyper["model"]["config"]["use_graph_state"] = True

# checked
elif architecture_name in ['NMPN','MPNN']:
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.NMPN",
            "config": {
                'name': "NMPN",
                'inputs': [{'shape': (None, 41), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, ), 'name': "edge_number", 'dtype': 'float32', 'ragged': True},
                           {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True},
                           {'shape': [desc_dim], 'name': "graph_descriptors", 'dtype': 'float32', 'ragged': False}],
                'input_embedding': {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 95, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                # 'gauss_args': {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},  # Not needed for PNA
                'set2set_args': {'channels': 64, 'T': 3, "pooling_method": "sum", "init_qstar": "0"},
                'pooling_args': {'pooling_method': "segment_sum"},
                'edge_mlp': {'use_bias': True, 'activation': 'swish', "units": [64, 64]},
                'use_set2set': True, 'depth': 3, 'node_dim': 128,
                "geometric_edge": False, "make_distance": False, "expand_distance": False,
                'verbose': 10,
                'output_embedding': 'graph',
                'output_mlp': {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                                                   "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 100,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 100, "epo": 100,
                        "verbose": 0
                    }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "BCEmask"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    }



# Enhanced DGIN (Directed GIN) with descriptor support
elif architecture_name == 'DGIN':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "DGIN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 100},
            "edge": {"input_dim": 5, "output_dim": 100},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "gin_mlp": {"units": [100, 100], "use_bias": True, "activation": ["relu", "relu"],
                    "use_normalization": True, "normalization_technique": "graph_batch"},
        "gin_args": {},
        "pooling_args": {"pooling_method": "sum"},
        "use_graph_state": True,  # Enable graph state for descriptors
        "edge_initialize": {"units": 100, "use_bias": True, "activation": "relu"},
        "edge_dense": {"units": 100, "use_bias": True, "activation": "linear"},
        "edge_activation": {"activation": "relu"},
        "node_dense": {"units": 100, "use_bias": True, "activation": "relu"},
        "verbose": 10,
        "depthDMPNN": 4,  # DMPNN depth as per paper
        "depthGIN": 4,    # GIN depth as per paper
        "dropoutDMPNN": {"rate": 0.15},  # Paper default
        "dropoutGIN": {"rate": 0.15},    # Paper default
        "output_embedding": "graph",
        "output_to_tensor": True,
        "last_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "output_mlp": {"use_bias": True, "units": output_dim, "activation": "linear"}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DGIN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# AddGNN (Additive Attention GNN) with descriptor support
elif architecture_name == 'AddGNN':
    print(f"Checking architecture: {architecture_name}")
    print("Found AddGNN architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "AddGNN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "addgnn_args": {"units": 200, "heads": 4, "activation": "relu", "use_bias": True},
        "depth": 3,
        "node_dim": 200,
        "use_set2set": True,
        "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AddGNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# AddGNN-PNA (Additive Attention GNN with PNA aggregation) with descriptor support
elif architecture_name == 'AddGNN-PNA':
    print(f"Checking architecture: {architecture_name}")
    print("Found AddGNN-PNA architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "AddGNN-PNA",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "addgnn_pna_args": {
            "units": 200, 
            "activation": "relu", 
            "use_bias": True,
            "aggregators": ["mean", "max", "min"],
            "scalers": ["identity", "amplification", "attenuation"],
            "delta": 1.0,
            "dropout_rate": 0.1,
            "use_edge_features": True,
            "use_skip_connection": True
        },
        "depth": 3,
        "node_dim": 200,
        "use_set2set": True,
        "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_addgnn_pna_model",
            "module_name": "kgcnn.literature.AddGNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# CMPNN-PNA (Communication-Efficient Graph Neural Networks with PNA aggregation) with descriptor support
elif architecture_name == 'CMPNN-PNA':
    print(f"Checking architecture: {architecture_name}")
    print("Found CMPNN-PNA architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "CMPNN-PNA",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "cmpnn_pna_args": {
            "units": 200, 
            "use_bias": True, 
            "activation": "relu",
            "aggregators": ["mean", "max", "min", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
            "delta": 1.0,
            "dropout_rate": 0.1
        },
        "depth": 3,
        "node_dim": 200,
        "use_set2set": True,
        "set2set_args": {"channels": 200, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_cmpnn_pna_model",
            "module_name": "kgcnn.literature.CMPNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# RGCN-PNA (Relational Graph Convolutional Network with PNA aggregation) with descriptor support
elif architecture_name == 'RGCN-PNA':
    print(f"Checking architecture: {architecture_name}")
    print("Found RGCN-PNA architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "RGCN-PNA",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "rgcn_pna_args": {
            "units": 200, 
            "use_bias": True, 
            "activation": "relu",
            "aggregators": ["mean", "max", "min", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
            "delta": 1.0,
            "dropout_rate": 0.1
        },
        "depth": 3,
        "node_dim": 200,
        "use_set2set": True,
        "set2set_args": {"channels": 200, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_rgcn_pna_model",
            "module_name": "kgcnn.literature.RGCN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# GraphTransformer (Graph Transformer Neural Network) with descriptor support
elif architecture_name == 'GraphTransformer':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GraphTransformer",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "transformer_args": {
            "units": 200,
            "num_heads": 8,
            "ff_units": 800,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_edge_features": True,
            "use_positional_encoding": True,
            "positional_encoding_dim": 64,
            "activation": "relu",
            "layer_norm_epsilon": 1e-6
        },
        "depth": 4,
        "use_set2set": True,
        "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Fix Set2Set channels to match transformer units
    model_config["set2set_args"]["channels"] = model_config["transformer_args"]["units"]
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphTransformer",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error",
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# checked
elif architecture_name in ['ChemProp', 'DMPNN']:
    print(f"Executing DMPNN/ChemProp configuration block for architecture: {architecture_name}")
    # Base inputs for DMPNN
    dmpnn_inputs = [
        {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
        {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True}
    ]
    
    # Add descriptor input if enabled - use graph_descriptors name
    if descs:
        # Use graph_descriptors as the standard name
        dmpnn_inputs.append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
        print(f"Added descriptor input with dimension {desc_dim} to DMPNN (as graph_descriptors)")
        print(f"DMPNN inputs: {len(dmpnn_inputs)} inputs configured")
    else:
        print(f"DMPNN inputs: {len(dmpnn_inputs)} inputs configured (no descriptors)")
    
    model_config = {
        "name": "DMPNN",
        "inputs": dmpnn_inputs,
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 100},
            "edge": {"input_dim": 5, "output_dim": 100},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        # Add descriptor embedding if descriptors are used
        "use_graph_state": descs,  # Enable graph state when using descriptors
        "pooling_args": {"pooling_method": "sum"},
        "edge_initialize": {"units": 200, "use_bias": True, "activation": "relu"},
        "edge_dense": {"units": 200, "use_bias": True, "activation": "linear"},
        "edge_activation": {"activation": "relu"},
        "node_dense": {"units": 200, "use_bias": True, "activation": "relu"},
        "verbose": 10, "depth": 5,
        "dropout": {"rate": 0.2},
        "output_embedding": "graph",
        "output_mlp": {
            "use_bias": [True, True, False], "units": [200, 100, output_dim],
            "activation": ["kgcnn>leaky_relu", "selu", "linear"]
        }
    }
    
    # Update output dimensions and activation
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"DMPNN model config - use_graph_state: {model_config.get('use_graph_state', 'Not set')}")
    print(f"DMPNN model config - inputs count: {len(model_config['inputs'])}")
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DMPNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 300, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.0001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                              },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }


# checked
elif architecture_name == 'RGCN':
    hyper =  {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.RGCN",
            "config": {
                "name": "RGCN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
                    {"shape": [None], "name": "edge_number", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 64},
                                    "edge": {"input_dim": 5, "output_dim": 64},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "dense_relation_kwargs": {"units": 64, "num_relations": 20},
                "dense_kwargs": {"units": 64},
                "activation_kwargs": {"activation": "swish"},
                "depth": 5, "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
            }
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 800,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                        "learning_rate_start": 1e-03, "learning_rate_stop": 5e-05, "epo_min": 250, "epo": 800,
                        "verbose": 0}
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 1e-03}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": 42, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "normalize_edge_weights_sym"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.2.0"
        }
    }


elif architecture_name == 'rGIN':
    print(f"Executing rGIN configuration block")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "rGIN",
        "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                   {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
        "input_embedding": {"node": {"input_dim": 96, "output_dim": 100},
                            "edge": {"input_dim": 5, "output_dim": 100},
                            "graph": {"input_dim": 100, "output_dim": 64}},
        "depth": 4,
        "dropout": 0.1,
        "gin_mlp": {"units": [100, 100], "use_bias": True, "activation": ["relu", "relu"],
                    "use_normalization": True, "normalization_technique": "graph_batch"},
        "rgin_args": {"random_range": 100},
        "last_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "output_embedding": "graph", "output_to_tensor": True,
        "output_mlp": {"use_bias": True, "units": output_dim, "activation": "linear"}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.rGIN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

elif architecture_name == 'rGINE':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "rGINE",
        "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                   {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
        "input_embedding": {"node": {"input_dim": 96, "output_dim": 100},
                            "edge": {"input_dim": 5, "output_dim": 100},
                            "graph": {"input_dim": 100, "output_dim": 64}},
        "depth": 4,
        "dropout": 0.1,
        "gin_mlp": {"units": [100, 100], "use_bias": True, "activation": ["relu", "relu"],
                    "use_normalization": True, "normalization_technique": "graph_batch"},
        "rgine_args": {"random_range": 100},
        "last_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "output_embedding": "graph", "output_to_tensor": True,
        "output_mlp": {"use_bias": True, "units": output_dim, "activation": "linear"}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model_edge",
            "module_name": "kgcnn.literature.rGIN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": "BCEmask"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }



elif architecture_name == 'GIN':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GIN",
        "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                   {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
        "input_embedding": {"node": {"input_dim": 96, "output_dim": 100},
                            "edge": {"input_dim": 5, "output_dim": 100},
                            "graph": {"input_dim": 100, "output_dim": 64}},
        "depth": 4,
        "dropout": 0.1,
        "gin_mlp": {"units": [100, 100], "use_bias": True, "activation": ["relu", "relu"],
                    "use_normalization": True, "normalization_technique": "graph_batch"},
        "gin_args": {},
        "last_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "output_embedding": "graph", "output_to_tensor": True,
        "output_mlp": {"use_bias": True, "units": output_dim, "activation": "linear"}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GIN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {"class_name": "MoleculeNetDataset",
                        "config": {},
                        "methods": []
                        },
            "data_unit": "unit"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }

elif architecture_name == 'GINE':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GIN",
        "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                   {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                   {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}],
        "input_embedding": {"node": {"input_dim": 96, "output_dim": 100},
                            "edge": {"input_dim": 5, "output_dim": 100},
                            "graph": {"input_dim": 100, "output_dim": 64}},
        "depth": 4,
        "dropout": 0.1,
        "gin_mlp": {"units": [100, 100], "use_bias": True, "activation": ["relu", "relu"],
                    "use_normalization": True, "normalization_technique": "graph_batch"},
        "gin_args": {},
        "last_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "output_embedding": "graph",
        "output_mlp": {"use_bias": True, "units": output_dim, "activation": "linear"}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model_edge",
            "module_name": "kgcnn.literature.GIN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                              },
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": []
            },
            "data_unit": "unit"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }

# GraphGPS (Graph Property Prediction with Subgraph) with descriptor support
elif architecture_name == 'GraphGPS':
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GraphGPS",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 200},
            "edge": {"input_dim": 5, "output_dim": 200},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_graph_state": True,  # Enable graph state for descriptors
        "transformer_args": {
            "units": 200,
            "num_heads": 8,
            "ff_units": 800,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_edge_features": True,
            "use_positional_encoding": True,
            "positional_encoding_dim": 64,
            "activation": "relu",
            "layer_norm_epsilon": 1e-6
        },
        "depth": 4,
        "use_set2set": True,
        "set2set_args": {"channels": 32, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
        "pooling_args": {"pooling_method": "segment_sum"},
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphGPS",
            "config": model_config
        },
        "training": {
            "fit": {
                "batch_size": 32,
                "epochs": 400,
                "validation_freq": 10,
                "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler",
                        "config": {
                            "learning_rate_start": 5e-04, "learning_rate_stop": 1e-05, "epo_min": 0, "epo": 400,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 5e-04}},
                "loss": "mean_absolute_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}}
        },
        "data": {
            "dataset": {
                "class_name": "ESOLDataset",
                "module_name": "kgcnn.data.datasets.ESOLDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.1.0"
        }
    }



# PNA (Principal Neighborhood Aggregation) - Enhanced GIN with multiple aggregators and degree scaling
elif architecture_name == 'PNA':
    hyper = {
        "model": {
            "class_name": "make_model_fixed",
            "module_name": "kgcnn.literature.PNA._make_fixed",
            "config": {
                "name": "PNA",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "pna_args": {"units": 128, "use_bias": True, "activation": "relu",
                             "aggregators": ["mean", "max", "min", "std"],
                             "scalers": ["identity", "amplification", "attenuation"],
                             "dropout_rate": 0.1},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to PNA")
        hyper["model"]["config"]["use_graph_state"] = True

# ExpC (Expressive Graph Neural Network with Path Counting) - Enhanced GIN with subgraph counting
elif architecture_name == 'ExpC':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.ExpC",
            "config": {
                "name": "ExpC",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "expc_args": {"units": 128, "use_bias": True, "activation": "relu",
                              "use_subgraph_counting": True, "subgraph_types": ["triangle", "square", "pentagon"],
                              "dropout_rate": 0.1},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to ExpC")
        hyper["model"]["config"]["use_graph_state"] = True

# EGAT (Edge-Guided Graph Attention) - Enhanced GAT with edge feature guidance
elif architecture_name == 'EGAT':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.EGAT",
            "config": {
                "name": "EGAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "egat_args": {"units": 128, "use_bias": True, "activation": "relu",
                              "attention_heads": 8, "attention_units": 64, "use_edge_features": True,
                              "dropout_rate": 0.1},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to EGAT")
        hyper["model"]["config"]["use_graph_state"] = True

# TransformerGAT (Transformer-enhanced GAT) - Combines local GAT with global transformer attention
elif architecture_name == 'TransformerGAT':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.TransformerGAT",
            "config": {
                "name": "TransformerGAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "transformer_gat_args": {"units": 128, "use_bias": True, "activation": "relu",
                                         "attention_heads": 8, "attention_units": 64, "transformer_heads": 8,
                                         "use_edge_features": True, "dropout_rate": 0.1},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              },
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to TransformerGAT")
        hyper["model"]["config"]["use_graph_state"] = True

# GRPE (Graph Relative Positional Encoding Transformer) - Enhanced GraphTransformer with relative positional encoding
elif architecture_name == 'GRPE':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GRPE",
            "config": {
                "name": "GRPE",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "grpe_args": {"units": 128, "use_bias": True, "activation": "relu",
                              "attention_heads": 8, "attention_units": 64, "max_path_length": 10,
                              "use_edge_features": True, "dropout_rate": 0.1},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to GRPE")
        hyper["model"]["config"]["use_graph_state"] = True

# KA-GAT (Kolmogorov-Arnold Graph Attention Network) - Enhanced GAT with Fourier-KAN
elif architecture_name == 'KAGAT':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.KAGAT",
            "config": {
                "name": "KAGAT",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "kagat_args": {"units": 128, "attention_heads": 8, "attention_units": 64,
                               "use_edge_features": True, "use_final_activation": True,
                               "has_self_loops": True, "dropout_rate": 0.1,
                               "hidden_dim": 64, "activation": "relu", "use_bias": True},
                "depth": 4,
                "verbose": 10,
                "pooling_nodes_args": {"pooling_method": "sum"},
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              },
                "metrics": ["accuracy", "balanced_accuracy", "precision", "recall"]
            },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to KAGAT")
        hyper["model"]["config"]["use_graph_state"] = True

# DHTNN (Double-Head Transformer Neural Network) - Enhanced DMPNN with double-head attention
elif architecture_name == 'DHTNN':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DHTNN",
            "config": {
                "name": "DHTNN",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "dhtnn_args": {"units": 128, "local_heads": 4, "global_heads": 4,
                               "local_attention_units": 64, "global_attention_units": 64,
                               "use_edge_features": True, "use_final_activation": True,
                               "has_self_loops": True, "dropout_rate": 0.1,
                               "activation": "relu", "use_bias": True},
                "depth": 4,
                "verbose": 10,
                "pooling_nodes_args": {"pooling_method": "sum"},
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 50, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.0001, "learning_rate_stop": 1e-06, "epo_min": 25, "epo": 50,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"lr": 0.0001,
                                                              "clipnorm": 1.0}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to DHTNN")
        hyper["model"]["config"]["use_graph_state"] = True

# DHTNNPlus (Enhanced Double-Head Transformer Neural Network) - Best of both worlds
elif architecture_name == 'DHTNNPlus':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DHTNNPlus",
            "config": {
                "name": "DHTNNPlus",
                "inputs": [
                    {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128},
                                    "graph": {"input_dim": 100, "output_dim": 64}},
                "dhtnnplus_args": {"units": 128, "local_heads": 4, "global_heads": 4,
                                   "local_attention_units": 64, "global_attention_units": 64,
                                   "use_edge_features": True, "use_collaboration": True,
                                   "collaboration_heads": 4, "use_gru_updates": True,
                                   "use_final_activation": True, "has_self_loops": True,
                                   "dropout_rate": 0.1, "activation": "relu", "use_bias": True},
                "depth": 4,
                "verbose": 10,
                "pooling_nodes_args": {"pooling_method": "sum"},
                "use_graph_state": True,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to DHTNNPlus")
        hyper["model"]["config"]["use_graph_state"] = True

# AWARE (Attentive Walk-Aggregating Graph Neural Networks) - Walk-based attention for graph-level prediction
elif architecture_name == 'AWARE':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.AWARE",
            "config": {
                "name": "AWARE",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128}},
                "aware_args": {"units": 128, "walk_length": 3, "num_walks": 10,
                               "attention_heads": 4, "dropout_rate": 0.1,
                               "activation": "relu", "use_bias": True},
                "depth": 4,
                "verbose": 10,
                "use_graph_state": False,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to AWARE")
        hyper["model"]["config"]["use_graph_state"] = True

# GraphGPS (Graph Generalization with Positional and Structural encoding) - General-purpose graph neural network
elif architecture_name == 'GraphGPS':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphGPS",
            "config": {
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                "input_edge_embedding": {"input_dim": 5, "output_dim": 128},
                "use_graph_state": False,
                "graphgps_args": {
                    "units": 128,
                    "heads": 8,
                    "dropout": 0.1,
                    "use_bias": True,
                    "activation": "relu",
                    "mp_type": "gcn",
                    "attn_type": "multihead",
                    "use_skip_connection": True,
                    "use_layer_norm": True,
                    "use_batch_norm": False
                },
                "depth": 4,
                "node_dim": 128,
                "use_set2set": False,
                "output_embedding": "graph",
                "output_to_tensor": True,
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to GraphGPS")
        hyper["model"]["config"]["input_graph_embedding"] = {"input_dim": 100, "output_dim": 64}
        hyper["model"]["config"]["use_graph_state"] = True

# MoGAT (Multi-order Graph Attention Network) - Water solubility prediction and interpretation
elif architecture_name == 'MoGAT':
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MoGAT",
            "config": {
                "name": "MoGAT",
                "inputs": [{"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],
                "input_embedding": {"node": {"input_dim": 95, "output_dim": 128},
                                    "edge": {"input_dim": 5, "output_dim": 128}},
                "attention_args": {"units": 128},
                "pooling_gat_nodes_args": {"pooling_method": "mean"},
                "depthato": 4,
                "depthmol": 4,
                "dropout": 0.2,
                "verbose": 10,
                "output_embedding": "graph",
                "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                               "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {
                        "class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                            "learning_rate_start": 0.001, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 200,
                            "verbose": 0
                        }
                    }
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001,
                                                                       "weight_decay": 1e-05}
                              }
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
            "execute_folds": 1
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "kgcnn_version": "2.0.3"
        }
    }
    
    # Add descriptor input if using descriptors
    if use_descriptors and descs:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if 'graph_descriptors' not in input_names:
            hyper["model"]["config"]["inputs"].append(
                {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
            )
            print(f"Added descriptor input with dimension {desc_dim} to MoGAT")
        hyper["model"]["config"]["use_graph_state"] = True

# ContrastiveGIN implementation
elif architecture_name == 'ContrastiveGIN':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveGIN architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveGIN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "gin_args": {
        "units": 128,
            "use_bias": True,
            "activation": "relu",
            "use_normalization": True,
            "normalization_technique": "graph_batch"
        },
        "depth": 5,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_gin_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                    "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                    "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                        {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastiveGAT implementation
elif architecture_name == 'ContrastiveGAT':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveGAT architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveGAT",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "attention_args": {
            "units": 128,
            "use_bias": True,
            "activation": "relu",
            "use_edge_features": True
        },
        "pooling_nodes_args": {
            "pooling_method": "sum"
        },
        "depth": 3,
        "attention_heads_num": 8,
        "attention_heads_concat": True,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_gat_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }



# ContrastiveGATv2 implementation
elif architecture_name == 'ContrastiveGATv2':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveGATv2 architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveGATv2",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "attention_args": {
            "units": 128,
            "use_bias": True,
            "activation": "relu",
            "use_edge_features": True
        },
        "pooling_nodes_args": {
            "pooling_method": "sum"
        },
        "depth": 3,
        "attention_heads_num": 8,
        "attention_heads_concat": True,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_gatv2_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }



# ContrastiveDMPNN implementation
elif architecture_name == 'ContrastiveDMPNN':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveDMPNN architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveDMPNN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "pooling_args": {
            "pooling_method": "sum"
        },
        "edge_initialize": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "edge_dense": {
            "units": 128,
            "use_bias": True,
            "activation": "linear"
        },
        "edge_activation": {
            "activation": "relu"
        },
        "node_dense": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "dropout": {
            "rate": 0.1
        },
        "depth": 3,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False,
            "augmentation_type": "molclr",  # Options: molclr, graphcl, mocl, digmol, claps, random
            "augmentation_args": {
                "node_mask_rate": 0.15,
                "edge_drop_rate": 0.15,
                "feature_noise_std": 0.01
            }
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_dmpnn_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastiveAttFP implementation
elif architecture_name == 'ContrastiveAttFP':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveAttFP architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveAttFP",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "attention_args": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "depthato": 3,
        "depthmol": 3,
        "dropout": 0.1,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_attfp_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastiveAddGNN implementation
elif architecture_name == 'ContrastiveAddGNN':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveAddGNN architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveAddGNN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 2], "name": "edge_image", "dtype": "int64", "ragged": True},
            {"shape": [None, 3, 3], "name": "lattice", "dtype": "float32", "ragged": False},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "geometric_edge": False,
        "make_distance": False,
        "expand_distance": False,
        "gauss_args": {
            "bins": 20,
            "distance": 4.0,
            "sigma": 0.4,
            "offset": 0.0,
            "dimension": 1
        },
        "set2set_args": {
            "channels": 64,
            "T": 3,
            "pooling_method": "mean",
            "init_method": "mean"
        },
        "pooling_args": {
            "pooling_method": "sum"
        },
        "addgnn_args": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "use_set2set": False,
        "node_dim": 128,
        "depth": 3,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_addgnn_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastiveDGIN implementation
elif architecture_name == 'ContrastiveDGIN':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveDGIN architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveDGIN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "pooling_args": {
            "pooling_method": "sum"
        },
        "edge_initialize": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "edge_dense": {
            "units": 128,
            "use_bias": True,
            "activation": "linear"
        },
        "edge_activation": {
            "activation": "relu"
        },
        "node_dense": {
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "dropoutDMPNN": {
            "rate": 0.1
        },
        "dropoutGIN": {
            "rate": 0.1
        },
        "depthDMPNN": 3,
        "depthGIN": 3,
        "gin_args": {
            "pooling_method": "sum",
            "epsilon_learnable": False
        },
        "gin_mlp": {
            "units": [128, 128],
            "use_bias": True,
            "activation": ["relu", "linear"],
            "use_normalization": True,
            "normalization_technique": "graph_batch"
        },
        "last_mlp": {
            "units": [128, 128],
            "use_bias": True,
            "activation": ["relu", "linear"],
            "use_normalization": True,
            "normalization_technique": "graph_batch"
        },
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False,
            "augmentation_type": "molclr",  # Default augmentation strategy
            "augmentation_args": {
                "node_mask_rate": 0.15,
                "edge_drop_rate": 0.15,
                "subgraph_ratio": 0.8,
                "feature_noise_std": 0.01
            }
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_dgin_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}},
                    {"map_list": {"method": "set_edge_indices_reverse"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastivePNA implementation
elif architecture_name == 'ContrastivePNA':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastivePNA architecture!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastivePNA",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "pna_args": {
            "aggregators": ["mean", "max", "min", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
            "units": 128,
            "use_bias": True,
            "activation": "relu"
        },
        "depth": 3,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_pna_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ContrastiveMoE implementation - The most efficient approach!
elif architecture_name == 'ContrastiveMoE':
    print(f"Checking architecture: {architecture_name}")
    print("Found ContrastiveMoE architecture - Combining MoE with Contrastive Learning!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ContrastiveMoE",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "multigraph_moe_args": {
            "num_representations": 4,
            "num_experts": 3,
            "expert_types": ["gin", "gat", "gcn"],
            "representation_types": ["original", "weighted", "augmented", "attention"],
            "use_edge_weights": True,
            "use_node_features": True,
            "use_attention": True,
            "dropout_rate": 0.1,
            "temperature": 1.0,
            "use_noise": True,
            "noise_epsilon": 1e-2,
            "units": 128
        },
        "depth": 3,
        "verbose": 10,
        "pooling_nodes_args": {"pooling_method": "sum"},
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, False], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]},
        "contrastive_args": {
            "use_contrastive_loss": True,
            "contrastive_loss_type": "infonce",
            "temperature": 0.1,
            "contrastive_weight": 0.1,
            "num_views": 2,
            "use_diversity_loss": False,
            "use_auxiliary_loss": False,
            "expert_diversity_weight": 0.05,  # Encourage expert diversity
            "routing_entropy_weight": 0.01,    # Encourage balanced routing
            "use_expert_contrastive": True,    # Apply contrastive learning to experts
            "use_routing_contrastive": True    # Apply contrastive learning to routing
        }
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_contrastive_moe_model",
            "module_name": "kgcnn.literature.ContrastiveGNN",  # Use ContrastiveGNN module
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function,
                "contrastive_weight": 0.1,
                "temperature": 0.1,
                "expert_diversity_weight": 0.05,
                "routing_entropy_weight": 0.01
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# Simple MoE implementation - Lightweight version
elif architecture_name == 'MoE':
    print(f"Checking architecture: {architecture_name}")
    print("Found MoE architecture - Simple MultiGraphMoE without contrastive learning!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "MoE",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "multigraph_moe_args": {
            "num_representations": 4,
            "num_experts": 3,
            "expert_types": ["gin", "gat", "gcn"],
            "representation_types": ["original", "weighted", "augmented", "attention"],
            "use_edge_weights": True,
            "use_node_features": True,
            "use_attention": True,
            "dropout_rate": 0.1,
            "temperature": 1.0,
            "use_noise": True,
            "noise_epsilon": 1e-2,
            "units": 128
        },
        "depth": 3,
        "verbose": 10,
        "pooling_nodes_args": {"pooling_method": "sum"},
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MultiGraphMoE",  # Use MultiGraphMoE as base
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# ConfigurableMoE implementation - Different graph types for each expert!
elif architecture_name == 'ConfigurableMoE':
    print(f"Checking architecture: {architecture_name}")
    print("Found ConfigurableMoE architecture - Different graph types for each expert!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "ConfigurableMoE",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "multigraph_moe_args": {
            "num_representations": 6,  # More representations for different graph types
            "num_experts": 4,  # More experts for different graph types
            "expert_types": ["gin", "gat", "gcn", "graphsage"],  # Different expert types
            "representation_types": [
                "original",           # Standard molecular graph
                "weighted",           # Edge-weighted graph
                "augmented",          # Augmented with additional edges
                "attention",          # Attention-based graph
                "substructure",       # Substructure-focused graph
                "fingerprint"         # Molecular fingerprint graph
            ],
            "expert_graph_configs": {
                "gin": {
                    "graph_types": ["original", "substructure"],
                    "representation_weight": 0.3,
                    "specialization": "molecular_structure"
                },
                "gat": {
                    "graph_types": ["attention", "weighted"],
                    "representation_weight": 0.25,
                    "specialization": "attention_patterns"
                },
                "gcn": {
                    "graph_types": ["original", "augmented"],
                    "representation_weight": 0.25,
                    "specialization": "graph_connectivity"
                },
                "graphsage": {
                    "graph_types": ["fingerprint", "weighted"],
                    "representation_weight": 0.2,
                    "specialization": "molecular_fingerprints"
                }
            },
            "use_edge_weights": True,
            "use_node_features": True,
            "use_attention": True,
            "dropout_rate": 0.1,
            "temperature": 1.0,
            "use_noise": True,
            "noise_epsilon": 1e-2,
            "units": 128
        },
        "depth": 3,
        "verbose": 10,
        "pooling_nodes_args": {"pooling_method": "sum"},
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions and activation based on config file
    print(f"Before update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"Before update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    model_config = update_output_dimensions(model_config, architecture_name)
    print(f"After update_output_dimensions: output_mlp units = {model_config['output_mlp']['units']}")
    print(f"After update_output_dimensions: output_mlp activation = {model_config['output_mlp']['activation']}")
    
    hyper = {
        "model": {
            "class_name": "make_configurable_moe_model",
            "module_name": "kgcnn.literature.MultiGraphMoE",  # Use MultiGraphMoE as base
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# MultiChem implementation with descriptors support
elif architecture_name == 'MultiChem':
    print(f"Checking architecture: {architecture_name}")
    print("Found MultiChem architecture with DUAL processing and descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "MultiChem",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 1], "name": "edge_indices_reverse", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "use_directed": True,
        "use_dual_features": True,
        "units": 128,
        "num_heads": 8,
        "depth": 4,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "use_residual": True,
        "pooling_args": {"pooling_method": "sum", "use_dual_features": True},
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [256, 128, output_dim],
                     "activation": ["relu", "relu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MultiChem",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# GraphSAGE implementation with descriptors support
elif architecture_name == 'GraphSAGE':
    print(f"Checking architecture: {architecture_name}")
    print("Found GraphSAGE architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "GraphSAGE",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 64},
            "edge": {"input_dim": 32, "output_dim": 32}
        },
        "node_mlp_args": {"units": [64, 32], "use_bias": True, "activation": ["relu", "linear"]},
        "edge_mlp_args": {"units": 64, "use_bias": True, "activation": "relu"},
        "pooling_args": {"pooling_method": "segment_mean"},
        "pooling_nodes_args": {"pooling_method": "sum"},
        "gather_args": {},
        "concat_args": {"axis": -1},
        "use_edge_features": True,
        "depth": 3,
        "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True, False], "units": [64, 32, output_dim],
                     "activation": ["relu", "relu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.GraphSAGE",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 500, "validation_freq": 10, "verbose": 2,
                    "callbacks": [{"class_name": "kgcnn>LinearLearningRateScheduler",
                                   "config": {"learning_rate_start": 0.5e-3, "learning_rate_stop": 1e-5,
                                              "epo_min": 400, "epo": 500, "verbose": 0}}]
                    },
            "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 5e-3}},
                        "loss": "mean_absolute_error"
                        },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    }

# MoGAT implementation with descriptors support
elif architecture_name == 'MoGAT':
    print(f"Checking architecture: {architecture_name}")
    print("Found MoGAT architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "MoGAT",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "equiv_initialize_kwargs": {"dim": 3, "method": "zeros"},
        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
        "pooling_args": {"pooling_method": "sum"},
        "conv_args": {"units": 128, "cutoff": None, "conv_pool": "sum"},
        "update_args": {"units": 128},
        "equiv_normalization": False, 
        "node_normalization": False,
        "depth": 3,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph", 
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True], "units": [128, output_dim], "activation": ["swish", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MoGAT",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# MultiGraphMoE implementation with descriptors support
elif architecture_name == 'MultiGraphMoE':
    print(f"Checking architecture: {architecture_name}")
    print("Found MultiGraphMoE architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "MultiGraphMoE",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "depth": 4,
        "multigraph_moe_args": {
            "units": 128,
            "num_experts": 4,
            "expert_types": ["GIN", "GAT", "GCN", "GraphSAGE","MNPN"],
            "dropout_rate": 0.1,
            "temperature": 1.0,
            "use_noise": True,
            "noise_epsilon": 0.01
        },
        "pooling_nodes_args": {"pooling_method": "sum"},
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.MultiGraphMoE",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# Schnet implementation with descriptors support
elif architecture_name == 'Schnet':
    print(f"Checking architecture: {architecture_name}")
    print("Found Schnet architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "Schnet",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "interaction_args": {"units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus"},
        "node_pooling_args": {"pooling_method": "sum"},
        "depth": 4,
        "gauss_args": {"bins": 20, "distance": 4, "offset": 0.0, "sigma": 0.4},
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.Schnet",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# HamNet implementation with descriptors support
elif architecture_name == 'HamNet':
    print(f"Checking architecture: {architecture_name}")
    print("Found HamNet architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "HamNet",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 64},
            "edge": {"input_dim": 5, "output_dim": 64}
        },
        "message_kwargs": {"units": 200, "units_edge": 200, "rate": 0.2, "use_dropout": True},
        "fingerprint_kwargs": {"units": 200, "units_attend": 200, "rate": 0.5, "use_dropout": True, "depth": 3},
        "gru_kwargs": {"units": 200},
        "union_type_node": "gru",
        "union_type_edge": "None",
        "given_coordinates": True,
        "depth": 3,
        "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, False], "units": [200, output_dim],
                     "activation": ['relu', 'linear'],
                     "use_dropout": [True, False],
                     "rate": [0.5, 0.0]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.HamNet",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 16, "epochs": 800, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Addons>AdamW", "config": {"lr": 0.001, "weight_decay": 1e-05}},
                "loss": "mean_squared_error"
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.2"
        }
    }

# PAiNN implementation with descriptors support
elif architecture_name == 'PAiNN':
    print(f"Checking architecture: {architecture_name}")
    print("Found PAiNN architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "PAiNN",
        "inputs": [
            {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {"node": {"input_dim": 95, "output_dim": 128}},
        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
        "pooling_args": {"pooling_method": "sum"},
        "conv_args": {"units": 128, "cutoff": None},
        "update_args": {"units": 128},
        "depth": 3,
        "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.PAiNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 30.0, "decay_steps": 40000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 3, "max_neighbours": 10000}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# DimeNetPP implementation with descriptors support
elif architecture_name == 'DimeNetPP':
    print(f"Checking architecture: {architecture_name}")
    print("Found DimeNetPP architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "DimeNetPP",
        "inputs": [
            {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128,
                     "embeddings_initializer": {"class_name": "RandomUniform",
                                                "config": {"minval": -1.7320508075688772,
                                                           "maxval": 1.7320508075688772}}}},
        "emb_size": 128, "out_emb_size": 256, "int_emb_size": 64, "basis_emb_size": 8,
        "num_blocks": 4, "num_spherical": 7, "num_radial": 6,
        "cutoff": 5.0, "envelope_exponent": 5,
        "num_before_skip": 1, "num_after_skip": 2, "num_dense_output": 3,
        "num_targets": 128, "extensive": False, "output_init": "zeros",
        "activation": "swish", "verbose": 10,
        "output_embedding": "graph",
        "output_mlp": {"use_bias": [True, False], "units": [128, 1],
                     "activation": ["swish", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.DimeNetPP",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 10, "epochs": 872, "validation_freq": 10, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {
                    "class_name": "Addons>MovingAverage", "config": {
                        "optimizer": {
                            "class_name": "Adam", "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                        "learning_rate": 0.001, "warmup_steps": 30.0, "decay_steps": 40000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                },
                "loss": "mean_absolute_error",
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"map_list": {"method": "set_range", "max_distance": 4, "max_neighbours": 20}},
                    {"map_list": {"method": "set_angle"}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "2.0.3"
        }
    }

# EGNN implementation with descriptors support
elif architecture_name == 'EGNN':
    print(f"Checking architecture: {architecture_name}")
    print("Found EGNN architecture with descriptors!")
    # Define model configuration in Python and update output dimensions
    model_config = {
        "name": "EGNN",
        "inputs": [
            {"shape": [None, 41], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True},
            {"shape": [None, 11], "name": "edge_attributes", "dtype": "float32", "ragged": True},
            {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
        ],
        "input_embedding": {
            "node": {"input_dim": 95, "output_dim": 128},
            "edge": {"input_dim": 5, "output_dim": 128},
            "graph": {"input_dim": 100, "output_dim": 64}
        },
        "equiv_initialize_kwargs": {"dim": 3, "method": "zeros"},
        "pooling_args": {"pooling_method": "sum"},
        "conv_args": {"units": 128, "conv_pool": "sum"},
        "update_args": {"units": 128},
        "depth": 4,
        "verbose": 10,
        "use_graph_state": True,
        "output_embedding": "graph",
        "output_to_tensor": True,
        "output_mlp": {"use_bias": [True, True, True], "units": [200, 100, output_dim],
                     "activation": ["kgcnn>leaky_relu", "selu", "linear"]}
    }
    
    # Update output dimensions based on config file
    model_config = update_output_dimensions(model_config, architecture_name)
    
    hyper = {
        "model": {
            "class_name": "make_model",
            "module_name": "kgcnn.literature.EGNN",
            "config": model_config
        },
        "training": {
            "fit": {"batch_size": 32, "epochs": 200, "validation_freq": 1, "verbose": 2, "callbacks": []
                    },
            "compile": {
                "optimizer": {"class_name": "Adam",
                              "config": {"lr": {
                                  "class_name": "ExponentialDecay",
                                  "config": {"initial_learning_rate": 0.001,
                                             "decay_steps": 1600,
                                             "decay_rate": 0.5, "staircase": False}
                              }
                              }
                },
                "loss": loss_function
            },
            "cross_validation": {"class_name": "KFold",
                                 "config": {"n_splits": 5, "random_state": None, "shuffle": True}},
        },
        "data": {
            "dataset": {
                "class_name": "MoleculeNetDataset",
                "config": {},
                "methods": [
                    {"set_attributes": {}}
                ]
            },
            "data_unit": "mol/L"
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "3.0.0"
        }
    }

# Print to visually make sure we have parsed correctly the parameters
print("My parameters")
print("Loss", nn_loss)
print("LSTM", nn_lstmunits)
print("DENSE", nn_denseunits)
print("PROBA", nn_return_proba)
print("LR start", nn_lr_start)

# parameters of the model for OCHEM
random_seed = seed
np.random.seed(seed);

# Check against other keras models in OCHEM
log_filename = 'model.log';
modelname = "model.h5";

# us to check cuda GPU vs ARM and to use CPU also!
if gpu >= 0:
    physical_devices = tf.config.list_physical_devices('GPU')
    if tf.test.is_built_with_cuda():
        try:
            # work on one card with cleaner memory growth settings
            tf.config.set_visible_devices(physical_devices[gpu], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu], True)
            print('growth_memory done')
        except:
            print("use tf1 method for growth_memory")
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
        print('cuda gpu')
    else:
        print('mac gpu')
else:
    tf.config.set_visible_devices([], 'GPU')
    print('using cpu')
print('before', TRAIN)

if TRAIN == "True":
    # change the number of output (target) of the model
    # define columns names to grab from the input file
    # read the files

    cols = ['Result%s' % (i) for i in range(output_dim)]
    descs = ['desc%s' % (i) for i in range(desc_dim)]

    # Initialize hyper if not defined
    if 'hyper' not in locals():
        hyper = None

    if len(descs)>0:
        print('There are Additional Descriptors/Conditions')
        if 'hyper' not in locals() or hyper is None:
            print('Warning: hyper not defined, skipping descriptor processing')
        elif 'use_graph_state' in hyper["model"]["config"].keys():
            print(hyper["model"]["config"]['use_graph_state'])
            hyper["model"]["config"]['use_graph_state']=True
            # For AttentiveFP, ensure we have exactly the right inputs in the right order
            if hyper["model"]["config"]["name"] == 'AttentiveFP':
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            elif hyper["model"]["class_name"] == 'make_model_edge' and hyper["model"]["config"]["name"] == 'GIN':
                # For GINE (edge model), ensure we have exactly the right inputs in the right order
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            elif hyper["model"]["config"]["name"] == 'GIN':
                # For GIN, ensure we have exactly the right inputs in the right order
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            else:
                # For other models, check if graph_descriptors is already in inputs to avoid duplicates
                input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
                if 'graph_descriptors' not in input_names:
                    hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}
            model_name=hyper["model"]["config"]["name"]
            model_module=hyper["model"]["module_name"]
            model_class=hyper["model"]["class_name"]
        else:
            print('Model does not support use_graph_state, adding descriptor input manually')
            # Add descriptor input manually for models that don't have use_graph_state
            input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
            if 'graph_descriptors' not in input_names:
                hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            # Add graph embedding to input_embedding if it doesn't exist
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}


    # Clean up duplicate inputs for DMPNN
    if 'hyper' in locals() and hyper is not None and hyper["model"]["config"]["name"] in ['DMPNN', 'ChemProp']:
        input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
        if "graph_descriptors" in input_names:
            # Ensure we only have one graph_descriptors input
            graph_desc_inputs = [input_config for input_config in hyper["model"]["config"]["inputs"] 
                               if input_config["name"] == "graph_descriptors"]
            if len(graph_desc_inputs) > 1:
                print("Cleaning up: Removing duplicate graph_descriptors inputs for DMPNN")
            hyper["model"]["config"]["inputs"] = [
                input_config for input_config in hyper["model"]["config"]["inputs"] 
                if input_config["name"] != "graph_descriptors"
            ]
            # Add back one graph_descriptors input
            hyper["model"]["config"]["inputs"].append(graph_desc_inputs[0])
    
    # Check if architecture was found
    if 'hyper' not in locals() or hyper is None:
        raise ValueError(f"Architecture '{architecture_name}' is not implemented. Available architectures: GCN, GAT, GATv2, CMPNN, CoAttentiveFP, AttentiveFPPlus, CMPNNPlus, DMPNNAttention, DGIN, AddGNN, GraphTransformer, RGCN, rGIN, rGINE, GIN, GINE, GraphGPS, PNA, ExpC, EGAT, TransformerGAT, GRPE, KAGAT, DHTNN, DHTNNPlus, AWARE, MoGAT, ContrastiveGIN")
    
    print(f"Selected architecture: {architecture_name}")
    print(f"Model name: {hyper['model']['config']['name']}")
    
    # failed next line for GINE parameters
    hyperparams = HyperParameter(hyper,
                                 model_name=hyper["model"]["config"]["name"],
                                 model_module=hyper["model"]["module_name"],
                                 model_class=hyper["model"]["class_name"],
                                 dataset_name="MoleculeNetDataset")

    inputs = hyperparams["model"]["config"]["inputs"]
    print(hyper)
    print('training')
    print('is overwrite',overwrite)
    dataset, invalid= prepData(TRAIN_FILE, cols, hyper=hyperparams, 
        modelname=architecture_name, overwrite=overwrite, descs=descs)

    # Fix input names for models with descriptors - standardize to graph_descriptors
    if descs:
        print("Processing input names for descriptors...")
        input_names = [input_config["name"] for input_config in hyperparams["model"]["config"]["inputs"]]
        
        # Check if we have graph_descriptors input
        if "graph_descriptors" in input_names:
            print(f"{architecture_name} already has graph_descriptors input - perfect!")
        else:
            print("Warning: No descriptor input found despite descriptors being enabled")

    print(f"Validating model inputs: {[input_config['name'] for input_config in hyperparams['model']['config']['inputs']]}")
    dataset.assert_valid_model_input(hyperparams["model"]["config"]["inputs"])  # failed for GCN code here

    # Model identification
    if architecture_name.startswith('Contrastive'):
        # Import contrastive model directly
        print(f"Creating contrastive model: {architecture_name}")
        from kgcnn.literature.ContrastiveGNN import make_contrastive_gin_model
        make_model = make_contrastive_gin_model
    else:
        make_model = get_model_class(hyperparams["model"]["module_name"], hyperparams["model"]["class_name"])

    # check Dataset
    data_name = dataset.dataset_name
    data_unit = hyperparams["data"]["data_unit"]
    data_length = dataset.length

    labels = np.array(dataset.obtain_property("graph_labels"), dtype="float")
    # data preparation
    xtrain, ytrain, xtest, ytest, hyperparam, isClass = splitingTrain_Val(dataset,labels,data_length, inputs = inputs, hypers=hyper,  idx=0, cuts=10, output_dim=output_dim, seed=seed)

    print("dataset length:", len(labels))

    # hyper_fit and epochs
    hyper_fit = hyperparam['training']['fit']
    epo = hyper_fit['epochs']
    epostep = hyper_fit['validation_freq']

    # how it works in Keras the Train / Val split
    filename = 'loss.csv'

    hl = CSVLogger(filename, separator=",", append=True)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=earlystop_patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=redlr_factor, verbose=0, patience=redlr_patience,
                                  min_lr=1e-5)

    model = None
    # Make model
    if architecture_name.startswith('Contrastive'):
        # For contrastive models, use the appropriate contrastive function
        print(f"Creating model with config: {hyperparam['model']['config']}")
        if architecture_name == 'ContrastiveGIN':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_gin_model
            model = make_contrastive_gin_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveGAT':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_gat_model
            model = make_contrastive_gat_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveGATv2':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_gatv2_model
            model = make_contrastive_gatv2_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveDMPNN':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_dmpnn_model
            model = make_contrastive_dmpnn_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveAttFP':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_attfp_model
            model = make_contrastive_attfp_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveAddGNN':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_addgnn_model
            model = make_contrastive_addgnn_model(**hyperparam['model']["config"])
        elif architecture_name == 'AddGNN-PNA':
            from kgcnn.literature.AddGNN import make_addgnn_pna_model
            model = make_addgnn_pna_model(**hyperparam['model']["config"])
        elif architecture_name == 'CMPNN-PNA':
            from kgcnn.literature.CMPNN import make_cmpnn_pna_model
            model = make_cmpnn_pna_model(**hyperparam['model']["config"])
        elif architecture_name == 'RGCN-PNA':
            from kgcnn.literature.RGCN import make_rgcn_pna_model
            model = make_rgcn_pna_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveDGIN':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_dgin_model
            model = make_contrastive_dgin_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastivePNA':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_pna_model
            model = make_contrastive_pna_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveMoE':
            from kgcnn.literature.ContrastiveGNN import make_contrastive_moe_model
            model = make_contrastive_moe_model(**hyperparam['model']["config"])
        elif architecture_name == 'MoE':
            from kgcnn.literature.MultiGraphMoE import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'ConfigurableMoE':
            from kgcnn.literature.MultiGraphMoE import make_configurable_moe_model
            model = make_configurable_moe_model(**hyperparam['model']["config"])
        elif architecture_name == 'MultiChem':
            from kgcnn.literature.MultiChem import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'Schnet':
            from kgcnn.literature.Schnet import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'HamNet':
            from kgcnn.literature.HamNet import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'EGNN':
            from kgcnn.literature.EGNN import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'DimeNetPP':
            from kgcnn.literature.DimeNetPP import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'HDNNP2nd':
            from kgcnn.literature.HDNNP2nd import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'Megnet':
            from kgcnn.literature.Megnet import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'CGCNN':
            from kgcnn.literature.CGCNN import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'INorp':
            from kgcnn.literature.INorp import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'RGCN':
            from kgcnn.literature.RGCN import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'GNNFilm':
            from kgcnn.literature.GNNFilm import make_model
            model = make_model(**hyperparam['model']["config"])
        elif architecture_name == 'ContrastiveGNN':
            from kgcnn.literature.ContrastiveGNN import make_model
            model = make_model(**hyperparam['model']["config"])
        else:
            # For other contrastive models, use the complex ContrastiveGNN module
            model = make_model(**hyperparam['model']["config"])
            print("Contrastive model created successfully!")
    else:
        # For models that support descriptors, we need to keep use_graph_state
        if hyperparam["model"]["config"]["name"] in ['DMPNN', 'ChemProp', 'DGIN', 'GraphGPS']:
            print(f"Creating {hyperparam['model']['config']['name']} model with use_graph_state support")
            model = make_model(**hyperparam['model']["config"])
        else:
            # Fix to make models working with the saved old ChemProp models
            r = dict(hyperparam["model"]["config"])
            # Remove parameters that might not be accepted by all models
            if 'use_graph_state' in r:
                del r['use_graph_state']
            if 'name' in r:
                del r['name']
            model = make_model(**r)

    opt = tf.optimizers.legacy.Adam(learning_rate=cfg_learning_rate)
    

    
    train_data_file = getConfig("Task", "train_data_file", "train_with_conditions.csv")
    try:
        df = pd.read_csv(train_data_file)
        target_cols = [col for col in df.columns if col.lower().startswith("result")]
        if len(target_cols) == 0:
            target_cols = ["Result0"]
        
        # Check if this is a classification task based on lossdef
        is_classification = lossdef in ["BCEmask", "CCEmask", "binary_crossentropy", "categorical_crossentropy"]
        
        if is_classification:
            # For classification tasks, use classification metrics regardless of NaNs
            print("Classification task detected, using classification metrics.")
            loss_fn = loss_function  # Use the original loss function (BCEmask)
            metric_fn = 'accuracy'
        else:
            # For regression tasks, check for NaNs
            if df[target_cols].isnull().values.any():
                print("NaNs detected in targets, using RMSEmask loss.")
                loss_fn = RMSEmask
                # Use masked metric for NaN handling
                metric_fn = MaskedRMSE()
            else:
                print("No NaNs in targets, using standard RMSE loss.")
                loss_fn = "mean_squared_error"
                metric_fn = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
    except Exception as e:
        print(f"Could not check for NaNs in targets: {e}")
        # Default based on lossdef
        if lossdef in ["BCEmask", "CCEmask", "binary_crossentropy", "categorical_crossentropy"]:
            loss_fn = loss_function
            metric_fn = 'accuracy'
        else:
            loss_fn = "mean_squared_error"
            metric_fn = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
    # --- End auto-switch block ---
    
    # need to change this for class / regression
    if architecture_name.startswith('Contrastive'):
        # Use special compilation for all contrastive models
        from kgcnn.literature.ContrastiveGNN import compile_contrastive_gnn_model
        compile_contrastive_gnn_model(
            model=model,
            optimizer=opt,
            learning_rate=cfg_learning_rate,
            loss=loss_function,
            metrics=['accuracy'] if lossdef in ["BCEmask", "CCEmask", "binary_crossentropy", "categorical_crossentropy"] else [metric_fn],
            contrastive_weight=hyperparam['training']['compile'].get('contrastive_weight', 0.1),
            diversity_weight=hyperparam['training']['compile'].get('diversity_weight', 0.01),
            alignment_weight=hyperparam['training']['compile'].get('alignment_weight', 0.01),
            use_regression_aware=hyperparam['training']['compile'].get('use_regression_aware', False),
            target_similarity_threshold=hyperparam['training']['compile'].get('target_similarity_threshold', 0.1),
            similarity_metric=hyperparam['training']['compile'].get('similarity_metric', 'euclidean')
        )
    else:
        # Use appropriate metrics based on loss function
        if lossdef in ["BCEmask", "CCEmask", "binary_crossentropy", "categorical_crossentropy"]:
            # Debug: Try using a simple loss function
            print("DEBUG: Using binary_crossentropy for debugging")
            model.compile(opt, loss="binary_crossentropy", metrics=['accuracy'])
        else:
            model.compile(opt, loss=loss_function, metrics=[metric_fn])

    print(model.summary())

    # Start and time training
    hyper_fit = hyperparam['training']['fit']
    start = time.process_time()

    # Debug: Print input shapes and names
    print(f"DEBUG: Model expects {len(model.inputs)} inputs")
    print(f"DEBUG: Model input names: {[inp.name for inp in model.inputs]}")
    print(f"DEBUG: xtrain has {len(xtrain)} inputs")
    if hasattr(xtrain, '__iter__'):
        for i, x in enumerate(xtrain):
            print(f"DEBUG: xtrain[{i}] shape: {x.shape}, type: {type(x)}")
    
    # Debug: Print loss function and model configuration
    print(f"DEBUG: Using loss function: {loss_function}")
    print(f"DEBUG: Loss function type: {type(loss_function)}")
    print(f"DEBUG: Model config - last_mlp: {hyperparam['model']['config'].get('last_mlp', 'NOT_FOUND')}")
    print(f"DEBUG: Model config - output_mlp: {hyperparam['model']['config'].get('output_mlp', 'NOT_FOUND')}")
    print(f"DEBUG: Model config - use_graph_state: {hyperparam['model']['config'].get('use_graph_state', 'NOT_FOUND')}")
    
    # need to change that to have ragged not numpy or tensor error
    hist = model.fit(xtrain, ytrain,
                     validation_data=(xtest, ytest),
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=nbepochs,
                     verbose=2,
                     callbacks=[es, reduce_lr, hl]
                     )

    model.save_weights(modelname)
    print("Saved model to disk")
    
    # Save descriptor usage state for consistent inference
    if use_descriptors and descs:
        hyperparam["model"]["config"]["_descriptors_used"] = True
        hyperparam["model"]["config"]["_desc_dim"] = desc_dim
        print(f"✅ Saved descriptor usage state: descriptors_used=True, desc_dim={desc_dim}")
    else:
        hyperparam["model"]["config"]["_descriptors_used"] = False
        hyperparam["model"]["config"]["_desc_dim"] = 0
        print("✅ Saved descriptor usage state: descriptors_used=False")
    
    pickle.dump(hyperparam, open("modelparameters.p", "wb"))

    # Probably this should become model.save_weights()
    tar = tarfile.open(MODEL_FILE, "w:gz");
    tar.add(modelname);
    tar.add("modelparameters.p")

    tar.close();

    try:
        os.remove(modelname);
        os.remove("modelparameters.p")
    except:
        pass;

    print("Relax!");

else:
    # Look in OCHEM how other Keras models are stored.
    tar = tarfile.open(MODEL_FILE);
    tar.extractall();
    tar.close();

    print("Loaded model from disk")
    hyper = pickle.load(open("modelparameters.p", "rb"))
    
    # RESTORE descriptor usage state for consistent inference
    if hasattr(hyper, 'get') and callable(hyper.get) and 'model' in hyper and 'config' in hyper['model']:
        if '_descriptors_used' in hyper['model']['config']:
            descriptors_used = hyper['model']['config']['_descriptors_used']
            saved_desc_dim = hyper['model']['config'].get('_desc_dim', 0)
            print(f"🔍 Restored descriptor usage state: descriptors_used={descriptors_used}, desc_dim={saved_desc_dim}")
            
            # Override current descriptor settings with saved state
            use_descriptors = descriptors_used
            desc_dim = saved_desc_dim
            descs = ['desc%s' % (i) for i in range(desc_dim)] if descriptors_used else []
            
            # Ensure use_graph_state matches the saved state
            if descriptors_used:
                hyper['model']['config']['use_graph_state'] = True
                # Ensure descriptor input is present
                input_names = [inp['name'] for inp in hyper['model']['config']['inputs']]
                if 'graph_descriptors' not in input_names:
                    hyper['model']['config']['inputs'].append(
                        {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                    )
                    print(f"✅ Added descriptor input with dimension {desc_dim} for consistent inference")
            else:
                # Special case for MultiChem: if descriptors are being used, keep use_graph_state=True
                if hyper['model']['config']['name'] == 'MultiChem' and len(descs) > 0:
                    hyper['model']['config']['use_graph_state'] = True
                    print("✅ MultiChem: Keeping use_graph_state=True for descriptor usage")
                else:
                    hyper['model']['config']['use_graph_state'] = False
                    # Remove descriptor input if present
                    hyper['model']['config']['inputs'] = [inp for inp in hyper['model']['config']['inputs'] 
                                                         if inp['name'] != 'graph_descriptors']
                    print("✅ Removed descriptor input for consistent inference")
        else:
            print("⚠️  No descriptor usage state found in saved model, using current settings")
    else:
        print("⚠️  Could not access model config, using current settings")
    
    # SAFETY CHECK: Ensure output_mlp activation is properly formatted
    # Handle both HyperParameter objects and regular dictionaries
    try:
        if hasattr(hyper, 'get') and callable(hyper.get):
            # It's a regular dictionary-like object
            if 'model' in hyper and 'config' in hyper['model'] and 'output_mlp' in hyper['model']['config']:
                output_mlp = hyper['model']['config']['output_mlp']
                if 'activation' in output_mlp:
                    # If activation is a string but units is a list, convert activation to a list
                    if isinstance(output_mlp.get('units', []), list) and isinstance(output_mlp['activation'], str):
                        # Create a list with the same length as units, filled with the activation string
                        units_list = output_mlp['units']
                        output_mlp['activation'] = [output_mlp['activation']] * len(units_list)
                        print(f"⚠️  Converted activation from string to list for {hyper['model']['config']['name']}: {output_mlp['activation']}")
                    # If activation is missing, set it to linear
                    elif 'activation' not in output_mlp:
                        if isinstance(output_mlp.get('units', []), list):
                            units_list = output_mlp['units']
                            output_mlp['activation'] = ['linear'] * len(units_list)
                        else:
                            output_mlp['activation'] = 'linear'
                        print(f"⚠️  Added missing activation for {hyper['model']['config']['name']}: {output_mlp['activation']}")
        else:
            # It's a HyperParameter object, access it differently
            print("⚠️  HyperParameter object detected, skipping activation fix (will be handled later)")
    except Exception as e:
        print(f"⚠️  Error during activation fix: {e}, continuing...")

    print(hyper)

    df = pd.read_csv(APPLY_FILE)

    validsmiles = df['smiles'].apply(lambda x: SmilesOK(x))
    # if 'Result0'  don't add this line
    if not 'Result0' in df.columns:
        df['Result0'] = 0

    df.to_csv(APPLY_FILE, index=False)
    cols = ['Result%s' % (i) for i in range(output_dim)]  # change this for MTL tasks not the case today
    descs = ['desc%s' % (i) for i in range(desc_dim)]

    """
    if len(descs)>0:
        print('There are Additional Descriptors/Conditions')
        if 'use_graph_state' in hyper["model"]["config"].keys():
            print(hyper["model"]["config"]['use_graph_state'])
            hyper["model"]["config"]['use_graph_state']=True
            # For AttentiveFP, ensure we have exactly the right inputs in the right order
            if hyper["model"]["config"]["name"] == 'AttentiveFP':
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            elif hyper["model"]["class_name"] == 'make_model_edge' and hyper["model"]["config"]["name"] == 'GIN':
                # For GINE (edge model), ensure we have exactly the right inputs in the right order
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 11), "name": "edge_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            elif hyper["model"]["config"]["name"] == 'GIN':
                # For GIN, ensure we have exactly the right inputs in the right order
                hyper["model"]["config"]["inputs"] = [
                    {"shape": (None, 41), "name": "node_attributes", "dtype": "float32", "ragged": True},
                    {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                    {"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False}
                ]
            else:
                # For other models, check if graph_descriptors is already in inputs to avoid duplicates
                input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
                if 'graph_descriptors' not in input_names:
                    hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}
            model_name=hyper["model"]["config"]["name"]
            model_module=hyper["model"]["module_name"]
            model_class=hyper["model"]["class_name"]
        else:
            print('Model does not support use_graph_state, adding descriptor input manually')
            # Add descriptor input manually for models that don't have use_graph_state
            input_names = [inp['name'] for inp in hyper["model"]["config"]["inputs"]]
            if 'graph_descriptors' not in input_names:
                hyper["model"]["config"]["inputs"].append({"shape": [desc_dim], "name": "graph_descriptors", "dtype": "float32", "ragged": False})
            # Add graph embedding to input_embedding if it doesn't exist
            if 'input_embedding' in hyper["model"]["config"] and 'graph' not in hyper["model"]["config"]["input_embedding"]:
                hyper["model"]["config"]["input_embedding"]["graph"] = {"input_dim": 100, "output_dim": 64}
    """

    print(hyper)
    print('evaluate/inference moed')
    print('is overwrite',overwrite)
    dataset, invalid= prepData(APPLY_FILE, cols, hyper=hyper, modelname=architecture_name,
                                overwrite=overwrite, descs=descs)

    # Handle HyperParameter object access
    try:
        if hasattr(hyper, 'get') and callable(hyper.get):
            # It's a regular dictionary-like object
            hyper_dict = hyper
        else:
            # It's a HyperParameter object, convert to dictionary
            try:
                hyper_dict = hyper.get_dict() if hasattr(hyper, 'get_dict') else dict(hyper)
                print("⚠️  Converted HyperParameter to dictionary for model creation")
            except Exception as e:
                print(f"⚠️  Error converting HyperParameter: {e}, using as-is")
                hyper_dict = hyper
    except Exception as e:
        print(f"⚠️  Error handling hyper object: {e}, using as-is")
        hyper_dict = hyper

    inputs = hyper_dict["model"]["config"]["inputs"]

    # Check if the saved model file exists
    if not os.path.exists(modelname):
        print(f"❌ Model file {modelname} not found!")
        print("❌ Cannot perform inference without a trained model.")
        print("❌ Please ensure you have a trained model file or run training first.")
        exit(1)
    else:
        print(f"✅ Found model file: {modelname}")
    
    # CRITICAL: For inference, preserve the exact model configuration from training
    # Do NOT modify the model configuration as it should match the saved weights exactly
    print("🔒 Inference mode: Preserving original model configuration from training")
    print(f"Original output_mlp configuration: {hyper_dict['model']['config']['output_mlp']}")
    
    # Check if the saved model was trained with descriptors
    saved_input_names = [inp['name'] for inp in hyper_dict['model']['config']['inputs']]
    has_graph_descriptors = 'graph_descriptors' in saved_input_names
    print(f"Saved model input names: {saved_input_names}")
    print(f"Saved model has graph_descriptors: {has_graph_descriptors}")
    print(f"Current inference has descriptors: {len(descs) > 0}")
    
    # CRITICAL: During inference, NEVER modify the model configuration
    # The saved model configuration must be preserved exactly as it was during training
    print("🔒 Inference mode: Using saved model configuration exactly as trained")
    print(f"Saved model inputs: {[inp['name'] for inp in hyper_dict['model']['config']['inputs']]}")
    print(f"Saved model input_embedding: {hyper_dict['model']['config'].get('input_embedding', 'Not found')}")
    
    # Ensure we use the exact same descriptors configuration as the saved model
    if has_graph_descriptors:
        if len(descs) == 0:
            print("⚠️  WARNING: Saved model was trained WITH descriptors, but current inference has NO descriptors")
            print("⚠️  This will likely cause errors. Consider using descriptors or retraining without descriptors.")
        else:
            print("✅ Descriptors configuration matches saved model")
    else:
        if len(descs) > 0:
            print("⚠️  WARNING: Saved model was trained WITHOUT descriptors, but current inference has descriptors")
            print("⚠️  This will likely cause errors. Consider removing descriptors or retraining with descriptors.")
        else:
            print("✅ No descriptors configuration matches saved model")

    # Model creation
    make_model = get_model_class(hyper_dict["model"]["module_name"], hyper_dict["model"]["class_name"])

    # CRITICAL: During inference, use the exact saved configuration
    # Do NOT modify any parameters, including use_graph_state
    print(f"🔒 Creating model with exact saved configuration")
    print(f"🔒 use_graph_state: {hyper_dict['model']['config'].get('use_graph_state', 'Not found')}")
    model = make_model(**hyper_dict['model']["config"])
    
    # Try to load stored best weights - FAIL if there's any mismatch
    try:
        print(f"Attempting to load weights from {modelname}...")
        model.load_weights(modelname)
        print("✅ Successfully loaded model weights")
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading weights: {e}")
        print("❌ Model architecture mismatch detected!")
        print("❌ This means the training and inference architectures are different.")
        print("❌ The model cannot be used with random weights - this would give meaningless results.")
        print("❌ Please ensure the model architecture is consistent between training and inference.")
        raise RuntimeError(f"Model architecture mismatch: {e}")

    # CRITICAL: Use the exact inputs from the saved model configuration
    print(f"🔒 Using exact inputs from saved model: {[inp['name'] for inp in inputs]}")
    print(f"🔒 Input shapes: {[inp['shape'] for inp in inputs]}")
    
    # Ensure the dataset provides exactly what the model expects
    x_pred = dataset.tensor(inputs)
    
    # Debug: Check what the model expects vs what we're providing
    print(f"🔒 Model input names: {[inp.name for inp in model.inputs]}")
    print(f"🔒 Model input shapes: {[inp.shape for inp in model.inputs]}")
    print(f"🔒 Data tensor names: {[name for name in x_pred.keys()] if isinstance(x_pred, dict) else 'Not a dict'}")
    print(f"🔒 Data tensor shapes: {[tensor.shape for tensor in x_pred.values()] if isinstance(x_pred, dict) else [x_pred.shape] if hasattr(x_pred, 'shape') else 'Unknown'}")
    
    # Ensure we're providing the right number of inputs
    if isinstance(x_pred, dict):
        expected_inputs = set([inp.name for inp in model.inputs])
        provided_inputs = set(x_pred.keys())
        print(f"🔒 Expected inputs: {expected_inputs}")
        print(f"🔒 Provided inputs: {provided_inputs}")
        if expected_inputs != provided_inputs:
            print(f"⚠️  Input mismatch! Missing: {expected_inputs - provided_inputs}, Extra: {provided_inputs - expected_inputs}")

    a_pred = model.predict(x_pred)

    dfres = pd.DataFrame(a_pred, columns=cols)

    # we need to check if the graph has not computed some cases and regenerate the full index
    if len(invalid) > 0:
        dfresall = pd.DataFrame(index=df.index, columns=dfres.columns)
        idx = [i for i in range(len(df)) if i not in invalid]
        dfres.index = idx
        dfresall.loc[idx, :] = dfres.loc[idx, :]

    else:
        dfresall = dfres

    dfresall.to_csv(RESULT_FILE, index=False)

    try:
        os.remove(modelname);
        os.remove("modelparameters.p")
    except:
        pass;

    APPLY_FILE = APPLY_FILE.replace('.csv','.sdf')
    if os.path.exists(APPLY_FILE):
        os.rename(APPLY_FILE, APPLY_FILE+".old")

    print("Relax!");

    """
    # Just before model creation, ensure only one graph_descriptors input
    if architecture_name in ['AttFP', 'AttentiveFP'] and use_descriptors and descs:
        # Build the correct input list
        base_inputs = []
        graph_descriptors_input = None
        for inp in hyper["model"]["config"]["inputs"]:
            name = inp.get("name")
            if name == "node_attributes" and not any(i.get("name") == "node_attributes" for i in base_inputs):
                base_inputs.append(inp)
            elif name == "edge_attributes" and not any(i.get("name") == "edge_attributes" for i in base_inputs):
                base_inputs.append(inp)
            elif name == "edge_indices" and not any(i.get("name") == "edge_indices" for i in base_inputs):
                base_inputs.append(inp)
            elif name == "graph_descriptors" and graph_descriptors_input is None:
                graph_descriptors_input = inp
        if graph_descriptors_input is not None:
            base_inputs.append(graph_descriptors_input)
        hyper["model"]["config"]["inputs"] = base_inputs
    """
