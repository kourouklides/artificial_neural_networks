"""

Utility to download a dataset

    Author: Ioannis Kourouklides, www.kourouklides.com
    License: https://github.com/kourouklides/artificial_neural_networks/blob/master/LICENSE


"""
# %%
# IMPORTS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# standard library imports
import json
import yaml

# %%


def none_or_int(input_arg):
    """
    Utility to return None or int value
    """
    if input_arg == 'None':
        value = None
    else:
        value = int(input_arg)

    return value


# %%


def none_or_float(input_arg):
    """
    Utility to return None or float value
    """
    if input_arg == 'None':
        value = None
    else:
        value = float(input_arg)

    return value


# %%


def save_model(model, models_path, model_name, weights_path, model_path, file_suffix, test_dict,
               args):
    """
    Utility to save the architecture, the weights and the whole model
    """
    architecture_path = models_path + model_name + '_architecture'

    last_suffix = file_suffix.format(
        epoch=args.n_epochs, val_acc=test_dict['val_acc'], val_loss=test_dict['val_loss'])

    if args.save_architecture:
        # Save only the archtitecture (as a JSON file)
        json_string = model.to_json()
        json.dump(json.loads(json_string), open(architecture_path + '.json', "w"))

        # Save only the archtitecture (as a YAML file)
        yaml_string = model.to_yaml()
        yaml.dump(yaml.load(yaml_string), open(architecture_path + '.yml', "w"))

    # Save only the weights (as an HDF5 file)
    if args.save_last_weights:
        model.save_weights(weights_path + last_suffix + '.h5')

    # Save the whole model (as an HDF5 file)
    if args.save_last_model:
        model.save(model_path + last_suffix + '.h5')
