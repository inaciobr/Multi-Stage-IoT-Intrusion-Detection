import os
import operator as op
from functools import reduce

import yaml


def get_constants():
    with open(os.path.join(os.getcwd(), '..', 'constants.yaml')) as f:
        constants = yaml.safe_load(f)

    constants['parquet_path'] = os.path.join(constants['path'], constants['parquet_name'])
    constants['csv_path'] = os.path.join(constants['path'], 'unb_cic_csv')
    constants['features']['protocol'] = reduce(op.concat, constants['protocol_layer'].values())

    constants['attack_category_map'] = {
        col: attack_category
        for attack_category, column_list in constants['attack_category'].items()
        for col in column_list
    }

    constants['protocol_layer_map'] = {
        protocol: layer
        for layer, protocol_list in constants['protocol_layer'].items()
        for protocol in protocol_list
    }

    return constants
