import os
import operator as op
from functools import reduce

import yaml


def get_constants():
    with open(os.path.join(os.getcwd(), '..', 'constants.yaml')) as f:
        constants = yaml.safe_load(f)

    constants['csv_path'] = os.path.join(constants['path'], 'unb_cic_csv')
    constants['parquet_path'] = os.path.join(constants['path'], constants['parquet_name'])
    constants['refined_parquet_path'] = os.path.join(constants['path'], constants['refined_parquet_name'])

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


def get_balanced_weights(df, target):
    counts = df[target].value_counts()
    num_labels = len(counts)

    return 1. / df[target].map(counts).astype('int') / num_labels


def get_features_list(df, constants):
    return [
        col
        for col in df.columns
        if col not in constants['target_columns']
    ]