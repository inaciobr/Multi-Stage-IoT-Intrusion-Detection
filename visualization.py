import math

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay

import utils

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.2f}'.format)


_constants = utils.get_constants()
eng_formatter = mpl.ticker.EngFormatter(places=1, sep='')

_general_attack_color = {
    category: mpl.colormaps['tab10'](i)
    for i, category in enumerate(sorted(_constants['attack_category']))
}

_attack_type_color = {
    label: _general_attack_color[_constants['attack_category_map'][label]]
    for label in _constants['attack_category_map']
}

_protocol_layer_color = {
    layer: mpl.colormaps['Dark2'](i)
    for i, layer in enumerate(_constants['protocol_layer'])
}

_protocol_color = {
    protocol: _protocol_layer_color[_constants['protocol_layer_map'][protocol]]
    for protocol in _constants['features']['protocol']
}

color_map = {
    'general_attack': _general_attack_color,
    'attack_type': _attack_type_color,
    'protocol_layer': _protocol_layer_color,
    'protocol': _protocol_color
}


def eng_formatter_full(value, total):
    return f"{eng_formatter(value)} ({value / total:.1%})"

def plot_frequency_barh(
    column,
    feature,
    title=None,
    color_map=None,
    get_totals=None,
    small=False
):
    fig, ax = plt.subplots(figsize=(7 if small else 14, 5 + column.nunique() // 8))
    fig.suptitle(title if title else f'{feature} Frequency', fontsize=16)

    counts = column.value_counts(ascending=True)

    if not get_totals:
        get_totals = lambda key: sum(counts)

    counts.plot(
        kind='barh',
        width=0.8,
        ax=ax,
        color=tuple(map(color_map.get, counts.index)) if color_map else None
    )

    h_displacement = 0.005 * max(counts)
    y_displacement = -0.005 * len(counts)

    
    for index, (key, value) in enumerate(counts.items()):
        plt.text(
            x=value + h_displacement,
            y=index + y_displacement,
            s=eng_formatter_full(value, get_totals(key))
        )

    ax.set_xlabel('Frequency')
    ax.set_ylabel(feature)
    ax.set_xlim(right=1.12 * max(counts))
    ax.xaxis.set_major_formatter(eng_formatter)

    plt.tight_layout()


def plot_binary_features(
    df,
    size=None,
    title=None,
    feature='Count',
    color_map=None
):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    counts = df.sum(axis=0).sort_values(ascending=False)
    counts.plot.bar(
        ax=ax,
        legend=False,
        color=counts.index.map(color_map) if color_map else None
    )

    ax.set_xlabel(None)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel(feature)
    ax.set_ylim(top=1.05 * ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(eng_formatter)

    for p in ax.patches:
        ax.annotate(
            eng_formatter_full(p.get_height(), size) if size else eng_formatter(p.get_height()),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='center',
            fontsize=8,
            xytext=(2, 8),
            textcoords='offset points'
        )
    
    plt.tight_layout()


def plot_box_attack_features(
    df,
    features,
    label_column,
    title,
    log_features={},
    color_map=None
):
    num_plots = len(features)
    ncols = 2
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4*nrows))
    fig.suptitle(title, fontsize=16)

    for ax, feature in zip(axes.flatten(), features):
        ax.set_title(f'{feature}')
        sns.boxplot(
            data=df,
            x=label_column,
            hue=label_column,
            y=feature,
            palette=color_map,
            ax=ax
        )

        if df[feature].max() > 100:
            ax.yaxis.set_major_formatter(eng_formatter)

        ax.set_xlabel(None)
        ax.set_ylabel(feature)

        if feature in log_features:
            ax.set_yscale('log')
            ax.set_ylabel(ax.get_ylabel() + ' (Log Scale)')

    # Remove empty subplots
    for ax in axes.flatten()[num_plots:nrows*ncols]:
        fig.delaxes(ax)

    plt.tight_layout(pad=1.8)


def plot_confusion_matrix(cm, labels=None):
    n_labels = len(cm)
    fig, ax = plt.subplots(figsize=(5*(1 + n_labels // 5), 3*(1 + n_labels // 5)))

    cmp = ConfusionMatrixDisplay(cm, display_labels=labels)

    plt.xticks(rotation=90)
    fig.tight_layout()

    cmp.plot(xticks_rotation='vertical', cmap='Blues', values_format='', ax=ax)


def print_percentage_styled(df):
    return df.style.background_gradient(
        vmin=0.0,
        vmax=1.0,
        cmap='Blues'
    ).format('{:.1%}')


def print_percentage_df(df, label, features):
    percentage_matrix = df.groupby(label, observed=False).agg({
        col: lambda x: x.astype('bool').mean()
        for col in features
    })

    return print_percentage_styled(percentage_matrix)


def print_incidence_df(df, features):
    counts = pd.DataFrame([
        df.loc[df[feature_x], features].mean()
        for feature_x in features
    ], index=features, columns=features)

    return print_percentage_styled(counts)
