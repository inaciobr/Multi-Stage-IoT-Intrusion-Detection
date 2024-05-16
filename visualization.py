import math

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


eng_formatter = mpl.ticker.EngFormatter(places=1, sep='')


def eng_formatter_full(value, total):
    return f"{eng_formatter(value)} ({value / total:.1%})"


def get_color_maps(constants):
    general_attack_color = {
        category: mpl.colormaps['tab10'](i)
        for i, category in enumerate(sorted(constants['attack_category']))
    }

    attack_type_color = {
        label: general_attack_color[constants['attack_category_map'][label]]
        for label in constants['attack_category_map']
    }

    network_layer_color = {
        layer: mpl.colormaps['Dark2'](i)
        for i, layer in enumerate(constants['protocol_layer'])
    }

    protocol_color = {
        protocol: network_layer_color[constants['protocol_layer_map'][protocol]]
        for protocol in constants['features']['protocol']
    }

    return {
        'general_attack': general_attack_color,
        'attack_type': attack_type_color,
        'network_layer': network_layer_color,
        'protocol': protocol_color
    }


def plot_frequency_barh(column, feature, title=None, color_map=None, get_totals=None):
    fig, ax = plt.subplots(figsize=(14, 5 + column.nunique() // 8))
    fig.suptitle(title if title else f'{feature} Frequency', fontsize=16)

    counts = column.value_counts(ascending=True)

    if not get_totals:
        get_totals = lambda key: sum(counts)

    barh = counts.plot(
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


def plot_binary_features(df, size=None, title=None, feature='Count', color_map=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    counts = df.sum(axis=0).sort_values(ascending=False)
    counts.plot.bar(
        ax=ax,
        legend=False,
        color=tuple(map(color_map.get, counts.index)) if color_map else None
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


def plot_box_attack_features(df, features, label_column, title, log_features={}, color_map=None):
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


def print_percentage_styled(df):
    return df.style.background_gradient(
        vmin=0.0,
        vmax=1.0,
        cmap='Blues'
    ).format('{:.1%}')


def print_percentage_df(df, label, features):
    percentage_matrix = df.groupby(label, observed=False).agg({
        col: 'mean'
        for col in features
    })

    return print_percentage_styled(percentage_matrix)


def print_incidence_df(df, features):
    counts = pd.DataFrame([
        df.loc[df[feature_x], features].mean()
        for feature_x in features
    ], index=features, columns=features)

    return print_percentage_styled(counts)