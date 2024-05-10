import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_frequency_barh(column, feature, color_map=None):
    fig, ax = plt.subplots(figsize=(14, 5 + column.nunique() // 8))
    fig.suptitle(f'{feature} Frequency', fontsize=16)

    counts = column.value_counts(ascending=True)
    barh = counts.plot(
        kind='barh',
        width=0.8,
        ax=ax,
        color=tuple(map(color_map.get, counts.index)) if color_map else None
    )

    h_displacement = 0.005 * max(counts)
    y_displacement = -0.005 * len(counts)

    for index, value in enumerate(counts):
        plt.text(
            x=value + h_displacement,
            y=index + y_displacement,
            s=full_eng_formatter(value, sum(counts))
        )

    ax.set_xlabel('Frequency')
    ax.set_ylabel(feature)
    ax.set_xlim(right=1.12 * max(counts))
    ax.xaxis.set_major_formatter(eng_formatter)

    plt.tight_layout()


def plot_binary_features(df, size, title, color_map=None):
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
    ax.set_ylabel('Count')
    ax.set_ylim(top=1.05 * ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(eng_formatter)

    for p in ax.patches:
        ax.annotate(
            full_eng_formatter(p.get_height(), size),
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