import os
import joblib
import operator as op

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import visualization as viz
import utils

_constants = utils.get_constants()


def get_train_test_split(
    df,
    features=None,
    target='general_label',
    sample_size=None,
    train_frac=0.80
):
    if not features:
        features = utils.get_features_list(df)

    X, y = df[features], df[target] if isinstance(target, str) else target

    if sample_size:
        y = y.sample(sample_size)
        X = X.loc[y.index]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_frac,
        random_state=_constants['seed']
    )

    len_df = len(X_train) + len(X_test)
    print(f"Training dataset size: {viz.eng_formatter_full(len(X_train), len_df)}.")
    print(f"Test dataset size: {viz.eng_formatter_full(len(X_test), len_df)}.")

    return X_train, X_test, y_train, y_test


def train_model(
    model,
    X_train,
    y_train,
    save_name=None
):
    utils.run_with_time(lambda: model.fit(X_train, y_train), title="Model fit")

    if save_name:
        model_dump(model, save_name)

    return model


def show_metrics(cm, labels):
    viz.plot_confusion_matrix(cm, labels)
    metrics_report = classification_report_from_confusion_matrix(cm, labels)
    print(metrics_report)


def evaluate_model(
    model,
    X_test,
    y_test
):
    y_pred = utils.run_with_time(lambda: model.predict(X_test), title="Predict")
    cm = confusion_matrix(y_test, y_pred)
    show_metrics(cm, labels=y_test.cat.categories)


def print_feature_importance(model, X_train):
    for col, importance in sorted(zip(X_train.columns, model['model'].feature_importances_), key=op.itemgetter(1), reverse=True):
        print(f"{col}: {importance:.3%}")


def get_classification_metrics(
    confusion_matrix,
    average=None
):
    df_len = len(confusion_matrix)
    diagonal = np.arange(df_len), np.arange(df_len)
    TP = confusion_matrix[diagonal]

    true_support = confusion_matrix.sum(axis=1)
    predicted_support = confusion_matrix.sum(axis=0)

    if average == 'weighted':
        TP = TP.sum()
        true_support = true_support.sum()
        predicted_support = predicted_support.sum()

    precision = TP / predicted_support
    recall = TP / true_support
    f1_score = 2 * precision * recall / (precision + recall)

    if average == 'macro':
        precision = precision.mean()
        recall = recall.mean()
        f1_score = f1_score.mean()

    if average is not None:
        true_support = true_support.sum()
        TP = TP.sum()
        accuracy = TP / true_support
    else:
        accuracy = np.zeros(df_len)

    return (precision, recall, f1_score, true_support, accuracy)


def classification_report_from_confusion_matrix(
    confusion_matrix,
    labels,
    fmt_metrics='{:10.3f}',
    fmt_support='{:10}'
):
    classification_metrics = get_classification_metrics(confusion_matrix)

    average_metrics = {
        'macro avg': get_classification_metrics(confusion_matrix, average='macro'),
        'weighted avg': get_classification_metrics(confusion_matrix, average='weighted'),
    }

    accuracy = average_metrics['weighted avg'][:-3:-1]

    metrics_header = ['precision', 'recall', 'f1-score', 'support']
    len_header = len(metrics_header)
    width_title = max(map(len, metrics_header)) + 1
    width = max(*map(len, list(labels) + list(average_metrics)), len(fmt_metrics.format(0)))

    head_fmt = ' ' + width*' ' + len_header*f"{{:>{width_title}}}" + "\n\n"
    row_fmt = f"{{:>{width}}} " + 3*fmt_metrics + fmt_support + '\n'
    acc_fmt = f"{{:>{width}}} " + 2*width_title*' ' +  fmt_metrics + fmt_support + '\n'

    head = head_fmt.format(*metrics_header)

    metrics_by_label = ''.join(
        row_fmt.format(*row)
        for row in zip(labels, *classification_metrics)
    ) + '\n'

    average_metrics = ''.join(
        row_fmt.format(label, *metrics)
        for label, metrics in average_metrics.items()
    )

    acc = acc_fmt.format('accuracy', *accuracy)

    return head + metrics_by_label + acc + average_metrics


def get_balanced_weights(target):
    counts = target.value_counts()
    return (1. / counts / len(counts)).to_dict()


def model_dump(model, model_name):
    model_path = os.path.join(_constants['model_path'], f"{model_name}.joblib")
    joblib.dump(model, model_path)


def model_load(model_name):
    model_path = os.path.join(_constants['model_path'], f"{model_name}.joblib")
    return joblib.load(model_path)
