import joblib
import operator as op

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
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

    print(f"Training dataset size: {viz.eng_formatter_full(len(X_train), len(X_train) + len(X_test))}.")
    print(f"Test dataset size: {viz.eng_formatter_full(len(X_test), len(X_train) + len(X_test))}.")

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


def test_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    y_train = 
    train_model(model, X_train, y_train)
    y_pred = utils.run_with_time(lambda: model.predict(X_test), title="Predict")

    print(classification_report(y_test, y_pred, labels=list(y_test.cat.categories), digits=3))
    viz.plot_confusion_matrix(y_test, y_pred)

    return model


def print_feature_importance(model, X_train):
    for col, importance in sorted(zip(X_train.columns, model['model'].feature_importances_), key=op.itemgetter(1), reverse=True):
        print(f"{col}: {importance:.3%}")


def get_balanced_weights(target):
    counts = target.value_counts()
    return (1. / counts / len(counts)).to_dict()


def model_dump(model, model_name):
    model_path = os.path.join(_constants['model_path'], f"{model_name}.joblib")
    joblib.dump(model, model_path)


def model_load(model, model_name):
    model_path = os.path.join(_constants['model_path'], f"{model_name}.joblib")
    return joblib.load(model_path)
