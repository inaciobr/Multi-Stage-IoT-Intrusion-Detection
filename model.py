def get_class_balanced_weights(target):
    counts = target.value_counts()
    return (1. / counts / len(counts)).to_dict()


def get_balanced_weights(target):
    class_balanced_weights = get_class_balanced_weights(target)
    return target.map(class_balanced_weights).astype('double')
