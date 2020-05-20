import numpy as np
from tools import timeit, log
from sklearn.neighbors import KNeighborsClassifier


@timeit
def knn_test_fix(y_test_hat, y_train_hat, y_train_true):
    # fix 5% preds
    threshold = sorted(np.max(y_test_hat, axis=1))[int(len(y_test_hat) * 0.05)]
#     threshold = 0.5
    log(f"y hat max < {threshold} will be fix")
    y_test_hat_fix = _adjust_preds(
        y_test_hat,
        y_train_hat, y_train_true,
        threshold=threshold
    )
    return y_test_hat_fix


def _adjust_preds(y_test_hat, y_train_hat, y_train_true, threshold=0.5, filter_train=False, n_neighbors=5):
    # filter high probability train x
    y_test_hat = np.array(y_test_hat)
    if filter_train:
        max_train_pred = np.max(y_train_hat, axis=1)
        filtered_train_index = np.where(max_train_pred > threshold)[0]
        log("threshold {}; total train samples {}, remove train samples {}"
            .format(threshold, len(y_train_hat), len(y_train_hat) - len(filtered_train_index)))
        y_train_hat, y_train_true = y_train_hat[filtered_train_index], y_train_true[filtered_train_index]

    # get low high probability test preds
    max_pred = np.max(y_test_hat, axis=1)
    _preds_discover(max_pred)
    low_preds_index = np.where(max_pred <= threshold)[0]
    log("threshold {}; total test preds {}, adjust test preds {}"
        .format(threshold, len(y_test_hat), len(low_preds_index)))
    y_test_fix = y_test_hat[low_preds_index, :]

    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                 metric='euclidean',
                                 n_jobs=-1)
    model.fit(y_train_hat, y_train_true)

    log(f"y test hat: {y_test_hat.shape}, index shape: {low_preds_index.shape}, test x shape: {y_test_fix.shape}")
    log(f"y train hat: {y_train_hat.shape}, y train true: {y_train_true.shape}")
    y_test_fix = model.predict_proba(y_test_fix)
    # 类别不全, 导致部分类别没有数据, 因此shape不同
    meaningful_fix = 0
    for class_i in range(len(y_test_fix)):
        logits = np.asarray(y_test_fix[class_i])
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        else:
            logits = 1 - logits
        meaningful_index = np.where(logits > 1 / n_neighbors)[0]
        meaningful_fix += len(meaningful_index)
        y_test_hat[low_preds_index[meaningful_index], class_i] = logits[meaningful_index].reshape((-1))
    log(f"totally fix {meaningful_fix} preds")
    return y_test_hat


def _preds_discover(preds):
    sorted_preds = sorted(preds)
    log("5% preds {}; 10% preds {}; 30% preds {}; 50% preds {}; 95% preds {}"
        .format(sorted_preds[int(len(preds) * 0.05)],
                sorted_preds[int(len(preds) * 0.1)],
                sorted_preds[int(len(preds) * 0.3)],
                sorted_preds[int(len(preds) * 0.5)],
                sorted_preds[int(len(preds) * 0.95)]))