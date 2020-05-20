from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

nesting_level = 0


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")


def y_hat_statistic(y_hat, solution):
    y_preds = np.argmax(y_hat, axis=1)
    solution_ohe = np.argmax(solution, axis=1)
    y_wrong_num_per_class = [0] * y_hat.shape[1]
    y_per_class = [0] * y_hat.shape[1]
    y_wrong_per_class_id = defaultdict(list)
    for i, (y_preds_i, y_true_i) in enumerate(zip(y_preds, solution_ohe)):
        y_per_class[y_true_i] += 1
        if y_preds_i != y_true_i:
            y_wrong_num_per_class[y_true_i] += 1
            y_wrong_per_class_id[y_true_i].append(i)

    y_wrong_percent_per_class = np.array([(round(y_wrong_num_per_class[i] / y_per_class[i], 4)
                                                 if y_per_class[i] > 0 else 0.999) for i in range(len(y_per_class))])
    indices = np.argsort(-y_wrong_percent_per_class)

    output_class_info = "class\twrong_num\ttotal_num\tpercent\t\tinstance_id\n"
    for indx in indices:
        output_class_info += str(indx) + "\t" + str(y_wrong_num_per_class[indx]) + "\t\t" + str(y_per_class[indx]) + "\t\t" + \
            str(y_wrong_percent_per_class[indx]) + "\t\t" + str(y_wrong_per_class_id[indx]) + "\n"

    return output_class_info, y_wrong_percent_per_class, y_wrong_per_class_id

