# coding: utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_data(data_path):
    """
    Read data from a file.

    Parameters
    ----------
    data_path : str
        Data path.

    Returns
    -------
    out : np.ndarray
        A one-dimensional array which is read from data_path.

    """
    with open(data_path, "r", encoding="utf-8") as f:
        return np.array([float(line) for line in f.read().strip().split("\n")])


def get_args():
    """
    Define arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',
                        help='Data path.')
    parser.add_argument('--K', type=int, default=1,
                        help='Number of predictions.')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Whether to display the results.')
    return parser.parse_args()


def GM_1_1(X_0, K, display=True):
    """
    Return the predictions through GM(1, 1).

    Parameters
    ----------
    X_0 : np.ndarray
        Raw data, a one-dimensional array.
    K : int
        number of predictions.
    display : bool
        Whether to display the results.

    Returns
    -------
    X_0_pred[N:] : np.ndarray
        predictions, a one-dimensional array.

    """
    N = X_0.shape[0]
    X_1 = np.cumsum(X_0)

    B = np.ones((N - 1, 2))
    B[:, 0] = [-(X_1[i] + X_1[i + 1]) / 2 for i in range(N - 1)]
    y = X_0[1:]

    assert np.linalg.det(np.matmul(B.transpose(), B)) != 0
    a, u = np.matmul(
        np.matmul(np.linalg.inv(np.matmul(B.transpose(), B)), B.transpose()),
        y
    )

    pred_range = np.arange(N + K)
    X_1_pred = (X_1[0] - u / a) * np.exp(-a * pred_range) + u / a
    X_0_pred = np.insert(np.diff(X_1_pred), 0, X_0[0])
    # print(X_1_pred)
    # print(X_0_pred)

    if display:
        plt.plot(np.arange(N), X_0, 'b-')
        plt.plot(pred_range, X_0_pred, 'r-')
        plt.legend(['Raw data', 'predictions'])
        plt.show()

    return X_0_pred[N:]


if __name__ == '__main__':
    args = get_args()
    X = read_data(args.data_path)
    X_pred = GM_1_1(X, args.K, args.display)
    print(X_pred)
