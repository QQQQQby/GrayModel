# coding: utf-8

import argparse
import numpy as np


def get_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return np.array([float(line) for line in f.read().strip().split("\n")])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to the data.')
    parser.add_argument('--output_path', type=str, default='',
                        help='Path to the output.')
    parser.add_argument('--not_show', action='store_true', default=False,
                        help="Whether not to show predictions.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    X_0 = get_data(args.data_path)
    N = len(X_0)
    X_1 = np.cumsum(X_0)
    B = np.ones((N - 1, 2))
    for i in range(N - 1):
        B[i, 0] = -0.5 * (X_1[i] + X_1[i + 1])
    print(X_0)
    print(X_1)
    print(B)
