from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pandas as pd

def train_svm(kernels=None, labels=None):
    if kernels is None:
        trn_k, trn_y = load_svmlight_file('dns_data_kernel/trn_kernel_mat.svmlight')
        val_k, val_y = load_svmlight_file('dns_data_kernel/val_kernel_mat.svmlight')
        tst_k, tst_y = load_svmlight_file('dns_data_kernel/tst_kernel_mat.svmlight')

        trn_k = trn_k.todense()
        val_k = val_k.todense()
        tst_k = tst_k.todense()
    else:
        trn_k, trn_y = kernels[0], labels[0]
        val_k, val_y = kernels[1], labels[1]
        tst_k, tst_y = kernels[2], labels[2]

    pred = dict()

    C = [0.01, 0.1, 1, 10, 100]

    val_errs = []
    for c in C:
        m = svm.SVC(kernel='precomputed', C=c)
        m.fit(trn_k, trn_y)

        trn_label = m.predict(trn_k)
        val_label = m.predict(val_k)

        trn_err = zero_one_loss(trn_label, trn_y)
        val_err = zero_one_loss(val_label, val_y)

        pred[c] = [trn_err, val_err, sum(m.n_support_)]
        val_errs.append(val_err)

    opt_c = C[val_errs.index(min(val_errs))]
    m = svm.SVC(kernel='precomputed', C=opt_c)
    m.fit(trn_k, trn_y)

    tst_label = m.predict(tst_k)

    tst_err = zero_one_loss(tst_label, tst_y)

    print("Test Error: {0:.2%}".format(tst_err))

    return pred


def plot_svm(pred):
    c_values = np.fromiter(pred.keys(), dtype=float)
    trn_errors = np.array([value[0] for _, value in pred.items()])
    val_errors = np.array([value[1] for _, value in pred.items()])
    nsv = np.array([value[2] for _, value in pred.items()])

    fig, (ax, tabax) = plt.subplots(nrows=2)
    # graph
    ax.plot(c_values, trn_errors, marker='o')
    ax.plot(c_values, val_errors, marker='o')
    ax.legend(['Training Error', 'Validation Error'])
    ax.set_ylabel('Error')
    ax.set_ylim([0, 1])
    ax.set_xlabel('C')
    ax.axis('auto')
    ax.set_xscale('log')

    # table
    tabax.axis('off')
    the_table = tabax.table(
        cellText=np.column_stack((np.round(trn_errors, 4), np.round(val_errors, 4), nsv.astype(int))),
        rowLabels=["C=" + str(elm) for elm in c_values],
        colLabels=('Training Error', 'Validation Error', 'Number of Support Vectors'),
        loc='center'
    )
    # Adjust layout to make room for the table:
    plt.subplots_adjust(bottom=0.05)
    plt.savefig('plot.png')

# Assignment 3
def compute_small_kernels():
    x = ["google.com", "facebook.com", "atqgkfauhuaufm.com", "vopydum.com"]
    k_norm = np.zeros(shape=[4, 4])
    k = np.zeros(shape=[4,4])
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            k[i, j] = subseq_kernel(x[i], x[j], lambd=0.4, q=3)
            k_norm[i, j] = subseq_kernel(x[i], x[j], lambd=0.4, q=3) / (sqrt(subseq_kernel(x[i], x[i], lambd=0.4, q=3)) * sqrt(subseq_kernel(x[j], x[j], lambd=0.4, q=3)))

    return k, k_norm

# function for computing kernels by myself
def compute_kernels():
    with open('dns_data/trn_legit.txt') as trn1, \
            open('dns_data/trn_malware.txt') as trn2, \
            open('dns_data/val_legit.txt') as val1, \
            open('dns_data/val_malware.txt') as val2, \
            open('dns_data/tst_legit.txt') as tst1, \
            open('dns_data/tst_malware.txt') as tst2:
        trn_legit = trn1.read().splitlines()
        trn_malware = trn2.read().splitlines()
        val_legit = val1.read().splitlines()
        val_malware = val2.read().splitlines()
        tst_legit = tst1.read().splitlines()
        tst_malware = tst2.read().splitlines()

        trn1.close(), trn2.close(), val1.close(), val2.close(), tst1.close(), tst2.close()

        x_trn = trn_legit + trn_malware
        y_trn = [1.] * len(trn_legit) + [-1.] * len(trn_malware)
        y_trn = np.array(y_trn)

        x_val = val_legit + val_malware
        y_val = [1.] * len(val_legit) + [-1.] * len(val_malware)
        y_val = np.array(y_val)

        x_tst = tst_legit + tst_malware
        y_tst = [1.] * len(tst_legit) + [-1.] * len(tst_malware)
        y_tst = np.array(y_tst)

        # 1000x1000 dot products between training examples
        k_trn = np.zeros(shape=[len(x_trn), len(x_trn)])
        for i in range(0, len(x_val)):
            print(i)
            for j in range(0, len(x_trn)):
                k_trn[i, j] = subseq_kernel(x_trn[i], x_trn[j]) / (sqrt(subseq_kernel(x_trn[i], x_trn[i])) * sqrt(subseq_kernel(x_trn[j], x_trn[j])))

        # 500x1000 dot products between validation and training examples.
        k_val = np.zeros(shape=[len(x_val), len(x_trn)])
        for i in range(0, len(x_val)):
            print(i)
            for j in range(0, len(x_trn)):
                k_val[i, j] = subseq_kernel(x_val[i], x_trn[j]) / (sqrt(subseq_kernel(x_val[i], x_val[i])) * sqrt(subseq_kernel(x_trn[j], x_trn[j])))

        # 2000x1000 dot products between test and training examples.
        k_tst = np.zeros(shape=[len(x_tst), len(x_trn)])
        for i in range(0, len(x_tst)):
            print(i)
            for j in range(0, len(x_trn)):
                k_tst[i, j] = subseq_kernel(x_tst[i], x_trn[j]) / (sqrt(subseq_kernel(x_tst[i], x_tst[i])) * sqrt(subseq_kernel(x_trn[j], x_trn[j])))

        return (k_trn, k_val, k_tst), (y_trn, y_val, y_tst)


def subseq_kernel(str1, str2, q, lambd):
    if min(len(str1), len(str2)) < q:
        return 0
    else:
        return subseq_kernel(str1[:-1], str2, q, lambd) \
               + sum([subseq_kernel2(str1[:-1], str2[:j], q - 1, lambd) * lambd**2 for j in findOccurrences(str2, str1[-1:])])


def subseq_kernel2(str1, str2, i, lambd):
    if i == 0:
        return 1
    elif min(len(str1), len(str2)) < i:
        return 0
    else:
        return lambd * subseq_kernel2(str1[:-1], str2, i, lambd) \
               + sum([subseq_kernel2(str1[:-1], str2[:j], i - 1, lambd) * lambd ** (len(str2) - j + 1) for j in findOccurrences(str2, str1[-1:])])


def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def main():
    # kernels, labels = compute_kernels()
    pred = train_svm()
    plot_svm(pred)

    k, k_norm = compute_small_kernels()
    print("K:")
    print(pd.DataFrame(k))
    print("K norm:")
    print(pd.DataFrame(k_norm))


if __name__ == '__main__':
    main()
