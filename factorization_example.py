from argparse import ArgumentParser
import numpy as np
from pyalc import *


def main():
    """ Main function, used for testing """
    argsParser = ArgumentParser()
    argsParser.add_argument("A")
    argsParser.add_argument("b")

    args = argsParser.parse_args()

    A_file = args.A
    b_file = args.b

    o_matrix_A = np.loadtxt(A_file)
    o_vector_b = np.loadtxt(b_file)
    print(o_matrix_A)
    print(o_vector_b)

    matrix_A, vector_b, reorder_row = partial_pivoting(o_matrix_A, o_vector_b)
    _, matrix_U, _, matrix_L = gaussian_elimination(matrix_A, vector_b, factorization_only=True)
    print(reorder_row)
    print(matrix_L)
    print(matrix_U)

if __name__ == "__main__":
    main()
