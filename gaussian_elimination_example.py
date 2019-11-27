from argparse import ArgumentParser
import numpy as np
from pyalc import *


def main():
    """ Main function, used for testing """
    argsParser = ArgumentParser()
    argsParser.add_argument("A")
    argsParser.add_argument("b")
    argsParser.add_argument("-p", type=str)
    
    args = argsParser.parse_args()

    A_file = args.A
    b_file = args.b
    pivot = args.p if args.p else None

    o_matrix_A = np.loadtxt(A_file)
    o_vector_b = np.loadtxt(b_file)
    print(o_matrix_A)
    print(o_vector_b)

    if pivot == "partial":
        matrix_A, vector_b, reorder_row = partial_pivoting(o_matrix_A, o_vector_b)
    elif pivot == "partial_scale":
        matrix_A, vector_b, reorder_row = partial_pivoting_with_scale(o_matrix_A, o_vector_b)
    elif pivot == "complete":
        matrix_A, vector_b, reorder_row, reorder_col = complete_pivoting(o_matrix_A, o_vector_b)
    else:
        matrix_A, vector_b = o_matrix_A, o_vector_b

    x, _, _, _ = gaussian_elimination(matrix_A, vector_b)
    #print(reorder_row)
    if pivot == "complete":
        print(f"Reordered solution: {x[np.argsort(reorder_col)]}")

if __name__ == "__main__":
    main()
