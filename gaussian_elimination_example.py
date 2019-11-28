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

    x, _, _, _, reorder_row, reorder_col = gaussian_elimination(o_matrix_A, o_vector_b, pivoting=pivot)
    print(reorder_col)
    if pivot == "complete":
        print(f"Reordered solution: {x[np.argsort(reorder_col)]}")

if __name__ == "__main__":
    main()
