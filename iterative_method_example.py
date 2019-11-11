from argparse import ArgumentParser
import numpy as np
from pyalc import *


def main():
    """ Main function, used for testing """
    argsParser = ArgumentParser()
    argsParser.add_argument("A")
    argsParser.add_argument("b")
    argsParser.add_argument("x")
    argsParser.add_argument("--method", type=str, default="gauss_jacobi")

    args = argsParser.parse_args()

    A_file = args.A
    b_file = args.b
    x_file = args.x
    method = args.method

    o_matrix_A = np.loadtxt(A_file)
    o_vector_b = np.loadtxt(b_file)
    o_vector_x = np.loadtxt(x_file)
    print(o_matrix_A)
    print(o_vector_b)
    print(o_vector_x)

    if method == "gauss_jacobi":
        solution = gauss_jacobi(o_matrix_A, o_vector_b, o_vector_x)
    elif method == "gauss_seidel":
        solution = gauss_seidel(o_matrix_A, o_vector_b, o_vector_x)
    elif method == "conjugate_gradient":
        solution = conjugate_gradient(o_matrix_A, o_vector_b, o_vector_x)
    else:
        print(f"Sorry. The method {method} is not implemented!")
        return
    
    print(f"Final solution with {method}: {solution}")

if __name__ == "__main__":
    main()

