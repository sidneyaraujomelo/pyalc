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
    argsParser.add_argument("--pivoting", type=str, default=None)
    argsParser.add_argument("--C")

    args = argsParser.parse_args()

    A_file = args.A
    b_file = args.b
    x_file = args.x
    method = args.method
    pivoting = args.pivoting
    if method == "conjugate_gradient":
        C_file = args.C
        o_matrix_C = np.loadtxt(C_file)

    o_matrix_A = np.loadtxt(A_file)
    o_vector_b = np.loadtxt(b_file)
    o_vector_x = np.loadtxt(x_file)
    print(o_matrix_A)
    print(o_vector_b)
    print(o_vector_x)
    if method == "conjugate_gradient":
        print(o_matrix_C)

    #matrix = np.array([[4.0, 5.0, 9.0], [1.0, 3.0, 2.0], [9.0, 2.0, 3.0]])
    #b = np.array([1, 2, 3])
    #matrix = np.array([[4, 6, 9, 12], [0, 0, 50, 13], [5, 32, 4, 31], [13, 0, 14, 5]])
    #b = np.array([40, 20, 10, 5])
    # matrix = np.array([[4.0, 5.0], [1.0, 3.0]])
    # b = np.array([4, 5])
    # matrix = np.array([[1.,-1.,1.,-1.],[1.,0.,0.,0.],[1.,1.,1.,1.],[1.,2.,4.,8.]])
    # b =  np.array([14., 4. , 2. , 2.])
    # matrix = np.array(
    #     [
    #         [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    #         [1.00, 0.63, 0.39, 0.25, 0.16, 0.10],
    #         [1.00, 1.26, 1.58, 1.98, 2.49, 3.13],
    #         [1.00, 1.88, 3.55, 6.70, 12.62, 23.80],
    #         [1.00, 2.51, 6.32, 15.88, 39.90, 100.28],
    #         [1.00, 3.14, 9.87, 31.01, 97.41, 306.02],
    #     ]
    # )
    # b = np.array([-0.01, 0.61, 0.91, 0.99, 0.60, 0.02])
    # matrix = np.array([[2.0,1.0],[5.0,7.0]])
    # b = np.array([11.0,13.0])

    #o_matrix_A = matrix
    #o_vector_b = b
    #o_vector_x = np.zeros_like(b)
    #print(o_vector_x)
    
    stop_step = 0
    reorder_row = None
    reorder_col = None
    scale_factor = None
    matrix_A = o_matrix_A
    vector_b = o_vector_b

    for p in range(matrix_A.shape[0]):
        if pivoting == "partial":
            matrix_A, vector_b, stop_step, reorder_row, _ = partial_pivoting(
                matrix_A, vector_b, p, stop_step=stop_step,
                reorder_row=reorder_row)
        elif pivoting == "partial_with_scale":
            print(scale_factor)
            if not isinstance(scale_factor, np.ndarray):
                scale_factor = np.max(np.abs(matrix_A), axis=1)
            matrix_A, vector_b, stop_step, reorder_row, _ = partial_pivoting_with_scale(
                matrix_A, vector_b, p, scale_factor=scale_factor,
                stop_step=stop_step, reorder_row=reorder_row)
            scale_factor = scale_factor[np.argsort(reorder_row)]
        elif pivoting == "complete":
            matrix_A, vector_b, stop_step, reorder_row, reorder_col = complete_pivoting(
                matrix_A, vector_b, p, stop_step=stop_step,
                reorder_row=reorder_row, reorder_col=reorder_col)
            
    stop_step = 0
    if method == "gauss_jacobi":
        solution = gauss_jacobi(matrix_A, vector_b, o_vector_x)
    elif method == "gauss_seidel":
        unsorted_solution = gauss_seidel(matrix_A, vector_b, o_vector_x)
        solution = unsorted_solution[np.argsort(reorder_col)]
    elif method == "conjugate_gradient":
        solution = conjugate_gradient(matrix_A, vector_b, o_vector_x, o_matrix_C)
    else:
        print(f"Sorry. The method {method} is not implemented!")
        return
    
    print(f"Final solution with {method}: {solution}")

if __name__ == "__main__":
    main()

