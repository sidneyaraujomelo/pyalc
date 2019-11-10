from argparse import ArgumentParser
import numpy as np


def partial_pivoting(matrix_A, vector_b):
    """ Apply partial pivoting to the system """
    print("=== Partial Pivoting ===")
    for p in range(matrix_A.shape[0]-1):
        print(f"== step {p}:")
        sub_matrix = matrix_A[p:, p:]
        #print(sub_matrix)
        max_col_id = np.argmax(sub_matrix, axis=0)[0]
        if max_col_id != 0:
            print(f"Exchange rows {p+max_col_id} and {p}")
            matrix_A[[p+max_col_id, p]] = matrix_A[[p, p+max_col_id]]
            vector_b[[p+max_col_id, p]] = vector_b[[p, p+max_col_id]]
        print(matrix_A)
        print(vector_b)
    return matrix_A, vector_b


def partial_pivoting_with_scale(matrix_A, vector_b):
    """ Apply partial pivoting with scale to the system """
    print("=== Partial Pivoting with Scale ===")
    scale_factor = np.max(np.abs(matrix_A), axis=1)
    print(f"Scale factors: {scale_factor}")
    for p in range(matrix_A.shape[0]-1):
        print(f"== step {p}:")
        sub_matrix = matrix_A[p:, p:]
        cur_col = sub_matrix[:, 0]
        scaled_col = np.abs(cur_col)/scale_factor[p:]
        #print(cur_col)
        print(f"Scaled columns: {scaled_col}")
        max_col_id = np.argmax(scaled_col)
        if max_col_id != 0:
            print(f"Exchange rows {p+max_col_id} and {p}")
            matrix_A[[p+max_col_id, p]] = matrix_A[[p, p+max_col_id]]
            vector_b[[p+max_col_id, p]] = vector_b[[p, p+max_col_id]]
        print(matrix_A)
        print(vector_b)
    return matrix_A, vector_b


def complete_pivoting(matrix_A, vector_b):
    """ Apply complete pivoting to the system """
    print("=== Complete Pivoting ===")
    reorder = np.arange(vector_b.shape[0])
    print(f"X order: {reorder}")
    for p in range(matrix_A.shape[0]-1):
        print(f"== step {p}:")
        sub_matrix = matrix_A[p:, p:]
        #print(sub_matrix)
        max_val_index = np.unravel_index(np.argmax(np.abs(sub_matrix)), sub_matrix.shape)
        print(max_val_index)
        max_row_id = max_val_index[0]
        if max_row_id != 0:
            print(f"Exchange rows {p+max_row_id} and {p}")
            matrix_A[[p+max_row_id, p]] = matrix_A[[p, p+max_row_id]]
            vector_b[[p+max_row_id, p]] = vector_b[[p, p+max_row_id]]
        max_col_id = max_val_index[1]
        if max_col_id != 0:
            print(f"Exchange columns {p+max_col_id} and {p}")
            matrix_A[:, [p+max_col_id, p]] = matrix_A[:, [p, p+max_col_id]]
            reorder[[p+max_col_id, p]] = reorder[[p, p+max_col_id]]
        print(matrix_A)
        print(vector_b)
        print(reorder)
        #input("...")
    return matrix_A, vector_b, reorder


def regressive_substitution(matrix_A, vector_b):
    """ Apply regressive substitution to a triangular superior matrix_A and vector_b, returns the
    solution x."""
    x = np.zeros(shape=(vector_b.shape[0],))
    i = vector_b.shape[0]-1
    x[i] = vector_b[i]/matrix_A[i][i]
    for k in range(vector_b.shape[0]-2, -1, -1):
        print(f"Partial solution: {x}")
        dem = matrix_A[k][k]
        #print(dem)
        post_x = x[k+1:]
        #print(post_x)
        post_a = matrix_A[k][k+1:]
        #print(post_a)
        num = vector_b[k] - np.sum(post_x*post_a)
        x[k] = num / dem
    print(f"Solution: {x}")
    return x


def gaussian_elimination(matrix_A, vector_b):
    """ Performs gaussian elimination in order solve a linear system """
    print("=== Gaussian Elimination ===")
    for p in range(matrix_A.shape[0]):
        a_pp = matrix_A[p][p]
        print(f"elemento pivot: {a_pp}")
        for c_c in range(p+1, matrix_A.shape[0]):
            print(f"linha corrente: {c_c}")
            m_p = matrix_A[c_c][p]/a_pp
            print(f"m_p: {m_p}")
            matrix_A[c_c] = matrix_A[c_c] - m_p * matrix_A[p]
            vector_b[c_c] = vector_b[c_c] - m_p * vector_b[p]
            print(matrix_A)
            print(vector_b)
            #input("...")
    x = regressive_substitution(matrix_A, vector_b)
    return x, matrix_A, vector_b


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
    print(o_matrix_A)

    if pivot == "partial":
        matrix_A, vector_b = partial_pivoting(o_matrix_A, o_vector_b)
    elif pivot == "partial_scale":
        matrix_A, vector_b = partial_pivoting_with_scale(o_matrix_A, o_vector_b)
    elif pivot == "complete":
        matrix_A, vector_b, reorder = complete_pivoting(o_matrix_A, o_vector_b)

    x, matrix_A, vector_b = gaussian_elimination(matrix_A, vector_b)

    if pivot == "complete":
        print(f"Reordered solution: {x[np.argsort(reorder)]}")

if __name__ == "__main__":
    main()