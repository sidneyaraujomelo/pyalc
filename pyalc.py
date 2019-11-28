import numpy as np

def partial_pivoting(matrix_A, vector_b, p, stop_step=0, reorder_row=None, reorder_col=None):
    """ Apply partial pivoting to the system """
    print("=== Partial Pivoting ===")
    if not isinstance(reorder_row, np.ndarray):
        reorder_row = np.arange(vector_b.shape[0])
    print(f"== step {p}:")
    sub_matrix = matrix_A[p:, p:]
    #print(sub_matrix)
    max_col_id = np.argmax(np.abs(sub_matrix), axis=0)[0]
    if max_col_id != 0:
        print(f"Exchange rows {p+max_col_id} and {p}")
        matrix_A[[p+max_col_id, p]] = matrix_A[[p, p+max_col_id]]
        vector_b[[p+max_col_id, p]] = vector_b[[p, p+max_col_id]]
        reorder_row[[p+max_col_id, p]] = reorder_row[[p, p+max_col_id]]
    print(matrix_A)
    print(vector_b)
    print(stop_step)
    if p == stop_step:
        stop_step = int(input("input new stop step "))
    return matrix_A, vector_b, stop_step, reorder_row, None


def partial_pivoting_with_scale(matrix_A, vector_b, p, scale_factor=1, stop_step=0, reorder_row=None, reorder_col=None):
    """ Apply partial pivoting with scale to the system """
    print("=== Partial Pivoting with Scale ===")
    if not isinstance(reorder_row, np.ndarray):
        reorder_row = np.arange(vector_b.shape[0])
    print(f"Scale factors: {scale_factor}")
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
        reorder_row[[p+max_col_id, p]] = reorder_row[[p, p+max_col_id]]
    print(matrix_A)
    print(vector_b)
    if p == stop_step:
        stop_step = int(input("input new stop step "))
    return matrix_A, vector_b, stop_step, reorder_row, None


def complete_pivoting(matrix_A, vector_b, p, stop_step=0, reorder_row=None, reorder_col=None):
    """ Apply complete pivoting to the system """
    print("=== Complete Pivoting ===")
    if not isinstance(reorder_row, np.ndarray):
        reorder_row = np.arange(vector_b.shape[0])
    if not isinstance(reorder_col, np.ndarray):
        reorder_col = np.arange(vector_b.shape[0])
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
        reorder_row[[p+max_row_id, p]] = reorder_row[[p, p+max_row_id]]
    max_col_id = max_val_index[1]
    if max_col_id != 0:
        print(f"Exchange columns {p+max_col_id} and {p}")
        matrix_A[:, [p+max_col_id, p]] = matrix_A[:, [p, p+max_col_id]]
        reorder_col[[p+max_col_id, p]] = reorder_col[[p, p+max_col_id]]
    print(matrix_A)
    print(vector_b)
    print(reorder_col)
    if p == stop_step:
        stop_step = int(input("input new stop step "))
    #input("...")
    return matrix_A, vector_b, stop_step, reorder_row, reorder_col


def regressive_substitution(matrix_A, vector_b):
    """ Apply regressive substitution to a triangular superior matrix_A and vector_b, returns the
    solution x."""
    x = np.zeros(shape=(vector_b.shape[0],))
    i = vector_b.shape[0]-1
    x[i] = vector_b[i]/matrix_A[i][i]
    for k in range(vector_b.shape[0]-2, -1, -1):
        print(f"Partial solution: {x}")
        dem = matrix_A[k][k]
        # print(dem)
        post_x = x[k+1:]
        # print(post_x)
        post_a = matrix_A[k][k+1:]
        # print(post_a)
        num = vector_b[k] - np.sum(post_x*post_a)
        x[k] = num / dem
    print(f"Solution: {x}")
    return x


def gaussian_elimination(matrix_A, vector_b, factorization_only=False, pivoting=None):
    """ Performs gaussian elimination in order solve a linear system """
    print("=== Gaussian Elimination ===")
    matrix_L = np.zeros_like(matrix_A)
    stop_step = 0
    reorder_row = None
    reorder_col = None
    scale_factor = None
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
        a_pp = matrix_A[p][p]
        matrix_L[p][p] = 1
        print(f"Posição do pivot: {p}, Elemento pivot: {a_pp}")
        for c_c in range(p+1, matrix_A.shape[0]):
            print(f"linha corrente: {c_c}")
            m_p = matrix_A[c_c][p]/a_pp
            print(f"m_p: {m_p}")
            matrix_A[c_c] = matrix_A[c_c] - m_p * matrix_A[p]
            vector_b[c_c] = vector_b[c_c] - m_p * vector_b[p]
            matrix_L[c_c][p] = m_p
            print(matrix_A)
            print(vector_b)
            #input("...")
        if p == stop_step:
            stop_step = int(input("input new stop step "))
    if factorization_only:
        return None, matrix_A, None, matrix_L
    x = regressive_substitution(matrix_A, vector_b)
    return x, matrix_A, vector_b, matrix_L, reorder_row, reorder_col


def gauss_jacobi(matrix_A, vector_b, vector_x, max_k=500, e=1e-10):
    new_x = np.copy(vector_x)
    last_result = 0
    tol = 0
    stop_step = 0
    for k in range(max_k):
        print(f"Iteraction {k}")
        for i in range(new_x.shape[0]):
            mask = np.ones_like(new_x, dtype=bool)
            mask[i] = 0
            #print(mask)
            slice_Ai = matrix_A[i][mask]
            slice_x = vector_x[mask]
            a_x_term = np.sum(slice_Ai*slice_x)
            new_x[i] = (vector_b[i]-a_x_term)/matrix_A[i][i]
        print(f"Step {k}, Partial solution: {new_x}")
        if k == stop_step:
            stop_step = int(input("input new stop step "))
        dif = new_x - vector_x
        error = np.linalg.norm(dif, np.inf)
        print(f"error: {error}")
        if error < e:
            return new_x
        else:
            vector_x = np.copy(new_x)


def gauss_seidel(matrix_A, vector_b, vector_x, max_k=500, e=1e-10):
    stop_step = 0
    new_x = np.copy(vector_x)
    last_result = 0
    tol = 0
    for k in range(max_k):
        print(f"Iteraction {k}")
        for i in range(new_x.shape[0]):
            mask = np.ones_like(new_x, dtype=bool)
            mask[i] = 0
            #print(mask)
            slice_Ai = matrix_A[i][mask]
            slice_x = np.hstack((new_x[:i], vector_x[i+1:]))
            a_x_term = np.sum(slice_Ai*slice_x)
            new_x[i] = (vector_b[i]-a_x_term)/matrix_A[i][i]
        print(f"Step {k}, Partial solution: {new_x}")
        if k == stop_step:
            stop_step = int(input("input new stop step "))
        dif = new_x - vector_x
        error = np.linalg.norm(dif, np.inf)
        print(f"error: {error}")
        if error < e:
            return new_x
        else:
            vector_x = np.copy(new_x)


def conjugate_gradient(matrix_A, vector_b, vector_x, matrix_C, max_k=500, e=1e-10):
    stop_step = 0
    x = np.copy(vector_x)
    #residuo no passo k
    r_k = vector_b - matrix_A.dot(x)
    #pre-condicionador z no passo k
    z_k = np.linalg.inv(matrix_C).dot(r_k)
    v = np.copy(z_k)
    for k in range(max_k):
        #termo de cima da equação do alpha
        rz_k = r_k.transpose().dot(z_k)
        #alpha
        t = rz_k / v.transpose().dot(matrix_A).dot(v)
        #input(a)
        #atualiza a solução
        x = x+t*v
        #input(x)
        #calcula residuo do passo k+1
        r_k1 = r_k - t*matrix_A.dot(v)
        #condição de saída
        if np.linalg.norm(r_k1, np.inf) < e:
            return x
        #pre-condicionador z no passo k+1
        z_k1 = np.linalg.inv(matrix_C).dot(r_k1)
        #calculo do beta
        beta = (z_k1.transpose().dot(r_k1)) / (z_k.transpose().dot(r_k))
        #atualiza p
        v = z_k1 + beta*v
        if k == stop_step:
            print(f"Matrix r_k {r_k}")
            print(f"Matrix z_k {z_k}")
            print(f"v {v}")
            print(f"t {t}")
            stop_step = int(input("input new stop step "))
        #Passa z e r na posição k+1 para k, pois vamos passar para o próximo ciclo!
        z_k = z_k1
        r_k = r_k1
        #input(rm_k)
        print(f"Step {k}, partial solution: {x}")
    return x
