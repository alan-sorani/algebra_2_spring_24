from sympy import Matrix
from scipy.linalg import block_diag
from numpy.linalg import det
from hsnf import column_style_hermite_normal_form as hnf
import numpy as np
from numpy.random import randint
from numpy import sort
import json

#TODO move these values to json
MAX_NUM_EIGENVALS = 4
MAX_INVERTIBLE_COEFFICIENT = 3
MAX_COEFFICIENT = 200

def jordan_form(block_data):
    """
    Returns a Jordan matrix given specified eigenvalue and block size data.

    Parameters
    ----------
    jordan_data : list[tuple[number, list[int]]]
        A list containing pairs of eigenvalue and a list of block sizes.
    Returns
    -------
    A Jordan matrix with the given eigenvalues and block sizes, where the eigenvalues/block sizes that appear first, appear further to the top-left of the matrix.
    """
    blocks = [Matrix.jordan_block(*data) for data in block_data]
    return Matrix(block_diag(*blocks))

def random_with_sum(sum : int, length : int = 0):
    """
    Returns a list of positive integers the sum to a given sum. If a length is given, the list contains that mnay integers.
    """

    res = []
    remaining_sum = sum

    if(length == 0):
        while(remaining_sum > 0):
            temp = randint(1,remaining_sum+1)
            res += [temp]
            remaining_sum -= temp
        return list(sort(res)[::-1])

    remaining_elements = length
    while(remaining_elements > 1):
        temp = randint(1,remaining_sum-remaining_elements+2)
        res += [temp]
        remaining_elements -= 1
        remaining_sum -= temp
    res += [remaining_sum]
    return list(sort(res)[::-1])

def random_jordan(n : int, num_eigenvals : int, max_eigenval : int, invertible : bool = False):
    """
    Returns a random non-scalar jordan matrix of a given size with integer eigenvalues with a specified maximal absolute value and a given number of eigenvalues.
    """
    assert(n >= num_eigenvals)
    # the dimensions of the generalised eigenspaces
    gen_eigenspace_dims = random_with_sum(sum = n, length = num_eigenvals)
    block_sizes = [random_with_sum(dim) for dim in gen_eigenspace_dims]
    # if there's a single eigenvalue, make sure the matrix isn't scalar
    while((num_eigenvals == 1) and (block_sizes == [[1 for i in range(n)]])):
        block_sizes = [random_with_sum(dim) for dim in gen_eigenspace_dims]
    eigenvals = sort(randint(-max_eigenval, max_eigenval+1, num_eigenvals))[::-1]
    while((len(set(eigenvals)) != num_eigenvals) or (invertible and 0 in eigenvals)):
        eigenvals = sort(randint(-max_eigenval, max_eigenval+1, num_eigenvals))[::-1]
    jordan_data = [(eigenval, blocks) for eigenval, blocks in zip(eigenvals, block_sizes)]
    block_data = [(
        jordan_data[i][1][j],jordan_data[i][0]
        ) for i in range(len(jordan_data)) for j in range(len(jordan_data[i][1]))]
    block_names = [f"J_{{{data[0]}}}({data[1]})" for data in block_data]
    matrix = jordan_form(block_data)
    compact_form = "diag" + chr(40)
    for name in block_names:
        compact_form += name + ", "
    compact_form = compact_form[:-2]
    compact_form += chr(41)
    return (matrix,compact_form)

def rand_invertible(n : int, max_coefficient : int):
    """
    Returns a random invertible matrix of a given size and a maximum absolute value for the matrix entries.
    """
    matrix = randint(-max_coefficient, max_coefficient+1, (n,n))
    while(det(matrix) == 0):
        matrix = randint(-max_coefficient, max_coefficient+1, (n,n))
    return Matrix(matrix)

def rand_unimodular(n: int, max_coefficient: int):
    """
    Returns a random unimodular matrix of a given size.
    """
    P = rand_invertible(n, max_coefficient)
    return Matrix(hnf(P)[1])

def rand_matrix(n : int, num_eigenvals : int, max_eigenval : int, max_coefficient : int):
    """
    Returns a random matrix, its Jordan form, and an invertible change-of-basis matrix.

    Parameters
    ----------
    n : int
        The size of the matrices.
    num_eigenvals: int
        The number of different eigenvalues of the returned matrix.
    max_eigenval: int
        The maximal absolute value for the eigenvalues of the matrix.
    max_coefficient: int
        A value used to bound coefficients in the generation of P.

    Returns
    -------
    tuple[Matrix((n,n)),Matrix((n,n)),Matrix((n,n))]
        A tuple (A,J,P) such that A,J,P are all n-by-n matrices such that A = P^{-1} * A * P.
        A,J have integer coefficients.
        A,J have num_eigenvals different eigenvalues each of which is at most max_eigenval in absolute value.
        The value of max_coefficients is the max value for the coefficients of a matrix M such that P is given from M as the unimodular matrix in its collumn-style Hermite normal form
    """
    A = float('inf')
    while((np.abs(np.array(A)) > MAX_COEFFICIENT).any()):
        J = random_jordan(n, num_eigenvals, max_eigenval)
        P = rand_unimodular(n, max_coefficient)
        A = P @ J[0] @ P.inv()

    return (A,J,P)

def create_exercises(sizes : dict[int, int], max_eigenval : int):
    """
    Create a list of exercises of the form (A,J,P) where A is a matrix, J is its Jordan form, and P is a matrix such that P^{-1} * A * P = J.
    """
    res = []
    prev_matrices = []
    for size, num_exercises in sizes.items():
        i = 1
        while(i <= num_exercises):
            num_eigenvals = randint(1, np.min([size,MAX_NUM_EIGENVALS, 2*max_eigenval+1])+1)
            A,J,P = rand_matrix(size, num_eigenvals, max_eigenval, MAX_INVERTIBLE_COEFFICIENT)
            while(A == J[0]):
                A,J,P = rand_matrix(size, num_eigenvals, max_eigenval, MAX_INVERTIBLE_COEFFICIENT)
            if(A not in prev_matrices):
                res += [(A,J[1],P)]
                prev_matrices += [A]
                i += 1
    return res

def matrix_to_latex(mat : Matrix):
    res = r"\pmat" + chr(123)
    text = str(mat)[8:-2]
    text = text.replace(
            "[", ""
            ).replace(
            "],", " \\\\"
            ).replace(
            ",", " &"
            ).replace(
            "]", ""
            )
    res += text
    res += chr(125)
    return res

def generic_vector(dim : int, space: str = "euclidean"):
    if(space == "euclidean"):
        res = "\\pmat" + chr(123)
        for i in range(dim):
            res += f"v_{{{i+1}}} \\\\ "
        res = res[:-3]
        res += chr(125)
        return res
    if(space == "matrices"):
        res = "\\pmat" + chr(123)
        size = int(np.sqrt(dim))
        for i in range(size):
            for j in range(size):
                res += f"a_{{{i+1},{j+1}}} & "
            res = res[:-2]
            res += "\\\\ "
        res = res[:-3]
        res += chr(125)
        return res
    if(space == "polynomials"):
        res = f"\\sum_{{i=0}}^{{{dim-1}}} a_{{i}} x^{{i}}"
        return res

def image_vector(matrix : Matrix,
                 space : str = "euclidean"
                 ):
    dim = matrix.shape[0]
    
    #TODO fill in vectors
    if(space == "euclidean"):
        res = ""
    if(space == "matrices"):
        res = ""
    if(space == "polynomials"):
        res = ""


def matrix_to_transformation(mat: Matrix,
                             space: str = "euclidean"):
    dim = mat.shape[0]
    vector_space = {
        "euclidean": f"\\mathbb{{R}}^{{{dim}}}",
        "matrices": f"\\mathrm{{Mat}}_{{{int(np.sqrt(dim))} \
\\times {int(np.sqrt(dim))}}}",
        "polynomials": f"\\mathbb{{R}}_{{{dim - 1}}}[x]"
    }[space]
    res = f"T \\colon {vector_space} &\\rightarrow {vector_space} \\\\\n"
    res += f"{generic_vector(dim, space)} &\\mapsto "
    res += f"A" #TODO fill in image
    return res

def exercises_to_latex(exercise_list):
    exercises = "\\begin{enumerate}\n\n"
    solutions = exercises
    for exercise in exercise_list:
        A,J,P = exercise
        exercises += "\\item\n\n"
        solutions += "\\item\n\n"
        exercises += "\\begin{align*}\n"
        solutions += "\\begin{align*}\n"
        exercises += "A = " + matrix_to_latex(A)
        solutions += "J &= " + "\\" + J + "\\\\\n"
        solutions += "P &= " + matrix_to_latex(P)
        exercises += "\n" + "\\end{align*}" + "\n\n"
        solutions += "\n" + "\\end{align*}" + "\n\n"
    exercises += "\\end{enumerate}\n"
    solutions += "\\end{enumerate}\n"
    return exercises, solutions

def get_parameters():
    with open('config.json') as file:
        params = json.load(file)
    for d in {'nilpotent_exercises', 'general_exercises'}:
        params[d] = dict(
                [
                (int(key), value) for key,value in params[
                    d
                    ].items()
                ])
    return params

if(__name__ == "__main__"):
    params = get_parameters()

    nil_ex_list = create_exercises(
            params["nilpotent_exercises"],
            0
            )
    general_ex_list = create_exercises(
            params["general_exercises"],
            params["max_eigenvalue"]
            )
    nil_ex, nil_sol = exercises_to_latex(nil_ex_list)
    general_ex, general_sol = exercises_to_latex(general_ex_list)
    with open(f"{params['filename']}_nil_ex.tex", "w") as file:
        file.write(nil_ex)
    with open(f"{params['filename']}_nil_sol.tex", "w") as file:
        file.write(nil_sol)
    with open(f"{params['filename']}_general_ex.tex", "w") as file:
        file.write(general_ex)
    with open(f"{params['filename']}_general_sol.tex", "w") as file:
        file.write(general_sol)
