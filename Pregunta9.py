
A = [
    [-4, 2, -4, -4, 1, 2, 5, 3, 5, 1],
    [1, 0, 4, 3, 0, -2, 3, 0, 1, 5],
    [5, 5, -4, 5, -4, 2, 2, 2, 4, 4],
    [-1, 3, 4, -1, -4, 0, 5, 0, 0, 5],
    [4, 1, 4, 2, 0, 0, 3, -1, 0, 2],
    [2, -2, 1, -1, -2, -3, 2, -2, 4, -1],
    [3, -2, -3, -2, -1, -3, 5, -1, 5, 0],
    [3, 4, -3, 3, -2, 2, -4, -4, 1, 5],
    [-4, 0, 3, 3, -3, -2, -2, 0, 5, -4],
    [-2, 4, 4, -2, -1, 1, 5, -1, 3, -3],
]

A2 = [
    [2, 2, 4, 5, -2, -3, 2, -2],
    [-1, -1, 3, 2, 1, 1, -4, 4],
    [2, 5, -3, -3, -2, 2, 5, 3],
    [-2, -4, 0, 1, -1, 5, -4, -1],
    [1, -2, -1, 5, 5, 2, 1, -2],
    [5, 4, 0, 3, 4, -1, -3, -2],
    [4, -4, 1, 2, 3, 3, -1, 3],
    [-2, 1, -3, 0, 5, 4, 4, -4],
]

def calc_determinante(A: list[list[float]]) -> float:
    """Función que calcula el determinante usando el método
    [Descomposición LU, eliminación gaussiana, Gauss-Jordan, Gauss-Jacobi o Gauss-Seidel]

    ## Parameters
    ``A``: Matriz cuadrada de tamaño n x n

    ## Return
    ``detA``: Determinante de la matriz A

    """
    # completar
    from src import descomposicion_LU
    
    det = 1
    L, U = descomposicion_LU(A)
    
    
    for i in range(0, len(U)):
        det *= U[i][i]
    
    return det

print(calc_determinante(A))
print(calc_determinante(A2))
