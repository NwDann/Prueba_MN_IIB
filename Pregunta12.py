
import numpy as np
import matplotlib.pyplot as plt
from src import gauss_jacobi, gauss_seidel


A = [[1, 1], [-2, 5]]
b = [7, 0]
n = len(A)


sol_jacobi, iteraciones_jacobi = gauss_jacobi(A=A, b=b, x0=[1]*n, tol=1e-5, max_iter=1000)
sol_seidel, iteraciones_seidel = gauss_seidel(A=A, b=b, x0=[1]*n, tol=1e-5, max_iter=1000)



# Graficar las trayectorias
iteraciones_jacobi = np.array(iteraciones_jacobi)
iteraciones_seidel = np.array(iteraciones_seidel)

plt.figure(figsize=(10, 5))

# Gauss-Jacobi
plt.subplot(1, 2, 1)
plt.plot(iteraciones_jacobi[:, 0], iteraciones_jacobi[:, 1], marker='o', label='Trayectoria')
plt.scatter(sol_jacobi[0], sol_jacobi[1], color='red', label='Solución')
plt.title('Gauss-Jacobi')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

# Gauss-Seidel
plt.subplot(1, 2, 2)
plt.plot(iteraciones_seidel[:, 0], iteraciones_seidel[:, 1], marker='o', label='Trayectoria')
plt.scatter(sol_seidel[0], sol_seidel[1], color='red', label='Solución')
plt.title('Gauss-Seidel')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()

plt.tight_layout()
plt.show()
