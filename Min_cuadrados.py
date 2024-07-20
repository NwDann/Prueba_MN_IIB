
def der_parcial_2(xs: list, ys: list) -> tuple[float, float, float, float]:
    """Retorna los coeficientes de la ecuación de la derivada parcial con respecto al parámetro 2.
    
    c_2 * a_2 + c_1 * a_1 + c_0 * a_0 = c_ind
    
    ## Parameters
    ``xs``: lista de valores de x.
    ``ys``: lista de valores de y.
    
    ## Return
    ``c_2``: coeficiente del parámetro 2.
    ``c_1``: coeficiente del parámetro 1.
    ``c_0``: coeficiente del parámetro 0.
    ``c_ind``: coeficiente del término independiente.
    """
    c_2 = sum(xi**4 for xi in xs)
    c_1 = sum(xi**3 for xi in xs)
    c_0 = sum(xi**2 for xi in xs)
    c_ind = sum(yi * xi**2 for yi, xi in zip(ys, xs))
    return (c_2, c_1, c_0, c_ind)

def der_parcial_1(xs: list, ys: list) -> tuple[float, float, float, float]:
    """Retorna los coeficientes de la ecuación de la derivada parcial con respecto al parámetro 1.
    
    c_2 * a_2 + c_1 * a_1 + c_0 * a_0 = c_ind
    
    ## Parameters
    ``xs``: lista de valores de x.
    ``ys``: lista de valores de y.
    
    ## Return
    ``c_2``: coeficiente del parámetro 2.
    ``c_1``: coeficiente del parámetro 1.
    ``c_0``: coeficiente del parámetro 0.
    ``c_ind``: coeficiente del término independiente.
    """
    c_2 = sum(xi**3 for xi in xs)
    c_1 = sum(xi**2 for xi in xs)
    c_0 = sum(xi for xi in xs)
    c_ind = sum(yi * xi for yi, xi in zip(ys, xs))
    return (c_2, c_1, c_0, c_ind)

def der_parcial_0(xs: list, ys: list) -> tuple[float, float, float, float]:
    """Retorna los coeficientes de la ecuación de la derivada parcial con respecto al parámetro 0.
    
    c_2 * a_2 + c_1 * a_1 + c_0 * a_0 = c_ind
    
    ## Parameters
    ``xs``: lista de valores de x.
    ``ys``: lista de valores de y.
    
    ## Return
    ``c_2``: coeficiente del parámetro 2.
    ``c_1``: coeficiente del parámetro 1.
    ``c_0``: coeficiente del parámetro 0.
    ``c_ind``: coeficiente del término independiente.
    """
    c_2 = sum(xi**2 for xi in xs)
    c_1 = sum(xi for xi in xs)
    c_0 = len(xs)
    c_ind = sum(ys)
    return (c_2, c_1, c_0, c_ind)

def parabola(x: float, pars: tuple[float, float, float]) -> float:
    """Ecuación de la parábola y = a2 * x^2 + a1 * x + a0.
    
    ## Parameters
    ``x``: valor de x.
    ``pars``: parámetros de la parábola. Deben ser de la forma (a2, a1, a0).
    
    ## Return
    ``y``: valor de y.
    """
    a2, a1, a0 = pars
    return a2 * x**2 + a1 * x + a0

xs2 = [
    -5.0000,
    -3.8889,
    -2.7778,
    -1.6667,
    -0.5556,
    0.5556,
    1.6667,
    2.7778,
    3.8889,
    5.0000,
]
ys2 = [
    57.2441,
    33.0303,
    16.4817,
    7.0299,
    0.5498,
    0.7117,
    3.4185,
    12.1767,
    24.9167,
    44.2495,
]

from src import ajustar_min_cuadrados
pars2 = ajustar_min_cuadrados(xs2, ys2, gradiente=[der_parcial_2, der_parcial_1, der_parcial_0])
pars2 # parámetros de la curva ajustada

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = [parabola(xi, pars2) for xi in x]

plt.scatter(xs2, ys2, label="Datos")
plt.plot(x, y, color="red", label=r"$ y = a_2 x^2 + a_1 x + a_0 $")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste por mínimos cuadrados")
plt.legend()
plt.show()