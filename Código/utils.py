from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)  # La respuesta a la pregunta de la vida, el universo y todo lo demás


def simular_experimentos_bernoulli(n_experimentos: int, p_exito: float) -> int:
    """Simula experimentos de Bernoulli y devuelve el número total de éxitos.

    Parameters:
    -----------
        `n_experimentos {int}`: Número de experimentos a simular.

        `p_exito {float}`: Probabilidad de que un dado experimento resulte en un éxito.

    Returns:
    --------
        `n_exitos {int}`: Número de experimentos que resultaron en exitos.
    """
    global RNG
    resultados = RNG.random(n_experimentos) < p_exito
    # Notar que esto debe definirse con un `<` y no con un `<=`
    # ya que en el caso de que p = 1, el resultado de la comparación
    # debe ser True para cualquier valor de la muestra aleatoria.
    # Analogamente, si p = 0, el resultado de la comparación debe ser False
    # aún si el número aleatorio es 0.
    return resultados.sum()


def plot_discrete_hist(
    data: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = "#8222d2",
    ecolor: Optional[str] = "#1b1752",
    edgecolor: Optional[str] = "black",
    zorder: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula y grafica un histograma de los valores en `data` agregando barras de \
error para cada bin utilizando la desviación estándar de una distribución binomial con \
`n = data.size` y `p = bin_count.size / data.size`.

    Parameters:
    -----------
        `data {np.ndarray}`: Valores cuya distribución se quiere observar en un \
histograma.

        `ax {plt.Axes, optional}`: Axes de Matplotlib en sobre los que graficar.

        `label {str, optional}`: Etiqueta para la leyenda del gráfico.

        `color {str, optional}`: Color de las barras del gráfico.

        `ecolor {str, optional}`: Color de las barras de error del gráfico.

        `edgecolor {str, optional}`: Color de los bordes de las barras del gráfico.

        `zorder {int, optional}`: Orden de las barras en las lineas del gráfico.

    Returns:
    --------
        `bin_centers {np.ndarray, size=N}`: Valores de los centros de los bines.

        `counts {np.ndarray, size=N}`: Cuentas por bin.

        `bin_errors {np.ndarray, size=N}`: Desviación estándar para las cuentas en \
cada bin.

    """
    bin_centers, counts = np.unique(data, return_counts=True)
    probs = counts / data.size
    bin_var = data.size * probs * (1 - probs)  # Var(Binom) = n·p·(1-p)
    bin_errors = np.sqrt(bin_var)  # The standard deviation for each bin
    plot_kwargs = dict(
        x=bin_centers,
        height=(counts / data.size),
        yerr=(bin_errors / data.size),
        label=label,
        align="center",
        alpha=0.75,
        width=1,  # Dado que los datos son discretos, el ancho de las barras es 1
        linewidth=0.5,
        color=color,
        edgecolor=edgecolor,
        zorder=zorder,
        error_kw=dict(
            capsize=5,
            capthick=1,
            ecolor=ecolor,
            elinewidth=2,
            zorder=(zorder + 1),
        ),
    )
    if ax is None:
        plt.bar(**plot_kwargs)
    else:
        ax.bar(**plot_kwargs)
    return bin_centers, counts, bin_errors


def plot_discrete_stem(
    variable_values: np.ndarray,
    probability_values: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = "#ffd30c",
    linecolor: Optional[str] = "k",
    zorder: Optional[int] = None,
) -> None:
    """Plots a discrete distribution as a stem plot.

    Parameters:
    -----------
        `variable_values {np.ndarray}`: Valores de la variable aleatoria.

        `prob_values {np.ndarray}`: Probabilidades asociadas a cada valor de la \
variable aleatoria.

        `ax {plt.Axes, optional}`: Axes de Matplotlib en sobre los que graficar.

        `label {str, optional}`: Etiqueta para la leyenda del gráfico.

        `color {str, optional}`: Color de las barras del gráfico.

        `linecolor {str, optional}`: Color de las lineas del gráfico.

        `zorder {int, optional}`: Orden de las barras en las lineas del gráfico.

    """
    marker_kwargs = dict(
        x=variable_values,
        y=probability_values,
        label=label,
        s=30,
        marker="h",
        color=color,
        edgecolors=linecolor,
        linewidths=0.5,
        zorder=zorder,
    )
    vlines_kwargs = dict(
        x=variable_values,
        ymin=np.zeros_like(variable_values),
        ymax=probability_values,
        linestyle="--",
        linewidth=0.75,
        colors=linecolor,
        zorder=(zorder - 1),
    )
    if ax is None:
        plt.scatter(**marker_kwargs)
        plt.vlines(**vlines_kwargs)
    else:
        ax.scatter(**marker_kwargs)
        ax.vlines(**vlines_kwargs)
    return None
