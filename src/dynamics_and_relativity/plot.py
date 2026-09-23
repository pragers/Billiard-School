import numpy as np
import matplotlib.pyplot as plt

uitrekking = np.array([2, 2, 4, 4, 6, 6, 7, 7])
acceleration = np.array([-2.7, -3.6, -2.8, -4.1, -5.4, -3.9, -1.1, -2.8])
error_a = np.array([1.9, 1.2, 2.2, 1.8, 1.0, 2.0, 1.3, 1.0])

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    uitrekking,
    acceleration,
    yerr=error_a,
    fmt="o",
    capsize=4,
    markersize=6,
    label="Metingen",
)

ax.set(
    xlabel=r"Uitrekking $x$ (cm)",
    ylabel=r"Versnelling $a$ (m/s$^2$)",
    title="Versnelling als functie van de uitrekking",
)

ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.15)
ax.minorticks_on()

ax.legend()
fig.tight_layout()

plt.show()
fig.savefig("output.svg")
