"""
Actividad 04 - Apartado 3: dataset sintético CIRCLES
Modelos: K-means, DBSCAN, SpectralClustering
Métricas externas: ARI, V-measure | Internas: Silhouette, Davies-Bouldin, Elbow
Aplicación gráfica local (Tkinter + Matplotlib) para PyCharm.

Organización:
    1. CONFIGURACIÓN
    2. CAPA DE DATOS     -> generar y preparar el dataset Circles
    3. CAPA DE MODELOS   -> entrenar K-means, DBSCAN, Spectral (+ codo)
    4. CAPA DE MÉTRICAS  -> validación externa e interna
    5. INTERFAZ GRÁFICA  -> 3D animado, tabla de métricas, gráfico del codo

Pasos: generar datos -> escalar -> entrenar -> predecir etiquetas -> evaluar -> visualizar.
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, v_measure_score,
                             silhouette_score, davies_bouldin_score)

# ======================================================================
# 1. CONFIGURACIÓN
# ======================================================================
SEED = 42
N_SAMPLES = 400
NOISE = 0.05
FACTOR = 0.5                     # radio del círculo interno / externo
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf"]
NOISE_COLOR = "#b0b0b0"          # DBSCAN: puntos de ruido (-1)
DEFAULTS = {"k": 2, "eps": 0.30, "min_samples": 5, "n_neighbors": 15}


# ======================================================================
# 2. CAPA DE DATOS
# ======================================================================
class DatasetRepository:
    @staticmethod
    def load_circles():
        X, y = make_circles(n_samples=N_SAMPLES, noise=NOISE, factor=FACTOR,
                            random_state=SEED)
        X = StandardScaler().fit_transform(X)                  # escalado
        X3d = np.column_stack([X, np.zeros(len(X))])           # z=0; la animación eleva
        return {"X": X, "y": y, "X3d": X3d}


# ======================================================================
# 3. CAPA DE MODELOS
# ======================================================================
class ModelRunner:
    @staticmethod
    def fit_all(X, k, eps, min_samples, n_neighbors):
        models = {
            "K-means": KMeans(n_clusters=k, n_init=10, random_state=SEED),
            "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
            "Spectral": SpectralClustering(
                n_clusters=k, affinity="nearest_neighbors",
                n_neighbors=n_neighbors, assign_labels="kmeans",
                random_state=SEED),
        }
        return {name: m.fit_predict(X) for name, m in models.items()}

    @staticmethod
    def elbow(X, k_max=8):
        ks = list(range(2, k_max + 1))
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_))
        return ks, inertias, sils


# ======================================================================
# 4. CAPA DE MÉTRICAS
# ======================================================================
class Metrics:
    @staticmethod
    def evaluate(X, y_true, labels):
        mask = labels != -1                         # las internas ignoran el ruido
        n_clusters = len(set(labels[mask]))
        res = {
            "Clusters": n_clusters,
            "Ruido": int((~mask).sum()),
            "ARI": adjusted_rand_score(y_true, labels),
            "V-measure": v_measure_score(y_true, labels),
            "Silhouette": np.nan,
            "Davies-Bouldin": np.nan,
        }
        if n_clusters >= 2 and mask.sum() > n_clusters:
            res["Silhouette"] = silhouette_score(X[mask], labels[mask])
            res["Davies-Bouldin"] = davies_bouldin_score(X[mask], labels[mask])
        return res


# ======================================================================
# 5. INTERFAZ GRÁFICA
# ======================================================================
def colors_for(labels):
    uniq = sorted(l for l in set(labels) if l != -1)
    cmap = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(uniq)}
    cmap[-1] = NOISE_COLOR
    return [cmap[l] for l in labels]


def levels_for(labels):
    uniq = sorted(l for l in set(labels) if l != -1)
    lv = {l: i for i, l in enumerate(uniq)}
    lv[-1] = -1
    return np.array([lv[l] for l in labels], dtype=float)


class App(tk.Tk):
    ROTATE_FRAMES = 360
    LIFT_FRAMES = 40
    LIFT_SCALE = 1.2

    def __init__(self):
        super().__init__()
        self.title("Actividad 04 - Apartado 3: Circles (K-means, DBSCAN, Spectral)")
        self.geometry("1350x850")
        self.anim = None
        self.paused = False
        self.data = DatasetRepository.load_circles()
        self._build_controls()
        self._build_tabs()
        self.run()

    # ---------- controles ----------
    def _build_controls(self):
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        self.vars = {}
        for label, key in [("k", "k"), ("eps (DBSCAN)", "eps"),
                           ("min_samples", "min_samples"),
                           ("n_neighbors (Spectral)", "n_neighbors")]:
            ttk.Label(bar, text=label).pack(side="left", padx=(10, 2))
            v = tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Entry(bar, textvariable=v, width=6).pack(side="left")
            self.vars[key] = v
        ttk.Button(bar, text="▶ Ejecutar", command=self.run).pack(side="left", padx=12)
        self.pause_btn = ttk.Button(bar, text="⏸ Pausar", command=self._toggle_anim)
        self.pause_btn.pack(side="left")
        ttk.Label(bar, foreground="#444",
                  text="Circles: 2 círculos concéntricos (no convexo)").pack(side="left", padx=15)

    def _toggle_anim(self):
        if not self.anim:
            return
        if self.paused:
            self.anim.resume(); self.pause_btn.config(text="⏸ Pausar")
        else:
            self.anim.pause(); self.pause_btn.config(text="▶ Reanudar")
        self.paused = not self.paused

    # ---------- pestañas ----------
    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        t1 = ttk.Frame(nb); nb.add(t1, text="Clusters 3D (animado)")
        self.fig3d = Figure(figsize=(13, 6.5), dpi=100)
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=t1)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)

        t2 = ttk.Frame(nb); nb.add(t2, text="Métricas")
        cols = ("Modelo", "Clusters", "Ruido", "ARI", "V-measure",
                "Silhouette", "Davies-Bouldin")
        self.tree = ttk.Treeview(t2, columns=cols, show="headings", height=5)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(fill="x", padx=10, pady=10)
        self.explain = tk.Text(t2, height=18, wrap="word", font=("Segoe UI", 10))
        self.explain.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        t3 = ttk.Frame(nb); nb.add(t3, text="Elbow / Silhouette vs k")
        self.fig_elbow = Figure(figsize=(10, 5), dpi=100)
        self.canvas_elbow = FigureCanvasTkAgg(self.fig_elbow, master=t3)
        self.canvas_elbow.get_tk_widget().pack(fill="both", expand=True)

    # ---------- flujo principal ----------
    def run(self):
        try:
            p = {k: (float(v.get()) if k == "eps" else int(v.get()))
                 for k, v in self.vars.items()}
            X, y = self.data["X"], self.data["y"]
            labels = ModelRunner.fit_all(X, p["k"], p["eps"], p["min_samples"],
                                         p["n_neighbors"])
        except Exception as e:
            messagebox.showerror("Error", f"Revisa los hiperparámetros.\n{e}")
            return
        self.results = {n: Metrics.evaluate(X, y, l) for n, l in labels.items()}
        self._fill_metrics()
        self._draw_elbow(X)
        self._start_animation(labels)

    def _fill_metrics(self):
        self.tree.delete(*self.tree.get_children())
        fmt = lambda v: "—" if (isinstance(v, float) and np.isnan(v)) else (
            f"{v:.3f}" if isinstance(v, float) else v)
        for name, r in self.results.items():
            self.tree.insert("", "end", values=(
                name, r["Clusters"], r["Ruido"], fmt(r["ARI"]), fmt(r["V-measure"]),
                fmt(r["Silhouette"]), fmt(r["Davies-Bouldin"])))
        self.explain.delete("1.0", "end")
        self.explain.insert("end", (
            "¿QUÉ SIGNIFICAN LAS MÉTRICAS?\n"
            "• ARI (externa): 1 = idéntico a las etiquetas reales; ~0 = azar.\n"
            "• V-measure (externa): homogeneidad y completitud (0 a 1).\n"
            "• Silhouette (interna): -1 a 1; alto = grupos compactos y separados.\n"
            "• Davies-Bouldin (interna): menor es mejor.\n"
            "• Elbow: codo de la inercia de K-means vs k (ver pestaña 3).\n\n"
            "EN CIRCLES: los grupos son anillos concéntricos, no compactos. K-means "
            "corta con una recta y mezcla ambos anillos (ARI ~0), aunque su Silhouette "
            "salga más alto. DBSCAN y Spectral siguen la conectividad/densidad y "
            "separan los anillos (ARI = 1), pero sus métricas internas salen bajas "
            "porque los anillos no son compactos. Por eso las métricas internas no "
            "son confiables en formas no convexas y hay que apoyarse en ARI/V-measure.\n"
        ))

    def _draw_elbow(self, X):
        ks, inertias, sils = ModelRunner.elbow(X)
        self.fig_elbow.clear()
        a1 = self.fig_elbow.add_subplot(121)
        a1.plot(ks, inertias, "o-", color=PALETTE[0])
        a1.set_title("Método del codo (K-means)")
        a1.set_xlabel("k"); a1.set_ylabel("Inercia (WCSS)"); a1.grid(alpha=.3)
        a2 = self.fig_elbow.add_subplot(122)
        a2.plot(ks, sils, "s-", color=PALETTE[1])
        a2.set_title("Silhouette vs k")
        a2.set_xlabel("k"); a2.set_ylabel("Silhouette"); a2.grid(alpha=.3)
        self.fig_elbow.tight_layout()
        self.canvas_elbow.draw()

    # ---------- animación 3D ----------
    def _start_animation(self, labels_by_model):
        if self.anim:
            self.anim.event_source.stop()
        self.paused = False
        self.pause_btn.config(text="⏸ Pausar")

        panels = [("Etiquetas reales (forma del dataset)", self.data["y"])]
        panels += list(labels_by_model.items())
        X3 = self.data["X3d"]
        self.fig3d.clear()
        self.axes, self.scatters, self.levels = [], [], []

        for i, (title, labels) in enumerate(panels):
            ax = self.fig3d.add_subplot(1, 4, i + 1, projection="3d")
            sc = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=colors_for(labels),
                            s=14, depthshade=True, edgecolors="none")
            ax.set_title(f"{title}\n({len(set(labels) - {-1})} grupos)", fontsize=9)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.set_xlim(X3[:, 0].min(), X3[:, 0].max())
            ax.set_ylim(X3[:, 1].min(), X3[:, 1].max())
            lv = levels_for(labels)
            ax.set_zlim(-self.LIFT_SCALE, self.LIFT_SCALE * max(lv.max(), 1))
            self.axes.append(ax); self.scatters.append(sc); self.levels.append(lv)
        self.fig3d.tight_layout()
        self.X3 = X3

        def update(frame):
            ramp = min(frame / self.LIFT_FRAMES, 1.0)      # fase 1: separar grupos
            for ax, sc, lv in zip(self.axes, self.scatters, self.levels):
                sc._offsets3d = (self.X3[:, 0], self.X3[:, 1],
                                 lv * self.LIFT_SCALE * ramp)
                ax.view_init(elev=22, azim=frame)           # fase 2: rotación
            return self.scatters

        self.anim = FuncAnimation(self.fig3d, update, frames=self.ROTATE_FRAMES,
                                  interval=40, blit=False, repeat=True)
        self.canvas3d.draw()


if __name__ == "__main__":
    App().mainloop()