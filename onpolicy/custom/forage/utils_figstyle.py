# Re-import everything after reset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

# Re-register Arial if needed
arial_fp = "fonts/ARIAL.TTF"  # Adjust path as necessary
# fm.fontManager.addfont(arial_fp)
arial_font = fm.FontProperties(fname=arial_fp)

def set_nature_style():
    sns.set(style="ticks")
    plt.rcParams.update({
        "font.family": arial_font.get_name(),
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

# Refactored plotting functions
def example_boxplot(ax=None):
    set_nature_style()
    data = [np.random.normal(loc=val, scale=1.0, size=20) for val in [10, 5]]
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.boxplot(data=data, palette="Greens", ax=ax)
    sns.stripplot(data=data, color='black', alpha=0.5, jitter=0.2, ax=ax)
    ax.set_xticklabels(['Observed', 'Shuffled'])
    ax.set_ylabel("Metric (%)")
    ax.text(0.5, max([d.max() for d in data]) + 1, '****', ha='center', fontsize=9)
    sns.despine(ax=ax)
    if 'fig' in locals():
        return fig

def example_line_sem(ax=None):
    set_nature_style()
    x = np.arange(10)
    y = np.random.rand(5, 10)
    mean = y.mean(axis=0)
    sem = y.std(axis=0) / np.sqrt(y.shape[0])
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(x, mean, color='blue')
    ax.fill_between(x, mean - sem, mean + sem, alpha=0.3, color='blue')
    ax.set_ylabel("Response")
    ax.set_xlabel("Time")
    sns.despine(ax=ax)
    if 'fig' in locals():
        return fig

def example_histogram(ax=None):
    set_nature_style()
    u1 = np.random.normal(-0.2, 0.05, 200)
    plsc1 = np.random.normal(0.2, 0.05, 200)
    both = np.random.normal(0.0, 0.03, 100)
    bins = np.linspace(-0.4, 0.4, 40)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.hist(u1, bins=bins, alpha=0.8, label="U1", color="#9999FF")
    ax.hist(plsc1, bins=bins, alpha=0.8, label="P1", color="#F4A5A5")
    ax.hist(both, bins=bins, alpha=0.8, label="Both", color="#990099")
    ax.set_xlabel(r"$|\mathbf{W}_{\mathrm{P1}}| - |\mathbf{W}_{\mathrm{U1}}|$")
    ax.set_ylabel("Cell count")
    ax.legend(loc='upper left', frameon=False)
    sns.despine(ax=ax)
    if 'fig' in locals():
        return fig

def example_tsne(ax=None):
    set_nature_style()
    X_embedded = np.random.randn(200, 2)
    y = np.random.randint(0, 8, 200)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c='lightgray', s=10)
    palette = sns.color_palette("hsv", 8)
    for i in range(8):
        idx = y == i
        ax.scatter(X_embedded[idx, 0], X_embedded[idx, 1], s=10, color=palette[i])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    sns.despine(ax=ax)
    if 'fig' in locals():
        return fig

def example_dotplot(ax=None):
    set_nature_style()
    labels = ['Attack', 'Escape', 'Defend', 'General sniffs', 'Tussling', 'Chasing',
              'Follow', 'Biteobj', 'Flinch', 'Exploreobj', 'Genital sniffs', 'Threaten']
    means = np.linspace(1.05, 0.01, len(labels)) + np.random.normal(0, 0.05, len(labels))
    errors = np.random.uniform(0.05, 0.15, len(labels))
    x = np.arange(len(labels))
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(3, len(labels)*0.3), 2.8))
    ax.errorbar(x, means, yerr=errors, fmt='o', color='black', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='right')
    ax.set_ylabel("Absolute coefficients")
    sns.despine(ax=ax)
    if 'fig' in locals():
        return fig