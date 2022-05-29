import matplotlib.pyplot as plt

# Try to match IEEEtran used by the report
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8

# Colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# reasonable figure sizes for plots
WIDTH = 3.37566
HEIGHT = 2

def small_legend():
    """
    Helper function to add the legend, with a smaller font size.
    """
    plt.legend(prop={'size': 6})

def export(plot_name, double=False, grid=True):
    """
    Helper function to resize and export the current figure automatically.
    """
    plt.grid(grid) # ensure grid is displayed
    plt.gcf().set_size_inches((WIDTH, HEIGHT * 2.3 if double == "triple" else HEIGHT * 1.5 if double else HEIGHT)) # resize
    plt.tight_layout() # better subplot layout
    plt.savefig("plots/%s.pdf" % plot_name, bbox_inches='tight') # export with tight bounding boxes

def plot_heatmap(qvalues, **kwargs):
    heaps, heap_sizes = list(zip(*qvalues.keys()))
    df = pd.DataFrame({'Heap': [1]*7+[2]*7+[3]*7, 'Heap size': list(np.arange(7, 0, -1))*3})
    df = df.merge(pd.DataFrame({'Heap': heaps, 'Heap size': heap_sizes, 'qvalue': qvalues.values()}),
                  on=['Heap', 'Heap size'], how='outer')
    heatmap = df.pivot('Heap', 'Heap size', 'qvalue')
    sns.heatmap(heatmap, cmap='RdYlGn', center=0, vmin=-1, vmax=1, annot=True, fmt=".1f", **kwargs)

def plot_heatmap_from_deep(qvalues, **kwargs):
    dict_converted = { (i//7+1, i%7+1): float(val) for i, val in enumerate(qvalues) }
    plot_heatmap(dict_converted, **kwargs)
