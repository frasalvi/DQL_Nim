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