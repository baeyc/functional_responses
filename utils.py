
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Variables used in graphs
secondary_labels_map = {
    '10': '50',
    '20': '25',
    '30': '17',
    '40': '12',
    '50': '10',
    '60': '8',
    '70': '7',
    '80': '6',
    '90': '5'
}
primary_group_order = ['10', '20', '30', '40', '50', '60', '70', '80', '90']

palette = {r"$\theta_1$": "tab:blue", r"$\theta_2$": "tab:orange"}


def invert(x):
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 500 / x[~near_zero]
    return x


def plot_boxplot_with_twinx_axis(data, **kwargs):
    ax1 = plt.gca()
    sns.boxplot(x=data['N'], y=data['value'], ax=ax1, order=primary_group_order,
                hue=data["Parameter set"], palette=palette, dodge=True, gap=0.2, fliersize=2, width=1)

    ax1.set_ylabel('estimate')
    ax1.set_xlabel(r'$N$ (nb individuals)')
    ax1.tick_params(axis='y')

    ax2 = ax1.twiny()
    primary_ticks_locs = ax1.get_xticks()
    primary_labels = [t.get_text() for t in ax1.get_xticklabels()]

    secondary_labels = [secondary_labels_map.get(
        label, '') for label in primary_labels]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(primary_ticks_locs)
    ax2.set_xticklabels(secondary_labels, ha='left', color='grey')
    ax2.set_xlabel(r'$n$ (nb obs per individual)', color='grey')
    ax2.tick_params(axis='x', labelcolor='grey')
