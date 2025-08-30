
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import pandas as pd
import re
import os
import pathlib
from typing import Tuple, Union

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


thetanames = [r'$\mu_{\lambda}$', r'$\mu_{h}$',
              r'$\sigma^2_{\lambda}$', r'$\sigma^2_{h}$', r'$\sigma_0^2$']


def load(fn):

    with open(fn, 'rb') as f:
        theta = jnp.load(f)

    (nrep, niter, nparams) = theta.shape

    str_split = re.split(r'[_.]', fn)
    size = int([idx[1:] for idx in str_split if idx[0] == "n"][0])
    cv = int([idx[2:] for idx in str_split if idx.startswith("cv")][0])

    theta = theta.reshape((nrep * niter, nparams))
    dtheta = pd.DataFrame(theta, columns=thetanames)
    dtheta['n'] = size
    dtheta['cv'] = cv
    dtheta['rep'] = jnp.repeat(jnp.arange(1, nrep + 1), niter)
    dtheta['iter'] = jnp.tile(jnp.arange(1, niter + 1), nrep)

    return dtheta


def loadtheta(fn):
    with open(fn, 'rb') as f:
        theta = jnp.load(f)

    (nrep, nparams) = theta.shape

    str_split = re.split(r'[_.]', fn)
    size = [idx[1:] for idx in str_split if idx[0] == "n"]
    size = int(size[0])
    cv = int([idx[2:] for idx in str_split if idx.startswith("cv")][0])

    theta = theta.reshape((nrep, nparams))
    dtheta = pd.DataFrame(theta)

    dtheta.columns = thetanames
    dtheta['N'] = jnp.repeat(size, nrep)
    dtheta['cv'] = jnp.repeat(cv, nrep)
    dtheta['rep'] = jnp.arange(1, nrep+1)

    return dtheta


def loadBIC(fn):
    with open(fn, 'rb') as f:
        bic = jnp.load(f, allow_pickle=True)

    nrep = len(bic)

    str_split = re.split(r'[_.]', fn)
    size = [idx[1:] for idx in str_split if idx[0] == "n"]
    size = int(size[0])
    cv = int([idx[2:] for idx in str_split if idx.startswith("cv")][0])

    modtype = "wrong" if 'bicwrong' in str_split else "correct"

    dbic = pd.DataFrame(bic)
    dbic.columns = ['BIC']
    dbic['N'] = size
    dbic['cv'] = cv
    dbic['rep'] = jnp.arange(1, nrep+1)
    dbic['Model'] = modtype

    return dbic


def plot_bias_std(path, residual):

    figsize = (4, 4)
    palette = "tab10"

    thetanames = [r'$\mu_{\lambda}$', r'$\mu_{h}$',
                  r'$\sigma^2_{\lambda}$', r'$\sigma^2_{h}$', r'$\sigma_0^2$']

    suffix = "residual_cv" if residual else "rand_eff_cv"
    pattern = re.compile(fr"{suffix}\d+")

    files = [f for f in os.listdir(path) if re.match(pattern, f)]
    params_data = pd.concat([load(os.path.join(path, f)) for f in files])

    params_last = (
        params_data
        .groupby(['n', 'cv', 'rep'])[thetanames].mean()
        .reset_index()
    )

    for col in [r'$\sigma^2_{\lambda}$', r'$\sigma^2_{h}$', r'$\sigma_0^2$']:
        params_last[col] = params_last[col].apply(lambda x: float(jnp.exp(x)))

    params_last = pd.melt(params_last,
                          id_vars=['n', 'cv', 'rep'],
                          var_name='Parameter',
                          value_name='value')

    cv_list = sorted(params_data['cv'].unique())

    def true_theta(cv):
        mean_latent = jnp.array([0.7, 0.5])
        if residual:
            cov_latent = jnp.array([[0.07, 0], [0, 0.05]])
            var_residual = (cv/100)**2 * jnp.exp(0.5*2 + 0.05)
        else:
            cov_latent = (cv/100)**2 * jnp.array([[0.7**2, 0], [0, 0.5**2]])
            var_residual = 0.1**2
        return jnp.concatenate([
            mean_latent,
            jnp.diag(cov_latent),
            jnp.array([var_residual])
        ])

    true_values = pd.concat(
        [
            pd.DataFrame({
                "Parameter": thetanames,
                "true_value": true_theta(cv),
                "cv": cv
            })
            for cv in cv_list
        ],
        ignore_index=True
    )

    df = (params_last
          .merge(true_values, on=["Parameter", "cv"])
          .assign(bias=lambda d: d['value'] - d['true_value'],
                  absbias=lambda d: (d['value'] - d['true_value']).abs(),
                  relativebias=lambda d:
                      (d['value'] - d['true_value']).abs() / d['true_value'])
          )

    bias_df = df.groupby(['n', 'cv', 'Parameter'], as_index=False)[
        'absbias'].mean()
    r_bias_df = df.groupby(['n', 'cv', 'Parameter'], as_index=False)[
        'relativebias'].mean()
    std_df = df.groupby(['n', 'cv', 'Parameter'], as_index=False)['value'].std()

    def facet(data, y, title_suffix, filter_fun=lambda d: d,
              ylim=None, fname="out.pdf"):
        g = sns.FacetGrid(filter_fun(data), col="Parameter",
                          col_wrap=3,
                          height=figsize[0], sharex=False, sharey=False)
        g.set_titles("")
        g.map_dataframe(sns.lineplot, x="n", y=y,
                        hue="cv", style="cv", marker="o", palette=palette)
        for i, ax in enumerate(g.axes.flat):
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.grid(axis='y', linestyle=':', alpha=0.6)
            ax.set_xlabel("N (nb individuals)", fontsize=12)
            ax.tick_params(axis='both', labelsize=12)
            secax = ax.secondary_xaxis(
                'top', functions=(invert, invert))
            secax.set_xticks([25, 12, 8, 6])
            secax.set_xlabel("n (nb obs per individual)",
                             fontsize=12, labelpad=10)
            secax.spines['top'].set_linestyle('--')
            secax.spines['top'].set_linewidth(1)
            secax.tick_params(axis='x', direction='in', length=4)
            if ylim:
                ax.set(ylim=ylim)
            if i % g._ncol == 0:
                ylabel = "Bias" if y != "value" else "Standard deviation"
                ax.set_ylabel(ylabel, fontsize=12)
        for ax, title in zip(g.axes.flat, g.col_names):
            ax.set_title(title.capitalize(), fontsize=15, loc='left')
        g.add_legend(title="CV (%)")
        g.legend.set_bbox_to_anchor((1.05, 0.5))
        g.legend.set_frame_on(False)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        g.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    tag = "residual" if residual else "rand_eff"

    mean_params = [r'$\mu_{\lambda}$', r'$\mu_{h}$']
    facet(bias_df[bias_df["Parameter"].isin(mean_params)], "absbias",
          "mean_bias", fname=f"bias_vs_n_by_cv_{tag}_mean.pdf")
    facet(bias_df[~bias_df["Parameter"].isin(mean_params)], "absbias",
          "var_bias",
          filter_fun=lambda d: d[d["absbias"] <= 1] if residual else d,
          fname=f"bias_vs_n_by_cv_{tag}_var.pdf")
    facet(std_df[std_df["Parameter"].isin(mean_params)], "value",
          "mean_std", fname=f"std_vs_n_by_cv_{tag}_mean.pdf")
    facet(std_df[~std_df["Parameter"].isin(mean_params)], "value",
          "var_std",
          filter_fun=lambda d: d[d["value"] <= 1] if residual else d,
          fname=f"std_vs_n_by_cv_{tag}_var.pdf")


def generate_bic_table(path, residual):

    # path = pathlib.Path(path)
    residual_str = "residual" if residual else "rand_eff"

    # cv_regex = re.compile(fr"_{residual_str}_cv(\d+)\.npy$")
    cv_regex = re.compile(fr"{residual_str}_cv(\d+)_allres.*")
    cv_list = sorted(
        int(m.group(1))
        for f in os.listdir(path)
        if (m := cv_regex.search(f))
    )

    all_files = [f for f in os.listdir(path)]
    all_bic_correct, all_bic_wrong = [], []

    for cv in cv_list:
        suffix = fr"_{residual_str}_cv{cv}\.npy$"

        patt_c = re.compile(fr"modchoice_misspenoise_biccorrect.*{suffix}")
        patt_w = re.compile(fr"modchoice_misspenoise_bicwrong.*{suffix}")

        df_c = pd.concat([loadBIC(os.path.join(path, fname))
                         for fname in all_files if patt_c.match(fname)])
        df_w = pd.concat([loadBIC(os.path.join(path, fname))
                         for fname in all_files if patt_w.match(fname)])

        df_c["cv"] = cv
        df_w["cv"] = cv

        all_bic_correct.append(df_c)
        all_bic_wrong.append(df_w)

    bic_correct = pd.concat(all_bic_correct, ignore_index=True).rename(
        columns={"BIC": "BIC correct"}
    )
    bic_wrong = pd.concat(all_bic_wrong,   ignore_index=True).rename(
        columns={"BIC": "BIC wrong"}
    )

    bic_diff = (
        bic_correct.drop(columns=["Model"])
        .merge(bic_wrong.drop(columns=["Model"]), on=["N", "cv"])
    )

    bic_diff = bic_diff[
        (~np.isinf(bic_diff["BIC correct"])) & (
            ~np.isinf(bic_diff["BIC wrong"]))
    ].copy()

    bic_diff["diffBICneg"] = bic_diff["BIC correct"] - bic_diff["BIC wrong"] < 0

    stat_bic = (
        bic_diff.groupby(["cv", "N"])["diffBICneg"]
        .mean()
        .reset_index()
        .pivot(index="cv", columns="N", values="diffBICneg")
        .sort_index(axis=1)
    )

    latex_code = stat_bic.to_latex(
        float_format="%.2f",
        caption=(
            "Proportion of datasets for which the smallest BIC "
            "is associated with the correct model (varying coefficient of variation)."
        ),
        label="tab:bic_comparison_cv",
    )

    return stat_bic, latex_code
