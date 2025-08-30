import utils

from pathlib import Path

# =============================================================================
# Plot graphs
# =============================================================================


# Ensure that "/results" contains the results files and is a subdirectory of the working directory.
# The working directory must contain this .py script.
# Example directory structure:
# /working_directory/
# ├── compute_results_variability_level.py
# └── results/

current_path = Path.cwd()
path = current_path / "results"


# results related to signal-to-noise ratio
utils.plot_bias_std(path, residual=True)
# results related to  random effects coefficient of variation
utils.plot_bias_std(path, residual=False)

# =============================================================================
# Generate model selection results tables
# =============================================================================


table_bic_rand_eff, latex_rand_eff = utils.generate_bic_table(
    path, residual=False)
print(table_bic_rand_eff)

table_bic_resid, latex_resid = utils.generate_bic_table(path, residual=True)
print(table_bic_resid)
