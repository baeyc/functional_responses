import utils


# =============================================================================
# Plot graphs
# =============================================================================

# Replace 'my-path' with the actual path to the directory containing the results files
path = 'my-path'

# Plot bias and standard deviation of estimators when changing the variability of the random effects
utils.plot_bias_std(path, residual=False)
# Plot bias and standard deviation of estimators when changing the variability of the residuals
utils.plot_bias_std(path, residual=True)


# # ---------------------------------------
# # Generate model selection results tables
# # ---------------------------------------


table_bic_rand_eff, latex_rand_eff = utils.generate_bic_table(
    path, residual=False)
print(table_bic_rand_eff)

table_bic_resid, latex_resid = utils.generate_bic_table(path, residual=True)
print(table_bic_resid)
