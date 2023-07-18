import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# Load the penguins dataset
df = sns.load_dataset("penguins")
print(type(df))

# Draw a categorical scatterplot to show each observation
ax = sns.swarmplot(data=df, x="body_mass_g", y="sex", hue="species")
ax.set(ylabel="")
ax

df = pd.DataFrame(
    {
        "Target": y_test,
        "model_svc_default_predict": model_svc_default_predict,
        "grid_model_svc_linear_kernel_predict": grid_model_svc_linear_kernel_predict,
        "grid_model_svc_poly_kernel_predict": grid_model_svc_poly_kernel_predict,
        "grid_model_svc_rbf_kernel_predict": grid_model_svc_rbf_kernel_predict,
        "grid_model_svc_sigmoid_kernel_predict": grid_model_svc_sigmoid_kernel_predict,
        "model_linear_svc_default_predict": model_linear_svc_default_predict,
        "grid_model_linear_svc_predict": grid_model_linear_svc_predict,
        "model_nu_svc_default_predict": model_nu_svc_default_predict,
        "grid_model_nu_svc_linear_kernel_predict": grid_model_nu_svc_linear_kernel_predict,
        "grid_model_nu_svc_poly_kernel_predict": grid_model_nu_svc_poly_kernel_predict,
        "grid_model_nu_svc_rbf_kernel_predict": grid_model_nu_svc_rbf_kernel_predict,
        "grid_model_nu_svc_sigmoid_kernel_predict": grid_model_nu_svc_sigmoid_kernel_predict
    }
)
