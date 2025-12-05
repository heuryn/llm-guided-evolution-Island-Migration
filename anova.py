import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import pairwise_tukeyhsd

import seaborn as sns
import numpy as np

# ==============================
# 1. Load data
# ==============================
df = pd.read_csv("auc.csv")

# Rename columns to be formula-safe (no hyphens, no spaces)
df = df.rename(columns={
    "Prompt-Group": "Prompt_Group",
    "Island-Config": "Island_Config",
    # make sure this matches your response column name if it's different
    # "OldResultName": "Result"
})

# Drop rows with missing response
df = df.dropna(subset=["Result"])

# Make factors categorical
df["Prompt_Group"] = df["Prompt_Group"].astype("category")
df["Island_Config"] = df["Island_Config"].astype("category")

print("First few rows:")
print(df.head(), "\n")
print("Prompt_Group levels:", df["Prompt_Group"].cat.categories.tolist())
print("Island_Config levels:", df["Island_Config"].cat.categories.tolist(), "\n")

# ==============================
# 2. Fit 2-way ANOVA model
# ==============================
formula = "Result ~ C(Prompt_Group) * C(Island_Config)"
print("Using formula:", formula, "\n")

model = ols(formula, data=df).fit()




# ==============================
# 3. ANOVA table
# ==============================
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA table (Type II):")
print(anova_table, "\n")

# ==============================
# 4. Model summary
# ==============================
#print("Model summary:")
#print(model.summary())

# ==============================
# 5. Quick residual checks (optional)
# ==============================
plt.figure()
plt.hist(model.resid, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of residuals")
plt.show()

plt.figure()
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. fitted")
plt.show()


tukey = pairwise_tukeyhsd(
    endog=df["Result"],
    groups=df["Prompt_Group"],
    alpha=0.05
)

print(tukey)


sns.pointplot(data=df, x="Prompt_Group", y="Result", errorbar="se")
plt.title("Prompt Group Means ± SE")
plt.show()
