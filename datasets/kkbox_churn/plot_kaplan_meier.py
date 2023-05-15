# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.lines import Line2D
from pycox.datasets import kkbox_v1
from sksurv.nonparametric import kaplan_meier_estimator

DIR2FIGURES = Path("img")
DIR2FIGURES.mkdir(exist_ok=True)

# %%
# TODO: replace the following line with the result of Olivier's preprocessing
df = kkbox_v1.read_df()
print(df.columns)
print(df.shape)
df.head()
df["event"] = df["event"].astype(bool)
# %%
print(
    "🤨 What is the duration of a second subscription ?"
    "It does not count the first subscription."
)
user_id = "bjUdTNeHxtRHT71oi1RQbLSz7cUcjhTnT63p0yx3ktc="
print(
    df.loc[
        df["msno"] == user_id, ["duration", "log_days_since_reg_init", "no_prev_churns"]
    ]
)
# %%
# scikit-survival KM vs lifelines KM ?
fig, ax = plt.subplots(1, 1)
time, survival_prob = kaplan_meier_estimator(df["event"], df["duration"])
sksurv_km_plot = ax.step(
    time,
    survival_prob,
    where="post",
    color="orange",
    linestyle="dashed",
)

km_estimator = KaplanMeierFitter()
km_estimator.fit(df["duration"], df["event"])
km_estimator.plot(ax=ax, color="blue", linestyle="dashed", legend=False)

labels_handles = {
    "sksurv": Line2D([0], [0], color="orange", lw=4, linestyle="dashed"),
    "lifelines": Line2D([0], [0], color="blue", lw=4),
}
ax.add_artist(plt.legend(labels_handles.values(), labels_handles.keys()))
ax.set_xlabel("time $days$")
ax.set_ylabel("est. probability of survival $\hat{S}(days)$")
plt.savefig(
    DIR2FIGURES / "kaplan_meier_package_compare.png",
)
# %%
print(
    r"""
      KM [GreenWood Confidence interval](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) are drawn but are tiny, eg. with n_samples = 10 0000
      """
)
km_estimator_small = KaplanMeierFitter()
km_estimator_small.fit(df.iloc[:1000]["duration"], df.iloc[:1000]["event"])
km_estimator_small.plot(label="lifelines KM, 1000 samples")
plt.savefig(
    DIR2FIGURES / "kaplan_meier_small_samples.png",
)

# %%
"""
Can we cluster on specific features to caracterize one dimensional KM survival curve ?
eg: 
    - gender (3 categories)
    - no_prev_churns (2 categories)
    - is_auto_renew (2 categories)
    - registerd_via (4 main categories + [other + NA])
    - age_at_start (5 categories)
    - city (20 categories)
"""
# add age_category
age_bins = [-1, 0, 20, 30, 40, np.inf]
df = df.assign(age_at_start_bin=pd.cut(df["age_at_start"], bins=age_bins))

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

sub_km_categories = {
    "gender": df["gender"].unique(),
    "is_auto_renew": df["is_auto_renew"].unique(),
    "no_prev_churns": df["no_prev_churns"].unique(),
    "registered_via": [9, 7, 3, 4, [13, 10, 16, 11]],
    "city": df["city"].unique(),
    "age_at_start_bin": df["age_at_start_bin"].unique(),
}

for (category_name, category_uniques), ax in zip(sub_km_categories.items(), axes.flat):
    for cat_ in category_uniques:
        if type(cat_) == list:
            cat_df = df.loc[(df[category_name].isin(cat_)) | (df[category_name].isna())]
            cat_label = str(cat_ + ["nan"])
        elif pd.isna(cat_):
            cat_df = df.loc[df[category_name].isna()]
            cat_label = "nan"
        else:
            cat_df = df.loc[df[category_name] == cat_]
            cat_label = str(cat_)

        km_estimator_ = KaplanMeierFitter()
        km_estimator_.fit(cat_df["duration"], cat_df["event"], label=str(cat_label))
        km_estimator_.plot_survival_function(ax=ax)
        ax.legend(title=category_name)
plt.savefig(
    DIR2FIGURES / "kaplan_meier_categorical.pdf",
)
plt.savefig(
    DIR2FIGURES / "kaplan_meier_categorical.png",
)
