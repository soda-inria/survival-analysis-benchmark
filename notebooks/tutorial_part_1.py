# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Survival Analysis Tutorial Part 1
#
# [I. What is time-censored data?](#I.-What-is-time-censored-data?) <br>
# [II. Single event survival analysis with Kaplan-Meier](#II.-Single-event-survival-analysis-with-Kaplan-Meier) <br>
# [III. Calibration using the integrated brier score (IBS)](#III.-Calibration-using-the-integrated-brier-score-(IBS)) <br>
# [IV. Predictive Survival Analysis](#IV.-Predictive-survival-analysis) <br>
# [V. Competing risks modeling with Aalen-Johanson](#V.-Competing-risks-modeling-with-Aalen-Johanson) <br>
# [VI. Cumulative incidence function (CIF) using our GradientBoostedCIF](#VI.-Cumulative-incidence-function-(CIF)-using-our-GradientBoostedCIF) <br>

# %% [markdown]
# ## I. What is time-censored data?
#
# ### I.1 Censoring
#
# Survival analysis is a time-to-event regression problem, with censored data. We call censored all individuals that didn't experience the event during the range of the observation window.
#
# In our setting, we're mostly interested in right-censored data, meaning we either lost track of the individual, or the study ended before the event of the individual could be observed.
#
# <figure>
# <img src="censoring.png" style="width:80%">
# <figcaption align = "center"> <i>image credit: scikit-survival</i> </figcaption>
# </figure>
#
# Individuals can join the study at the same or different times, and the study may or may not be ended by the time of observation.
#
# Survival analysis techniques have wide applications. In the medical landscape, events can consist in patients dying of cancer, or on the contrary recovering from some disease. In predictive maintenance, events can model machine failure. In marketing, we can consider user churning as events, or we could focus on users becoming premium.
#
#
# ### I.2 Our target `y`
#
# For each individual $i\in[1, N]$, our survival analysis target $y_i$ is comprised of two elements:
#
# - The event $\delta_i\in\{0, 1\}$, where $0$ is censoring and $1$ is experiencing the event.
# - The censored time-to-event $d_i=min(t_{i}, c_i) > 0$, that is the minimum between the date of the experienced event $t_i$ and the censoring date $c_i$. In a real-world setting, we don't have direct access to $t_i$ when $\delta_i=0$.
#
# Here is how we represent our target:

# %%
import pandas as pd

df = pd.read_parquet("data_truck_no_covariates.parquet")
df[["event", "duration"]].head()

# %% [markdown]
# In this exemple, we study the accident of truck-driver pairs. Censored pairs (when event is False) haven't had a mechanical failure or an accident during the study.

# %% [markdown]
# ### I.3 Why is it a problem to train time-to-event regression models?
#
# Without survival analysis, we have two options when confronting censored data:
# - We ignore them, by only keeping events that happened and performing naive regression on them.
# - We consider that all censored events happen at the end of our observation window.
#
# **Both approaches are wrong and lead to biased results.**
#
# Let's compute the average duration yielded by both approaches on our truck dataset. We will compare them to the mean of the ground-truth event time $T$, that we would obtained with an infinite observation window. 
#
# Note that we have access to the random variable $T$ because we generated this synthetic dataset. With real-world data, you only have access to $Y = \min(T, C)$, where $C$ is a random variable representing the censoring time.

# %%
stats_1 = (
    df.loc[df["event"]]["duration"]
    .apply(["mean", "median"])
)
print(
    f"Biased method 1 - mean: {stats_1['mean']:.2f} days, "
    f"median: {stats_1['median']:.2f} days"
)

# %%
import numpy as np

max_duration = df["duration"].max()
stats_2 = (
    pd.Series(
        np.where(df["event"], df["duration"], max_duration)
    )
    .apply(["mean", "median"])
)
print(
    f"Biased method 2 - mean: {stats_2['mean']:.2f} days, "
    f"median: {stats_2['median']:.2f} days"
)

# %%
true_stats = df["ground_truth_duration"].apply(["mean", "median"])
print(
    f"Ground truth - mean: {true_stats['mean']:.2f} days, "
    f"median: {true_stats['median']:.2f} days"
)

# %% [markdown]
# We see that none of this naive methods gives a good estimate of the ground truth. A naive regression would try to estimate $\mathbb{E}[T|X]$, where $X$ are our covariates, but we only have access to $Y = \min(T, C)$.

# %% [markdown]
# ## II. Single event survival analysis with Kaplan Meier
#
# We now introduce the survival analysis approach to the problem of estimating the time-to-event from censored data. For now, we ignore any information from $X$ and focus on $y$ only.
#
# Here our quantity of interest is the survival probability:
#
# $$S(t)=P(T > t)$$ 
#
# This represents the probability that an event doesn't occur at or before some given time $t$, i.e. that it happens at some time $T > t$.
#
# The most commonly used method to estimate this function is the **Kaplan Meier** estimator. It gives us an unbiased estimate of the survival probability, ignoring any information available in $X$.
#
# $$\hat{S}(t)=\prod_{i: t_i\leq t} (1 - \frac{d_i}{n_i})$$
#
# Where $t_i$ is the time of event for individual $i$ that experienced the event, $d_i$ is the number of individuals having experienced the event at $t_i$, and $n_i$ are the remaining individuals at risk at $t_i$. Note that individuals that were censored before $t_i$ are no longer considered at risk at $t_i$.
#
# In real-world application, we aim at estimating $\mathbb{E}[T]$ or $Q_{50\%}[T]$. The latter quantity represents the median survival duration i.e. the duration before 50% of our population at risk experiment the event. We can also be interested in estimating the survival probability after some reference time $P(T > t_{ref})$, e.g. a random clinical trial estimating the capacity of a drug to improve the survival probability after 6 months.

# %%
import plotly.express as px
from sksurv.nonparametric import kaplan_meier_estimator


times, survival_probas = kaplan_meier_estimator(df["event"], df["duration"])

km_proba = pd.DataFrame(dict(time=times, survival_proba=survival_probas))
fig = px.line(
    km_proba,
    x="time",
    y="survival_proba",
    title="Kaplan-Meier survival probability",
)
fig.add_hline(
    y=0.50,
    annotation_text="Median",
    line_dash="dash",
    line_color="red",
    annotation_font_color="red",
)

fig.update_layout(
    height=500,
    width=800,
    xaxis_title="time (days)",
    yaxis_title="$\hat{S}(t)$",
    yaxis_range=[0, 1],
)


# %% [markdown]
# We can read the median time to event directly from this curve, in this case close to 1020 days.
# Note that since we have censored data, $S(t)$ doesn't reach 0 within our observation window and we have residuals of 30%.

# %% [markdown]
# ***Exercice*** <br>
# Based on `times` and `survival_proba`, estimate the median survival time.
# *Hint: Use `np.searchsorted`*.

# %%
def get_median_survival_proba(times, survival_proba):
    """Get the closest time to a survival proba of 50%.
    """
    ### Your code here
    median_survival_proba_time = 0
    ###
    return median_survival_proba_time


# %%
get_median_survival_proba(times, survival_probas)

# %% [markdown]
# <details><summary><i><b>Solution</b></i></summary>
# <br>
#     
# ```python
# def get_median_survival_proba(times, survival_proba):
#     """Get the closest time to a survival proba of 50%.
#     """
#     # Search sorted needs an ascending ordered array.
#     sorted_survival_proba = survival_proba[::-1]
#     median_idx = np.searchsorted(sorted_survival_proba, 0.50)
#     median_survival_proba_time = times[-median_idx]
#     return median_survival_proba_time
# ```
#     
# </details>

# %% [markdown]
# We can enrich our analysis by introducing covariates, that are statistically associated to the events and durations.

# %%
df = pd.read_parquet("data_truck.parquet")
df["event"] = df["event"] > 0
df

# %% [markdown]
# For exemple, let's use Kaplan Meier to get a sense of the impact of the **brand**, by stratifying on this variable.
#
# ***Exercice*** <br>
# Plot the stratified Kaplan Meier of the brand, i.e. for each different brand:
# 1. Filter the dataset on this brand
# 2. Estimate the survival probability with Kaplan Meier
# 3. Subplot this survival probability.
#
# What are the limits of this method?

# %%
import plotly.graph_objects as go


brands = df["brand"].unique()
fig_data = []
for brand in brands:
    ### Your code here
    pass
    ###
go.Figure(fig_data)

# %% [markdown]
# <details><summary><i><b>Solution</b></i></summary>
# <br>
#     
# ```python
# import plotly.graph_objects as go
#
# brands = df["brand"].unique()
# fig_data = []
# for brand in brands:
#     df_brand = df.loc[df["brand"] == brand]
#     times_, survival_probas_ = kaplan_meier_estimator(df_brand["event"], df_brand["duration"])
#     fig_data.append(
#         go.Scatter(x=times_, y=survival_probas_, name=brand)
#     )
# go.Figure(fig_data)
# ```
#
# <br>
# The stratified method quickly become impracticable as the covariate groups grow. We need estimator that can handle covariates.
#     
# </details>

# %% [markdown]
# Next, we'll study how to add covariates $X$ to our analysis.

# %% [markdown]
# ## III. Calibration using the integrated brier score (IBS)

# %% [markdown]
# The Brier score is a proper scoring rule that measures the calibration of our survival probability predictions. It is comprised between 0 and 1 (lower is better).
# It answers the question "how close to the real probabilities are our estimates?". A good calibration makes our predictions easier to explain.

# %% [markdown]
# <details><summary>Mathematical formulation</summary>
#     
# $$\mathrm{BS}^c(t) = \frac{1}{n} \sum_{i=1}^n I(d_i \leq t \land \delta_i = 1)
#         \frac{(0 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(d_i)} + I(d_i > t)
#         \frac{(1 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(t)}$$
#     
# In the survival analysis context, the Brier Score can be seen as the Mean Squared Error (MSE) between our probability $\hat{S}(t)$ and our target label $\delta_i \in {0, 1}$, weighted by the inverse probability of censoring $\frac{1}{\hat{G}(t)}$.
# - When no event or censoring has happened at $t$ yet, i.e. $I(d_i > t)$, we penalize a low probability of survival with $(1 - \hat{S}(t|\mathbf{x}_i))^2$.
# - Conversely, when an individual has experienced an event before $t$, i.e. $I(d_i \leq t \land \delta_i = 1)$, we penalize a high probability of survival with $(0 - \hat{S}(t|\mathbf{x}_i))^2$.
#     
# <figure>
# <img src="BrierScore.svg" style="width:80%">
# </figure>
#     
# </details>

# %%
times, survival_proba = kaplan_meier_estimator(df["event"], df["duration"])


# %%
def make_target(event, duration):
    """Return scikit-survival's specific target format.
    """
    y = np.empty(
        shape=event.shape[0],
        dtype=[("event", bool), ("duration", float)],
    )
    y["event"] = event
    y["duration"] = duration
    
    return y


# %%
from sksurv.metrics import brier_score
from sksurv.functions import StepFunction

# Create a callable function from times and survival proba.
survival_func = StepFunction(times, survival_proba)

# Bound `times` to the range of `duration`.
# This is needed to compute the brier score.
times = np.linspace(
    df["duration"].min(),
    df["duration"].max() - 1,
    num=100,
)

# Call the function with the new `times` variable and
# get the matching survival proba.
survival_proba = survival_func(times)

# Stack `N` survival proba vectors to simulate predictions
# for all individuals.
n_samples = df.shape[0]
km_survival_proba_matrix = np.vstack([survival_proba] * n_samples)

# Adapt the event and duration to scikit-survival specific
# numpy array target.
y = make_target(df["event"], df["duration"])

_, km_brier_scores = brier_score(
    survival_train=y,
    survival_test=y,
    estimate=km_survival_proba_matrix,
    times=times,
)

# %%
from matplotlib import pyplot as plt
import seaborn as sns; sns.set_style("darkgrid")

plt.plot(times, km_brier_scores);
plt.title("Brier score of Kaplan Meier estimation (lower is better)");
plt.xlabel("time (days)");

# %% [markdown]
# Additionnaly, we compute the Integrated Brier Score (IBS) which we will use to rank estimators:
# $$IBS = \frac{1}{t_{max} - t_{min}}\int^{t_{max}}_{t_{min}} BS(t) dt$$

# %%
from sksurv.metrics import integrated_brier_score

km_ibs = integrated_brier_score(
    survival_train=y,
    survival_test=y,
    estimate=km_survival_proba_matrix,
    times=times,
)
km_ibs

# %% [markdown]
# Finally, let's also introduce the concordance index (C-index). This metric evaluates the discriminative power of a model by comparing pairs of individuals having experienced the event. The C-index of a pair $(i, j)$ is maximized when individual $i$ has experienced the event before $j$ and the estimated risk of $i$ is higher than the one of $j$. 
#
# This metric is also comprised between 0 and 1 (higher is better), 0.5 corresponds to a random prediction.
#
# <details><summary>Mathematical formulation</summary>
#     
# $$\mathrm{C_{index}} = \frac{\sum_{i,j} I(d_i < d_j \space \land \space \delta_i = 1 \space \land \space \mu_i < \mu_j)}{\sum_{i,j} I(d_i < d_j \space \land \space \delta_i = 1)}$$
#
# Let's introduce the cumulative hazards $\Lambda(t)$, which is the negative log of the survival function $S(t)$:
#
# $$S(t) = \exp(-\Lambda(t)) = \exp(-\int^t_0 \lambda(u)du)$$
#     
# Therefore:
#     
# $$\Lambda(t) = -\log(S(t))$$
#
# Finally, the risk is obtained by summing over the entire cumulative hazard:
#     
# $$\mu_i = \int^{t_{max}}_{t_{min}} \Lambda(t, x_i) dt = \int^{t_{max}}_{t_{min}} - \log (S(t, x_i)) dt$$
#     
# </details>

# %% [markdown]
# To compute the C-index of our Kaplan Meier estimates, we assign every individual with the same survival probabilities given by the Kaplan Meier.

# %%
from sksurv.metrics import concordance_index_censored


def get_c_index(event, duration, survival_proba_matrix):
    if survival_proba_matrix.ndim != 2:
        raise ValueError(
            "`survival_probas` must be a 2d array of "
            "shape (n_samples, times)."
        )
    # Cumulative hazard is also known as risk.
    cumulative_hazard = survival_to_risk_estimate(survival_proba_matrix)
    metrics = concordance_index_censored(event, duration, cumulative_hazard)
    return metrics[0]


def survival_to_risk_estimate(survival_proba_matrix):
    return -np.log(survival_proba_matrix + 1e-8).sum(axis=1)


# %%
km_c_index = get_c_index(df["event"], df["duration"], km_survival_proba_matrix)
km_c_index

# %% [markdown]
# This is equivalent to a random prediction. Indeed, as our Kaplan Meier is a descriptive statistics, it can't be used to rank individuals predictions.

# %% [markdown]
# ## IV. Predictive survival analysis
#
# We now introduce some quantities which are going to be at the core of our predictions.
#
# The most important concept is the hazard rate $\lambda(t)$. This quantity represents the "speed of failure" or the probability that an event occurs in the next $dt$, given that it hasn't occured yet. This can be written as:
#
# $$\begin{align}
# \lambda(t) &=\lim_{dt\rightarrow 0}\frac{P(t \leq T < t + dt | P(T \geq t))}{dt} \\
# &= \lim_{dt\rightarrow 0}\frac{P(t \leq T < t + dt)}{dtS(t)} \\
# &= \frac{f(t)}{S(t)}
# \end{align}
# $$
#
# where $f(t)$ represents the probability density. This quantity estimates the probability that an event occurs in the next $dt$, independently of this event having happened before. <br>
# If we integrate $f(t)$, we found the cumulative incidence function (CIF) $F(t)=P(T < t)$, which is the opposite of the survival function $S(t)$:
#
# $$F(t)=\int^\infty_0f(t)dt=1-S(t)$$

# %% [markdown]
# ### IV.1 Cox Proportional Hazards
#
# The Cox PH model is the canonical way of dealing with covariates $X$ in survival analysis. It computes a log linear regression on the target $Y = \min(T, C)$, and consists in a baseline term $\lambda_0(t)$ and a covariate term with weights $\beta$.
# $$\lambda(t, x_i) = \lambda_0(t) \exp(x_i^\top \beta)$$
#
# Note that only the baseline depends on the time $t$, but we can extend Cox PH to time-dependent covariate $x_i(t)$ and time-dependent weigths $\beta(t)$. We won't cover these extensions in this tutorial.
#
# This methods is called ***proportional*** hazards, since for two different covariate vectors $x_i$ and $x_j$, their ratio is:
# $$\frac{\lambda(t, x_i)}{\lambda(t, x_j)} = \frac{\lambda_0(t) e^{x_i^\top \beta}}{\lambda_0(t) e^{x_j^\top \beta}}=\frac{e^{x_i^\top \beta}}{e^{x_j^\top \beta}}$$
#
# This ratio is not dependent on time, and therefore the hazards are proportional.
#
# Let's run it on our truck-driver dataset.

# %%
df

# %%
duration = df.pop("duration")
event = df.pop("event")
y = make_target(event, duration)
X = df

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape

# %% [markdown]
# ***Exercice***
#
# Create a `ColumnTransformer` to encode categories `brand` and `model_id`.
#
# *Hint*: Use `sklearn.preprocessing.OneHotEncoder` and `sklearn.compose.make_column_transformer`.

# %%
### Your code here
transformer = None
###

# %% [markdown]
# <details><summary><b><i>Solution</i></b></summary>
# <br>
#
# ```python
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder
#
# transformer = make_column_transformer(
#     (OneHotEncoder(), ["brand", "model_id"]),
#     remainder="passthrough",
# )
# ```
# </details>

# %%
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis

cox_ph = make_pipeline(
    transformer,
    CoxPHSurvivalAnalysis(alpha=1e-4)
)
cox_ph.fit(X_train, y_train)

# %%
step_funcs = cox_ph.predict_survival_function(X_test)

fig, ax = plt.subplots()
for idx, step_func in enumerate(step_funcs[:5]):
    survival_proba = step_func(times)
    ax.plot(times, survival_proba, label=idx)
ax.set(
    title="Survival probabilities $\hat{S(t)}$ of CoxPH",
    xlabel="time (days)",
    ylabel="S(t)",
)
plt.legend();

# %%
X_test.head().reset_index(drop=True)

# %% [markdown]
# We see that we can get some intuition about the features importance from the first 5 truck-driver pairs and their survival probabilities.

# %% [markdown]
# ***Exercice***
#
# Plot the feature importance $\beta$ of the model (stored under `_coef`) with their names from the `get_feature_names_out()` method of the transformer.
#
# *Hint*: You can access an element of a pipeline as simply as `pipeline[idx]`.

# %%
### Your code here
feature_names = []
weight = []
###

features = pd.DataFrame(
    dict(
        feature_name=feature_names,
        weight=weight,
    )
)
ax = sns.barplot(features, y="feature_name", x="weight", orient="h")

# %% [markdown]
# <details><summary><i><b>Solution</b></i></summary>
# <br>
#
# ```python
# feature_names = cox_ph[0].get_feature_names_out()
# weights = cox_ph[-1].coef_
#
# features = (
#     pd.DataFrame(dict(
#         feature_name=feature_names,
#         weight=weights,
#     ))
#     .sort_values("weight", ascending=False)
# )
# ax = sns.barplot(features, y="feature_name", x="weight", orient="h")
# ax.set_title("Cox PH feature importance of $\lambda(t)$");
# ```
#
# </details>

# %% [markdown]
# Finally, we compute the Brier score for our model.

# %%
from sksurv.metrics import brier_score, integrated_brier_score

test_times = np.linspace(
    y_test["duration"].min(),
    y_test["duration"].max() - 1,
    num=100,
)
cox_survival_proba_matrix = np.vstack([step_func(test_times) for step_func in step_funcs])

_, cox_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=cox_survival_proba_matrix,
    times=test_times,
)

cox_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=cox_survival_proba_matrix,
    times=test_times,
)

fig, ax = plt.subplots()
ax.plot(times, cox_brier_scores, label="CoxPH")
ax.plot(times, km_brier_scores, label="KaplanMeier")
ax.set(
    title="Brier Scores of Cox PH",
    xlabel="time (days)",
    ylabel="$BS(t)^c$",
)
plt.legend();

print(f"KaplanMeier IBS: {km_ibs:.4f}")
print(f"CoxPH IBS: {cox_ibs:.4f}")

# %%
cox_c_index = get_c_index(
    y_test["event"],
    y_test["duration"],
    cox_survival_proba_matrix,
)

print(f"Kaplan Meier C-index: {km_c_index:.4f}")
print(f"Cox PH C-index: {cox_c_index:.4f}")

# %% [markdown]
# We have slightly improved upon the Kaplan Meier.

# %% [markdown]
# ### IV.2 Random Survival Forest

# %%
from sksurv.ensemble import RandomSurvivalForest

rsf = make_pipeline(
    transformer,
    RandomSurvivalForest(n_estimators=10, max_depth=4, n_jobs=-1),
)
rsf.fit(X_train, y_train)

# %%
step_funcs = rsf.predict_survival_function(X_test)

fig, ax = plt.subplots()
for idx, step_func in enumerate(step_funcs[:5]):
    survival_proba = step_func(test_times)
    ax.plot(test_times, survival_proba, label=idx)
ax.set(
    title="Survival probabilities $\hat{S}(t)$ of Random Survival Forest",
    xlabel="time (days)",
    ylabel="S(t)",
)
plt.legend();

# %%
rsf_survival_proba_matrix = np.vstack(
    [step_func(test_times) for step_func in step_funcs]
)

_, rsf_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=rsf_survival_proba_matrix,
    times=test_times,
)

rsf_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=rsf_survival_proba_matrix,
    times=test_times,
)

fig, ax = plt.subplots()
ax.plot(test_times, rsf_brier_scores, label="RandomSurvivalForest")
ax.plot(test_times, cox_brier_scores, label="CoxPH")
ax.plot(test_times, km_brier_scores, label="KaplanMeier")
ax.set(
    title="Brier Scores",
    xlabel="time (days)",
    ylabel="$BS(t)^c$",
)
plt.legend();

print(f"KaplanMeier IBS: {km_ibs:.4f}")
print(f"CoxPH IBS: {cox_ibs:.4f}")
print(f"RandomSurvivalForest IBS: {rsf_ibs:.4f}")

# %%
rsf_c_index = get_c_index(y_test["event"], y_test["duration"], rsf_survival_proba_matrix)

print(f"Random Survival Index C-index: {rsf_c_index:.4f}")
print(f"Cox PH C-index: {cox_c_index:.4f}")
print(f"Kaplan Meier C-index: {km_c_index:.4f}")

# %% [markdown]
# ### IV.3 GradientBoostedIBS

# %%
import sys; sys.path.append("..")
from models.yasgbt import YASGBTRegressor
from model_selection.wrappers import PipelineWrapper

gb_model = make_pipeline(
    transformer,
    YASGBTRegressor(n_iter=20, learning_rate=0.2),
)
gb_model = PipelineWrapper(gb_model)
gb_model.fit(X_train, y_train, times)

# %%
gbcif_survival_proba_matrix = gb_model.predict_survival_function(X_test, times)

_, gbcif_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=gbcif_survival_proba_matrix,
    times=test_times,
)

gbcif_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=gbcif_survival_proba_matrix,
    times=test_times,
)

fig, ax = plt.subplots()
ax.plot(test_times, gbcif_brier_scores, label="GradientBoostedCIF")
ax.plot(test_times, rsf_brier_scores, label="RandomSurvivalForest")
ax.plot(test_times, cox_brier_scores, label="CoxPH")
ax.plot(test_times, km_brier_scores, label="KaplanMeier")
ax.set(
    title="Brier Scores",
    xlabel="time (days)",
    ylabel="$BS(t)^c$",
)
plt.legend();

print(f"GradientBoostedCIF IBS: {gbcif_ibs:.4f}")
print(f"RandomSurvivalForest IBS: {rsf_ibs:.4f}")
print(f"CoxPH IBS: {cox_ibs:.4f}")
print(f"KaplanMeier IBS: {km_ibs:.4f}")

# %%
gbcif_c_index = get_c_index(y_test["event"], y_test["duration"], gbcif_survival_proba_matrix)

print(f"GradientBoostedCIF C-index: {gbcif_c_index:.4f}")
print(f"Random Survival Index C-index: {rsf_c_index:.4f}")
print(f"Cox PH C-index: {cox_c_index:.4f}")
print(f"Kaplan Meier C-index: {km_c_index:.4f}")

# %% [markdown]
# ## V. Competing risks modeling with Aalen-Johanson

# %%

# %%

# %% [markdown]
# ## VI. Cumulative incidence function (CIF) using our GradientBoostedCIF

# %%
