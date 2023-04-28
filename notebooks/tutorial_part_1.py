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
# In our setting, we're mostly interested in right-censored data, meaning we that the event of interest did not occur before the end of the observation period (typically the time of collection of the dataset):
#
# <figure>
# <img src="censoring.png" style="width:80%">
# <figcaption align = "center"> <i>image credit: scikit-survival</i> </figcaption>
# </figure>
#
# Individuals can join the study at the same or different times, and the study may or may not be ended by the time of observation.
#
# Survival analysis techniques have wide applications:
#
# - In the **medical** landscape, events can consist in patients dying of cancer, or on the contrary recovering from some disease.
# - In **predictive maintenance**, events can model machine failure.
# - In **insurance**, we are interesting in modeling the time to next claim for a portfolio of insurance contracts.
# - In **marketing**, we can consider user churning as events, or we could focus on users becoming premium.
#
#
# As we will see, for all those application, it is not possible to directly train a machine learning based regression model on such  **right-censored** time-to-event target since we only have a lower bound on the true time to event for some data points. **Naively removing such points from the data would cause the model predictions to be biased**.
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
import numpy as np

truck_failure_events = pd.read_parquet("truck_failure_10k_any_event.parquet")
truck_failure_events

# %% [markdown]
# In this exemple, we study the accident of truck-driver pairs. Censored pairs (when event is 0 or False) haven't had a mechanical failure or an accident during the study.

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
naive_stats_1 = (
    truck_failure_events.query("event == True")["duration"]
    .apply(["mean", "median"])
)
print(
    f"Biased method 1 - mean: {naive_stats_1['mean']:.2f} days, "
    f"median: {naive_stats_1['median']:.2f} days"
)

# %%
max_duration = truck_failure_events["duration"].max()
naive_stats_2 = (
    pd.Series(
        np.where(
            truck_failure_events["event"],
            truck_failure_events["duration"],
            max_duration,
        )
    )
    .apply(["mean", "median"])
)
print(
    f"Biased method 2 - mean: {naive_stats_2['mean']:.2f} days, "
    f"median: {naive_stats_2['median']:.2f} days"
)

# %% [markdown]
# Neither naive methods can estimate the true mean and median failure times. In our case, the data comes from a simple truck fleed model and we have access to the uncensored times (we can wait as long as we want to extend the observation period as needed to have all truck fail).
#
# Let's have a look at the true mean and median time-to-failure:

# %%
truck_failure_events_uncensored = pd.read_parquet("truck_failure_10k_any_event_uncensored.parquet")

# %%
true_stats = truck_failure_events_uncensored["duration"].apply(["mean", "median"])
print(
    f"Ground truth - mean: {true_stats['mean']:.2f} days, "
    f"median: {true_stats['median']:.2f} days"
)

# %% [markdown]
# We see that none of neither of the naive methods gives a good estimate of the ground truth.
#
# If we have access to covariates $X$ (also known as input features in machine learning), a naive regression method would try to estimate $\mathbb{E}[T|X]$, where $X$ are our covariates, but we only have access to $Y = \min(T, C)$ where $T$ is the true time to failure and $C$ is the censoring duration.

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
# The most commonly used method to estimate this function is the **Kaplan Meier** estimator. It gives us an **unbiased estimate of the survival probability**.
#
# $$\hat{S}(t)=\prod_{i: t_i\leq t} (1 - \frac{d_i}{n_i})$$
#
# Where:
#
# - $t_i$ is the time of event for individual $i$ that experienced the event,
# - $d_i$ is the number of individuals having experienced the event at $t_i$,
# - $n_i$ are the remaining individuals at risk at $t_i$.
#
# Note that **individuals that were censored before $t_i$ are no longer considered at risk at $t_i$**.
#
# Note that, contrary to machine learning regressors, this estimator is **unconditional**: it only extracts information from $y$ only, and cannot model information about each individual typically provided in a feature matrix $X$.
#
# In real-world application, we aim at estimating $\mathbb{E}[T]$ or $Q_{50\%}[T]$. The latter quantity represents the median survival duration i.e. the duration before 50% of our population at risk experiment the event. We can also be interested in estimating the survival probability after some reference time $P(T > t_{ref})$, e.g. a random clinical trial estimating the capacity of a drug to improve the survival probability after 6 months.

# %%
import plotly.express as px
from sksurv.nonparametric import kaplan_meier_estimator


times, survival_probs = kaplan_meier_estimator(
    truck_failure_events["event"], truck_failure_events["duration"]
)

km_proba = pd.DataFrame(dict(time=times, survival_probs=survival_probs))
fig = px.line(
    km_proba,
    x="time",
    y="survival_probs",
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
# Based on `times` and `survival_probs`, estimate the median survival time.
# *Hint: Use `np.searchsorted`*.

# %%
def get_median_survival_probs(times, survival_probs):
    """Get the closest time to a survival proba of 50%.
    """
    ### Your code here
    median_survival_probs_time = 0
    ###
    return median_survival_probs_time












get_median_survival_probs(times, survival_probs)


# %%
### Solution

def get_median_survival_probs(times, survival_probs):
    """Get the closest time to a survival proba of 50%.
    """
    # Search sorted needs an ascending ordered array.
    sorted_survival_probs = survival_probs[::-1]
    median_idx = np.searchsorted(sorted_survival_probs, 0.50)
    median_survival_probs_time = times[-median_idx]
    return median_survival_probs_time

get_median_survival_probs(times, survival_probs)

# %% [markdown]
# This should be an unbiased estimate of the median uncensored duration:

# %%
truck_failure_events_uncensored["duration"].median()

# %% [markdown]
# We can enrich our analysis by introducing covariates, that are statistically associated to the events and durations.

# %%
truck_failure_features = pd.read_parquet("truck_failure_10k_features.parquet")
truck_failure_features

# %%
truck_failure_features_and_events = pd.concat(
    [truck_failure_features, truck_failure_events], axis="columns"
)
truck_failure_features_and_events

# %% [markdown]
# For exemple, let's use Kaplan Meier to get a sense of the impact of the **brand**, by stratifying on this variable.
#
# ***Exercice***
#
# Plot the stratified Kaplan Meier of the brand, i.e. for each different brand:
# 1. Filter the dataset on this brand using pandas, for instance using the `.query` method of the dataframe;
# 2. Estimate the survival curve with Kaplan Meier on each brand subset;
# 3. Plot the survival curve for each subset.
#
# What are the limits of this method?

# %%
import matplotlib.pyplot as plt


def plot_brands_km(df):
    brands = df["brand"].unique()
    fig_data = []
    for brand in brands:
        # TODO: replace the following by your code here:
        pass

    plt.title("Survival curves by brand")

    
plot_brands_km(truck_failure_features_and_events)

# %% [markdown]
# **Solution**: click below to expand the cell:

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
import matplotlib.pyplot as plt


def plot_brands_km(df):
    brands = df["brand"].unique()
    fig_data = []
    for brand in brands:
        df_brand = df.query("brand == @brand")
        x, y = kaplan_meier_estimator(df_brand["event"], df_brand["duration"])
        plt.plot(x, y, label=brand)

    plt.legend()
    plt.title("Survival curves by brand")
    
plot_brands_km(truck_failure_features_and_events)

# %% [markdown]
# We can observe that drivers of "Cheapz" trucks seem to experiment a higher number of failures in the early days but then the cumulative number of failures for each group seem to become comparable. Very truck seem to operate after 2500 days (~7 years) without having experienced any failure.
#
# The stratified KM method is nice to compare two groups but quickly becomes impracticable as the number of covariate groups grow. We need estimator that can handle covariates.

# %% [markdown]
# Let's now attempt to quantify how a survival curve estimated on a training set performs on a test set.
#
# ## III. Survival model evaluation using the Integrated Brier Score (IBS) and the Concordance Index (C-index)

# %% [markdown]
# The Brier score and the C-index are measures that assess the quality of predicted survival curve on a sample of data. The Brier score is a proper scoring rule, meaning that a model has minimal Brier score if and only if it correctly estimates the true survival probabilities induced by the underlying data generating process. In that respect the **Brier score** assesses both the **calibration** and the **ranking power** of a survival probability estimator.
#
# On the other hand, the **C-index** only assesses the **ranking power**: it is invariant to a monotonic transform of the survival probabilities. It only focus on the ability of a predictive survival model to identify which individual is likely to fail first out of any pair of two individuals.
#
#
#
# It is comprised between 0 and 1 (lower is better).
# It answers the question "how close to the real probabilities are our estimates?".
#
#
#

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
from sklearn.model_selection import train_test_split


def train_test_split_within(X, y, **kwargs):
    """Ensure that test data durations are within train data durations."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    mask_duration_inliers = y_test["duration"] < y_train["duration"].max()
    y_test = y_test[mask_duration_inliers]
    X_test = X_test[mask_duration_inliers]
    return X_train, X_test, y_train, y_test


X = truck_failure_features
y = truck_failure_events

X_train, X_test, y_train, y_test = train_test_split_within(X, y, random_state=0)

# %%
from scipy.interpolate import interp1d

km_times, km_survival_probs = kaplan_meier_estimator(
    y_train["event"], y_train["duration"]
)

km_predict = interp1d(
    km_times,
    km_survival_probs,
    kind="previous",
    bounds_error=False,
    fill_value="extrapolate",
)

# %%
from sksurv.metrics import brier_score

def make_test_times(duration):
    """Bound times to the range of duration."""
    return np.linspace(duration.min(), duration.max() - 1, num=100)

time_grid = make_test_times(y_test["duration"])

# KM is a constant predictor: it always estimate the same survival
# curve for any individual in the training and test sets as it does
#not depend on features values of the X_train and X_test matrices.
km_curve = km_predict(time_grid)
y_pred_km_train = np.vstack([km_curve] * y_train.shape[0])
y_pred_km_test = np.vstack([km_curve] * y_test.shape[0])


def as_sksurv_recarray(y_frame):
    """Return scikit-survival's specific target format."""
    y_recarray = np.empty(
        shape=y_frame.shape[0],
        dtype=[("event", np.bool_), ("duration", np.float64)],
    )
    y_recarray["event"] = y_frame["event"]
    y_recarray["duration"] = y_frame["duration"]
    return y_recarray


_, km_brier_scores_test = brier_score(
    survival_train=as_sksurv_recarray(y_train),
    survival_test=as_sksurv_recarray(y_test),
    estimate=y_pred_km_test,
    times=time_grid,
)
_, km_brier_scores_train = brier_score(
    survival_train=as_sksurv_recarray(y_train),
    survival_test=as_sksurv_recarray(y_train),
    estimate=y_pred_km_train,
    times=time_grid,
)

# %%
from matplotlib import pyplot as plt
import seaborn as sns; sns.set_style("darkgrid")

plt.plot(times, km_brier_scores_train, label="train");
plt.plot(times, km_brier_scores_test, label="test");
plt.title("Time-varying Brier score of Kaplan Meier estimation (lower is better)");
plt.legend()
plt.xlabel("time (days)");

# %% [markdown]
# Additionnaly, we compute the Integrated Brier Score (IBS) which we will use to rank estimators:
# $$IBS = \frac{1}{t_{max} - t_{min}}\int^{t_{max}}_{t_{min}} BS(t) dt$$

# %%
km_ibs_train = integrated_brier_score(
    survival_train=as_sksurv_recarray(y_train),
    survival_test=as_sksurv_recarray(y_train),
    estimate=y_pred_km_train,
    times=times,
)
print(f"IBS of Kaplan-Meier estimator on train set: {km_ibs:.3f}")

# %%
from sksurv.metrics import integrated_brier_score

km_ibs_test = integrated_brier_score(
    survival_train=as_sksurv_recarray(y_train),
    survival_test=as_sksurv_recarray(y_test),
    estimate=y_pred_km_test,
    times=times,
)
print(f"IBS of Kaplan-Meier estimator on test set: {km_ibs_test:.3f}")

# %% [markdown]
# Since the KM estimator always predicts the same constant survival curve for any samples in `X_train` or `X_test`, it has the same IBS on both subsets. Still, it's an interesting baseline because it's well calibrated among all the constant survival curve predictors.

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


def get_c_index(event, duration, survival_probs_matrix):
    if survival_probs_matrix.ndim != 2:
        raise ValueError(
            "`survival_probs` must be a 2d array of "
            "shape (n_samples, times)."
        )
    # Cumulative hazard is also known as risk.
    cumulative_hazard = survival_to_risk_estimate(survival_probs_matrix)
    metrics = concordance_index_censored(event, duration, cumulative_hazard)
    return metrics[0]


def survival_to_risk_estimate(survival_probs_matrix):
    return -np.log(survival_probs_matrix + 1e-8).sum(axis=1)


# %%
km_c_index_test = get_c_index(y_test["event"], y_test["duration"], y_pred_km_test)
km_c_index_test

# %% [markdown]
# This is equivalent to a random prediction. Indeed, as our Kaplan Meier is a descriptive statistics, it can't be used to rank individuals predictions.

# %% [markdown]
# Next, we'll study how to add covariates $X$ to our analysis.
#
# ## IV. Predictive survival analysis
#
# We now introduce some quantities which are going to be at the core of many survival analysis models.
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
duration = df.pop("duration")
event = df.pop("event")
y = make_target(event, duration)
X = df

# %%
from sklearn.model_selection import train_test_split


# %%
X_train, X_test, y_train, y_test = train_test_split_within(X, y)
X_train.shape, X_test.shape

# %% [markdown]
# ***Exercice***
#
# Create a `ColumnTransformer` to encode categories `brand` and `truck_model`.
#
# *Hint*: Use `sklearn.preprocessing.OneHotEncoder` and `sklearn.compose.make_column_transformer`.

# %%
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

### Your code here
transformer = None
###

# %%
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

transformer = make_column_transformer(
    (OneHotEncoder(), ["brand", "truck_model"]),
    remainder="passthrough",
)

# %%
X_train

# %%
y_train

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
test_times = make_test_times(y_test["duration"])

fig, ax = plt.subplots()
for idx, step_func in enumerate(step_funcs[:5]):
    survival_probs = step_func(test_times)
    ax.plot(times, survival_probs, label=idx)
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

# %%
feature_names = cox_ph[0].get_feature_names_out()
weights = cox_ph[-1].coef_

# %%
features = (
    pd.DataFrame(
        dict(
            feature_name=feature_names,
            weight=weights,
        )
    )
    .sort_values("weight")
)
ax = sns.barplot(features, y="feature_name", x="weight", orient="h")
ax.set_title("Cox PH feature importance of $\lambda(t)$");

# %% [markdown]
# Finally, we compute the Brier score for our model.

# %%
from sksurv.metrics import brier_score, integrated_brier_score


cox_survival_probs_matrix = np.vstack([step_func(test_times) for step_func in step_funcs])

_, cox_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=cox_survival_probs_matrix,
    times=test_times,
)

cox_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=cox_survival_probs_matrix,
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

print(f"CoxPH IBS: {cox_ibs:.4f}")
print(f"KaplanMeier IBS: {km_ibs:.4f}")

# %%
cox_c_index = get_c_index(
    y_test["event"],
    y_test["duration"],
    cox_survival_probs_matrix,
)

print(f"Cox PH C-index: {cox_c_index:.4f}")
print(f"Kaplan Meier C-index: {km_c_index:.4f}")

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
    survival_probs = step_func(test_times)
    ax.plot(test_times, survival_probs, label=idx)
ax.set(
    title="Survival probabilities $\hat{S}(t)$ of Random Survival Forest",
    xlabel="time (days)",
    ylabel="S(t)",
)
plt.legend();

# %%
rsf_survival_probs_matrix = np.vstack(
    [step_func(test_times) for step_func in step_funcs]
)

_, rsf_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=rsf_survival_probs_matrix,
    times=test_times,
)

rsf_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=rsf_survival_probs_matrix,
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

print(f"RandomSurvivalForest IBS: {rsf_ibs:.4f}")
print(f"CoxPH IBS: {cox_ibs:.4f}")
print(f"KaplanMeier IBS: {km_ibs:.4f}")

# %%
rsf_c_index = get_c_index(y_test["event"], y_test["duration"], rsf_survival_probs_matrix)

print(f"Random Survival Index C-index: {rsf_c_index:.4f}")
print(f"Cox PH C-index: {cox_c_index:.4f}")
print(f"Kaplan Meier C-index: {km_c_index:.4f}")

# %% [markdown]
# ### IV.3 GradientBoostedIBS

# %%
import sys; sys.path.append("..")
from models.gradient_boosted_cif import GradientBoostedCIF
from model_selection.wrappers import PipelineWrapper


gb_cif = make_pipeline(
    transformer,
    GradientBoostedCIF(n_iter=20, learning_rate=0.2),
)
gb_cif = PipelineWrapper(gb_cif)
gb_cif.fit(X_train, y_train, times)

# %%
gb_survival_probs_matrix = gb_cif.predict_survival_function(X_test, test_times)

_, gb_brier_scores = brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=gb_survival_probs_matrix,
    times=test_times,
)

gb_ibs = integrated_brier_score(
    survival_train=y_train,
    survival_test=y_test,
    estimate=gb_survival_probs_matrix,
    times=test_times,
)

fig, ax = plt.subplots()
ax.plot(test_times, gb_brier_scores, label="GradientBoostedCIF")
ax.plot(test_times, rsf_brier_scores, label="RandomSurvivalForest")
ax.plot(test_times, cox_brier_scores, label="CoxPH")
ax.plot(test_times, km_brier_scores, label="KaplanMeier")
ax.set(
    title="Brier Scores",
    xlabel="time (days)",
    ylabel="$BS(t)^c$",
)
plt.legend();

print(f"GradientBoostedCIF IBS: {gb_ibs:.4f}")
print(f"RandomSurvivalForest IBS: {rsf_ibs:.4f}")
print(f"CoxPH IBS: {cox_ibs:.4f}")
print(f"KaplanMeier IBS: {km_ibs:.4f}")

# %%
gb_c_index = get_c_index(y_test["event"], y_test["duration"], gb_survival_probs_matrix)

print(f"GradientBoostedCIF C-index: {gb_c_index:.4f}")
print(f"Random Survival Index C-index: {rsf_c_index:.4f}")
print(f"Cox PH C-index: {cox_c_index:.4f}")
print(f"Kaplan Meier C-index: {km_c_index:.4f}")

# %% [markdown]
# ## V. Competing risks modeling with Aalen-Johanson
#
# So far, we've been dealing with a single kind of risk: any accident. What if we have different types of accident? This is the point of competing risks modeling. It aims at modeling the probability of incidence for different events, where these probabilities interfer with each other. A truck that had an accident is withdrawn from the fleet, and therefore can't experienced any other ones.
#
# For any event $k \in [1, K]$, the cumulative incidence function of the event $k$ becomes:
#
# $$CIF_k = P(T < t, \mathrm{event}=k)$$
#
# Aalen-Johanson estimates the CIF for multi-event $k$, by computing the global (any event) survival probabilities and the cause-specific hazards.
#
# <details><summary>Mathematical formulation</summary>
#     
# <br>
# We first compute the cause-specific hazards $\lambda_k$, by simply counting for each individual duration $t_i$ the number of individuals that have experienced the event $k$ at $t_i$ ($d_{i,k}$), and the number of people still at risk at $t_i$ ($n_i$).
#
# $$
# \hat{\lambda}_k(t_i)=\frac{d_{k,i}}{n_i}
# $$
#
# Then, we compute the survival probability any event with Kaplan Meier any event, where we can reused the cause-specific hazards.
#     
# $$
# \hat{S}(t)=\prod_{i:t_i\leq t} (1 - \frac{d_i}{n_i})=\prod_{i:t_i\leq t} (1 - \sum_k\hat{\lambda}_{k}(t_i))
# $$
#
# Finally, we compute the CIF of event $k$ as the sum of the cause-specific hazards, weighted by the survival probabilities.
#
# $$\hat{F}_k(t)=\sum_{i:t_i\leq t} \hat{\lambda}_k(t_i) \hat{S}(t_{i-1})$$
#     
#     
# </details>

# %% [markdown]
# Let's load our dataset another time. Notice that we have 3 types of event (plus the censoring 0). In the previous section, we only considered binary "any events" by applying `event > 0` to our event column.

# %%
df = pd.read_parquet("data_truck.parquet")
df

# %% [markdown]
# Let's use lifelines to estimate the ${CIF_k}$ using Aalen-Johanson. We need to indicate which event to fit on, so we'll iteratively fit the model on all events.

# %%
from lifelines import AalenJohansenFitter

total_cif = np.zeros(df.shape[0])

fig, ax = plt.subplots()
for event in [1, 2, 3]:
    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(df["duration"], df["event"], event_of_interest=event)
    ajf.plot(ax=ax, label=f"event {event}")
    cif_df = ajf.cumulative_density_
    cif_times = cif_df.index
    total_cif += cif_df[cif_df.columns[0]].values

ax.plot(cif_times, total_cif, label="total", linestyle="--", color="black")
ax.set(title="CIFs from Aalen Johansen", xlabel="time (days)")
plt.legend();


# %% [markdown]
# This non-conditional model helps us identify 3 types of events, having momentum at different times.

# %% [markdown]
# ## VI. Cumulative incidence function (CIF) using our GradientBoostedCIF
#
# We can now try to estimate the conditional cumulative incidence function using our GradientBoostedCIF.

# %%
def get_X_y(df):
    y = np.empty(
        df.shape[0],
        dtype=[("event", np.int8), ("duration", float)],
    )
    y["event"] = df.pop("event")
    y["duration"] = df.pop("duration")
    return df, y


# %%
df = pd.read_parquet("data_truck.parquet")
X, y = get_X_y(df)
X_train, X_test, y_train, y_test = train_test_split_within(X, y)

test_times = make_test_times(y_test["duration"])

total_mean_cif = np.zeros(times.shape[0])

fig, ax = plt.subplots()
for event in [1, 2, 3]:    
    gb_cif = make_pipeline(
        transformer,
        GradientBoostedCIF(event, n_iter=20, learning_rate=0.2),
    )
    gb_cif = PipelineWrapper(gb_cif)
    
    gb_cif.fit(X_train, y_train, times)
    cif_matrix_k = gb_cif.predict_cumulative_incidence(X_test, test_times)
    
    mean_cif_k = cif_matrix_k.mean(axis=0)
    total_mean_cif += mean_cif_k
    ax.plot(times, mean_cif_k, label=f"event {event}")

ax.plot(times, total_mean_cif, label="total", linestyle="--", color="black")
plt.legend();

# %% [markdown]
# In the second section of this tutorial, we'll study our GradientBoostedCIF in more depth by understanding how to find the median survival probability and compute its feature importance.
