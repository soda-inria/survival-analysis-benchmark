# %%
from pycox.datasets import kkbox_v1
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxPHSurvivalAnalysis

# %%
# TODO: replace the following line with the result of Olivier's preprocessing
df = kkbox_v1.read_df()
print(df.columns)
print(df.shape)
df.head()
df["event"] = df["event"].astype(bool)
# %%
# Prepare data
df_cleaned = (
    df.assign(
        gender=lambda df: df["gender"].astype(str),
        city=lambda df: df["city"].astype(str),
        registered_via=lambda df: df["registered_via"].astype(str),
    )
    .fillna(
        {
            "gender": "nan_gender",
            "city": "nan_city",
            "registered_via": "nan_registered_via",
        }
    )
    .drop(["msno"], axis=1)
)
X_raw, y = get_x_y(df_cleaned, ("event", "duration"), pos_label=1)


def replace_nan(df):
    df["gender"] = df["gender"].astype(str)
    gender_map = dict(zip(df["gender"].unique(), range(df["gender"].nunique())))
    df["gender"] = df["gender"].map(gender_map)

    df["city"] = df["city"].astype(str).replace("nan", -1).astype(int)

    df["registered_via"] = (
        df["registered_via"].astype(str).replace("nan", -1).astype(int)
    )
    return df


X_raw = replace_nan(X_raw)
# %%
one_hot_enc = OneHotEncoder()
col_transformer = ColumnTransformer(
    [],
    # [("onehot_categorical", one_hot_enc, ["gender", "city", "registered_via"])],
    # remainder="passthrough",
    remainder=StandardScaler(),
)


X = col_transformer.fit_transform(X_raw)
# %%
set_config(display="text")  # displays text representation of estimators
cox_ph_estimator = CoxPHSurvivalAnalysis()
cox_ph_estimator.fit(X, y)
# %%
# TODO: unfinished analysis
# Plot evaluation metrics and train survival curves
