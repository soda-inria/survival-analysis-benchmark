import numpy as np
from matplotlib import pyplot as plt


def plot_individuals_survival_curve(df_tables, df_lines, y, n_indiv=5):
    """For each estimator, plot individual survival curves with
    event or censoring marker, for a sample of rows.

    Parameters
    ----------
        df_tables: pd.DataFrame,
            | Method | IBS
            Useful to rank estimators plot by mean IBS.

        df_lines: pd.DataFrame
            | model  | times | survival_probs
            Individual survival_probs to sample from and plot
        
        y: np.ndarray,
            Target vector, containing survival or 
            censoring times, useful to plot markers

        n_indiv: int,
            Number of individual curves to display.
            
    """
    # use df_tables to sort df_lines
    df_tables["mean_IBS"] = df_tables["IBS"].str.split("±") \
        .str[0].str.strip().astype(float)
    
    df_lines = df_lines.merge(
        df_tables[["Method", "mean_IBS"]],
        left_on="model",
        right_on="Method",
    )
    df_lines.sort_values("mean_IBS", inplace=True)
    
    n_rows = df_lines.shape[0]
    
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(10, 17),
        dpi=300,
        constrained_layout=True,
    )

    cols = df_lines.columns
    col_to_idx = dict(zip(cols, range(len(cols))))

    # Some models has been tested on a small datasets,
    # so we need to use the first `max_indiv_id` rows to compare all models
    max_indiv_id = min([len(el) for el in df_lines["survival_probs"].values])
    idxs_indiv = np.random.uniform(high=max_indiv_id, size=n_indiv).astype(int)

    for idx, row in enumerate(df_lines.values):
        for jdx in idxs_indiv:
            times = row[col_to_idx["times"]]
            surv_probs = row[col_to_idx["survival_probs"]][jdx, :]
            axes[idx].plot(
                times,
                surv_probs
            )
            color = axes[idx].lines[-1].get_color()
            # place the dot on the curve for viz purposes
            is_event = y["event"][jdx]
            event_time = y["duration"][jdx]
            surv_prob_projected = surv_probs[
                np.argmin(
                    np.abs(times - event_time)
                )
            ]
            axes[idx].plot(
                event_time,
                surv_prob_projected,
                "^" if is_event else "o",
                color=color,
            )
        axes[idx].set_title(row[col_to_idx["model"]])
