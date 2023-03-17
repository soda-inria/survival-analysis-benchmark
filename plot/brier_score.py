from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")


def plot_brier_scores(df_lines):
    """Plot brier score curve for each estimator.

    Parameters
    ----------
    df_lines: pd.DataFrame
        model | times | brier_scores
        'times' and 'brier_scores' are numpy array

    Notes
    -----
    'df_tables' and 'df_lines' are loaded with
    `model_selection.cross_validation.get_all_results()` after
    cross validation with `model_selection.cross_validation.run_cv()`
    """

    cols = df_lines.columns
    col_to_idx = dict(zip(cols, range(len(cols))))
    
    fig, ax = plt.subplots() #figsize=(14, 5), dpi=300)
    for row in df_lines.values:
        ax.plot(
            row[col_to_idx["times"]],
            row[col_to_idx["brier_scores"]],
            label=row[col_to_idx["model"]],
        )
    plt.xlabel("Duration (days)")
    plt.ylabel("Brier score")
    legend = plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=6,
        facecolor='white',
    )
    frame = legend.get_frame()
    frame.set_linewidth(2)

    return fig
