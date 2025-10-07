
# =============================================================================
# Marriage Analysis — Scatterplot Helpers (matplotlib-only)
# =============================================================================
import os, math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_ALIASES = {
    "country": ["country", "Country", "Country Name", "COUNTRY", "iso3", "ISO3"],
    "year":    ["year", "Year", "TIME", "date_year"],
    "fertility": ["fertility", "tfr", "FertilityRate", "SP.DYN.TFRT.IN", "fertility_rate"],
    "marriage":  ["marriage", "marriage_rate", "crude_marriage_rate", "marriages_per_1000", "MarriageRate"],
}

def _resolve_columns(df: pd.DataFrame, column_aliases: Dict[str, List[str]]):
    resolved = {}
    for logical, candidates in column_aliases.items():
        for c in candidates:
            if c in df.columns:
                resolved[logical] = c
                break
        if logical not in resolved:
            raise KeyError(f"Could not resolve column for '{logical}'. Tried: {candidates}")
    return resolved

def _maybe_subselect_years(df, year_col, year_range):
    if year_range is None:
        return df
    lo, hi = year_range
    return df[(df[year_col] >= lo) & (df[year_col] <= hi)]

def _pearsonr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or y.size < 3:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum())
    if denom == 0:
        return float("nan")
    return float((xm*ym).sum() / denom)

def _ols_slope_intercept(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan"), float("nan")
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])

def _simple_scatter_with_fit(x, y, title, xlabel, ylabel, savepath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, alpha=0.6)
    m, b = _ols_slope_intercept(x, y)
    if not (np.isnan(m) or np.isnan(b)):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m*xs + b, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, dpi=160)
    plt.close(fig)
    r = _pearsonr(x, y)
    return {"r": r, "slope": m, "intercept": b}

def pooled_scatter(df, column_aliases=DEFAULT_ALIASES, year_range=None,
                   savepath="pooled_marriage_vs_fertility.png",
                   title="Fertility vs Marriage (pooled)"):
    cols = _resolve_columns(df, column_aliases)
    df2 = df.dropna(subset=[cols["fertility"], cols["marriage"], cols["year"]]).copy()
    df2 = _maybe_subselect_years(df2, cols["year"], year_range)
    return _simple_scatter_with_fit(
        df2[cols["marriage"]].to_numpy(),
        df2[cols["fertility"]].to_numpy(),
        title, cols["marriage"], cols["fertility"], savepath
    )

def within_country_scatter(df, column_aliases=DEFAULT_ALIASES, year_range=None,
                           savepath="within_marriage_vs_fertility.png",
                           title="Fertility vs Marriage (within-country demeaned)"):
    cols = _resolve_columns(df, column_aliases)
    df2 = df.dropna(subset=[cols["fertility"], cols["marriage"], cols["country"], cols["year"]]).copy()
    df2 = _maybe_subselect_years(df2, cols["year"], year_range)
    df2["_x"] = df2[cols["marriage"]] - df2.groupby(cols["country"])[cols["marriage"]].transform("mean")
    df2["_y"] = df2[cols["fertility"]] - df2.groupby(cols["country"])[cols["fertility"]].transform("mean")
    return _simple_scatter_with_fit(
        df2["_x"].to_numpy(), df2["_y"].to_numpy(),
        title, f"{cols['marriage']} (demeaned)", f"{cols['fertility']} (demeaned)", savepath
    )

def lagged_scatter(df, lags=[1,2,3], column_aliases=DEFAULT_ALIASES, year_range=None,
                   out_dir=".", title_prefix="Fertility_t vs Marriage_{t-L} — L="):
    import pandas as pd, numpy as np, os
    os.makedirs(out_dir, exist_ok=True)
    cols = _resolve_columns(df, column_aliases)
    df2 = df.dropna(subset=[cols["fertility"], cols["marriage"], cols["country"], cols["year"]]).copy()
    df2 = _maybe_subselect_years(df2, cols["year"], year_range)
    df2 = df2.sort_values([cols["country"], cols["year"]])
    results = []
    for L in lags:
        g = df2.groupby(cols["country"], group_keys=False).apply(
            lambda d: d.assign(**{f"_xlag{L}": d[cols["marriage"]].shift(L)})
        )
        gL = g.dropna(subset=[f"_xlag{L}", cols["fertility"]])
        savepath = os.path.join(out_dir, f"lag{L}_marriage_vs_fertility.png")
        stats = _simple_scatter_with_fit(
            gL[f"_xlag{L}"].to_numpy(),
            gL[cols["fertility"]].to_numpy(),
            f"{title_prefix}{L}", f"{cols['marriage']} (t-{L})", f"{cols['fertility']} (t)", savepath
        )
        stats["lag"] = L
        stats["n"] = int(gL.shape[0])
        results.append(stats)
    return pd.DataFrame(results)[["lag","n","r","slope","intercept"]]

def partial_scatter(df, controls, column_aliases=DEFAULT_ALIASES, year_range=None,
                    savepath="partial_marriage_vs_fertility.png",
                    title="Fertility vs Marriage (partial | controls)"):
    cols = _resolve_columns(df, column_aliases)
    need = [cols["fertility"], cols["marriage"]] + controls
    for c in controls:
        if c not in df.columns:
            raise KeyError(f"Control '{c}' not found in DataFrame.")
    df2 = df.dropna(subset=need + [cols["year"]]).copy()
    df2 = _maybe_subselect_years(df2, cols["year"], year_range)

    def regress_out(y, X):
        X_ = np.column_stack([X.to_numpy(), np.ones((X.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(X_, y.to_numpy(), rcond=None)
        yhat = X_.dot(beta)
        return y.to_numpy() - yhat

    X = df2[controls]
    y_resid = regress_out(df2[cols["fertility"]], X)
    x_resid = regress_out(df2[cols["marriage"]], X)

    return _simple_scatter_with_fit(
        x_resid, y_resid, title,
        f"{cols['marriage']} | " + ", ".join(controls),
        f"{cols['fertility']} | " + ", ".join(controls),
        savepath
    )
