"""Weekly deaths in the Netherlands (CBS table 70895ned) as polar charts.

Writes PNG/SVG charts and one GIF animation into the current directory.
"""
import io
import time

import matplotlib

matplotlib.use("Agg")  # headless rendering; no GUI windows needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

# Deterministic SVG output: without this every run produces different element
# ids, so the files change (and get committed) even when the data did not.
plt.rcParams["svg.hashsalt"] = "oversterfte-nederland"

CBS_V4_BASE = "https://datasets.cbs.nl/odata/v1/CBS"
TABLE_ID = "70895ned"
TOTAL_AGE = "Totaal leeftijd"
TOTAL_SEX = "Totaal mannen en vrouwen"
FACET_AGES = ["0 tot 65 jaar", "65 tot 80 jaar", "80 jaar of ouder"]
FACET_SEXES = ["Mannen", "Vrouwen"]
COVID_YEARS = (2020, 2021, 2022)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
MONTH_TICKS = np.arange(0, 2 * np.pi, np.pi / 6)
THETA_FULL = np.linspace(0, 2 * np.pi, 53)

# ==========================================
# THEMES
# ==========================================
SCIENTIFIC = {
    "name": "Scientific Theme",
    "file_suffix": "",
    "dpi": 200,
    "bg_canvas": "white",
    "bg_face": "white",
    "grid_color": "#E5E5E5",
    "text_color": "#333333",
    "spine_color": "#888888",
    "spine_width": 0.8,
    "tick_weight": "normal",
    "title_color": "#212121",
    "title_size": 14,
    "title_y": 1.04,
    "subtitle_color": "#616161",
    "history_color": "#CFD8DC",
    "covid_colors": {2020: "#00838F", 2021: "#00695C", 2022: "#2E7D32"},  # jewel tones
    "covid_labels": {2020: "2020 (COVID)", 2021: "2021 (COVID)", 2022: "2022 (COVID)"},
    "c_prev2": "#78909C",
    "c_prev1": "#37474F",
    "c_current": "#D32F2F",  # classic crimson
    "prev_label": "{year}",
    "current_label": "{year}",
    "legend_edge": "#E0E0E0",
    "legend_text": None,
    # per-figure details
    "overlay": {
        "history": {"linewidth": 0.4, "alpha": 0.4},
        "history_legend": {"alpha": 0.7, "label": "History ({first}-{last})"},
        "covid_lw": 1.5, "covid_alpha": None, "prev_lw": (1.5, 1.8), "current_lw": 3.2,
        "legend_x": 1.25,
        "suptitle": "Deaths in the Netherlands per week",
        "title": "{sex}, {age}\nHighlighting Pandemic anomalies across complete historical record",
    },
    "median": {
        "sd": {"alpha": 0.25, "label": "Standard Deviation", "color": "#90A4AE"},
        "range": {"alpha": 0.10, "label": "Historical Range Bound", "color": "#CFD8DC"},
        "median": {"label": "Historical Median", "color": "#455A64", "linewidth": 1.2},
        "covid_lw": 1.3, "covid_alpha": 0.85, "prev_lw": (1.2, 1.5), "current_lw": 2.5,
        "legend_x": 1.25,
        "suptitle": "Deaths per week in the Netherlands (Full Dataset)",
        "title": "{sex}, {age}, complete spread distribution analysis",
    },
}

BLOG = {
    "name": "Modern Blog Theme",
    "file_suffix": "_blog",
    "dpi": 300,
    "bg_canvas": "#F7FAFC",
    "bg_face": "#FFFFFF",
    "grid_color": "#E2E8F0",
    "text_color": "#4A5568",
    "spine_color": "#CBD5E0",
    "spine_width": 1.0,
    "tick_weight": "bold",
    "title_color": "#1A202C",
    "title_size": 16,
    "title_y": 1.05,
    "subtitle_color": "#718096",
    "history_color": "#CBD5E0",
    "covid_colors": {2020: "#0EA5E9", 2021: "#10B981", 2022: "#6366F1"},  # vivid blog palette
    "covid_labels": {
        2020: "2020 Pandemic Outbreak",
        2021: "2021 Delta Wave",
        2022: "2022 Omicron Structural Shift",
    },
    "facet_covid_labels": {2022: "2022 Omicron Shift"},
    "c_prev2": "#718096",
    "c_prev1": "#4A5568",
    "c_current": "#FF3366",  # neon coral pink
    "prev_label": "{year} Trend",
    "current_label": "Current Tracker ({year})",
    "legend_edge": "#E2E8F0",
    "legend_text": {"color": "#2D3748", "fontsize": 9.5},
    "overlay": {
        "history": {"linewidth": 0.5, "alpha": 0.5},
        "history_legend": {"alpha": 0.8, "label": "Historical Baseline ({first}-{last})"},
        "covid_lw": 1.6, "covid_alpha": None, "prev_lw": (1.5, 1.8), "current_lw": 3.5,
        "legend_x": 1.32,
        "suptitle": "Tracking Mortality Trends in the Netherlands",
        "title": "Demographic Filters: {sex} | {age}",
    },
    "median": {
        "sd": {"alpha": 0.35, "label": "Standard Deviation Bounds", "color": "#CBD5E0"},
        "range": {"alpha": 0.15, "label": "Historical Minimum / Maximum", "color": "#E2E8F0"},
        "median": {"label": "Historical Median Trend", "color": "#2D3748", "linewidth": 1.5},
        "covid_lw": 1.2, "covid_alpha": 0.7, "prev_lw": (1.2, 1.5), "current_lw": 3.0,
        "legend_x": 1.35,
        "suptitle": "Historical Mortality Spread Context",
        "title": "Demographic Filters: {sex} | {age}",
    },
}


# ==========================================
# 1. DATA ACQUISITION & TRANSITIONAL CLEANING
# ==========================================
def _get_json_with_retry(session, url, retries=5, base_delay=15, timeout=60):
    """GET a URL as JSON, retrying with backoff on transient failures."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = base_delay * attempt
                print(f"  Attempt {attempt}/{retries} failed for {url} ({e!r}); retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts. Last error: {last_err}")


def get_odata_v4_collection(session, table_id, collection):
    """Fetch a full OData v4 collection (following @odata.nextLink pagination) as a DataFrame."""
    url = f"{CBS_V4_BASE}/{table_id}/{collection}"
    rows = []
    while url:
        payload = _get_json_with_retry(session, url)
        rows.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink")
    return pd.DataFrame(rows)


def fetch_cbs_table_v4(table_id):
    """
    Fetch table_id from the CBS OData v4 API and reshape the long-format
    Observations into one row per Perioden x Geslacht x LeeftijdOp31December
    with the death count in 'Overledenen_1' and human-readable labels.
    """
    print(f"Fetching data from CBS OData v4 ({table_id})...")
    with requests.Session() as session:  # one connection reused for all requests
        obs = get_odata_v4_collection(session, table_id, "Observations")
        print("  Fetching dimension code tables...")
        measure_codes = get_odata_v4_collection(session, table_id, "MeasureCodes")
        titles = {
            dim: get_odata_v4_collection(session, table_id, f"{dim}Codes")
            .set_index("Identifier")["Title"]
            for dim in ("Geslacht", "LeeftijdOp31December", "Perioden")
        }

    # This table has a single "deaths" measure; find it by title rather than
    # hardcoding its v4 identifier (identifiers changed between v3 and v4).
    death_measure = measure_codes[measure_codes["Title"].str.contains("Overledenen", case=False, na=False)]
    if death_measure.empty:
        raise RuntimeError(f"Could not find an 'Overledenen' measure in MeasureCodes for table {table_id}")
    obs = obs[obs["Measure"] == death_measure.iloc[0]["Identifier"]]

    # Map v4 dimension codes back to their labels.
    data = pd.DataFrame({dim: obs[dim].map(mapping) for dim, mapping in titles.items()})
    data["Overledenen_1"] = pd.to_numeric(obs["Value"], errors="coerce")

    unmapped = data[list(titles)].isna().any(axis=1)
    if unmapped.any():
        print(f"  Warning: {unmapped.sum()} observation(s) had unmapped codes and were dropped.")
        data = data[~unmapped]
    return data.reset_index(drop=True)


def load_deaths():
    """Fetch the CBS table and return tidy weekly deaths (gender, age, year, week, deaths)."""
    data = fetch_cbs_table_v4(TABLE_ID)
    data = data.dropna(subset=["Overledenen_1"])

    df = data[data.Perioden.str.contains("week")]
    df = df[~df.Perioden.str.contains("1995 week 0")].reset_index(drop=True)

    # Transitional weeks: a year starts/ends with a partial week ("... (3 dagen)").
    # The partial "week 0" is merged into the neighbouring partial week 1 or
    # partial last week of the previous year; week 0 itself is then dropped.
    period = df.Perioden
    counts = df.Overledenen_1
    partial = period.str.contains("dag")
    to_first_week = partial & period.str.contains("week 1")
    to_last_week = partial & period.shift(-1).str.contains("week 0").fillna(False).astype(bool)

    deaths = counts.where(~partial)
    deaths = deaths.mask(to_first_week, counts.shift(1) + counts)
    deaths = deaths.mask(to_last_week, counts.shift(-1) + counts)

    parsed = period.str.extract(r"(\d{4})\s+week\s+(\d+)").astype(int)
    tidy = pd.DataFrame({
        "gender": df.Geslacht,
        "age": df.LeeftijdOp31December,
        "year": parsed[0],
        "week": parsed[1],
        "deaths": deaths,
    }).dropna(subset=["deaths"])
    tidy["deaths"] = tidy["deaths"].astype(int)
    return tidy


def week_year_pivot(df, gender, age):
    """Matrix of deaths with weeks as rows and years as columns for one demographic group."""
    sub = df[(df.gender == gender) & (df.age == age)]
    return sub.pivot(index="week", columns="year", values="deaths")


# ==========================================
# 2. POLAR ROUTINES & MATHEMATICAL BRIDGING
# ==========================================
def year_curve(pivot, year, current_year):
    """
    (theta, values) for one year mapped onto a 52-week circle.

    Past years are closed with week 1 of the following year (or their own first
    week), the current year stops at its latest reported week.
    """
    column = pivot[year]

    def filled(weeks):
        return column.reindex(weeks).interpolate(method="linear").bfill().ffill().to_numpy()

    if year == current_year:
        last_week = int(column.last_valid_index())
        return np.arange(last_week) / 52 * 2 * np.pi, filled(range(1, last_week + 1))

    values = filled(range(1, 53))
    closing = np.nan
    if year + 1 in pivot.columns and 1 in pivot.index:
        closing = pivot.at[1, year + 1]
    if pd.isna(closing):
        closing = values[0]
    return THETA_FULL, np.append(values, closing)


def all_curves(pivot, current_year):
    return {int(y): year_curve(pivot, y, current_year) for y in pivot.columns}


def style_polar_ax(ax, theme, *, tick_size=10.0, grid_width=0.6, x_pad=None, r_size=None):
    ax.set_facecolor(theme["bg_face"])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTHS, fontsize=tick_size, color=theme["text_color"],
                       fontweight=theme["tick_weight"])
    if x_pad is not None:
        ax.tick_params(axis="x", pad=x_pad)
    ax.grid(True, color=theme["grid_color"], linestyle="--", linewidth=grid_width)
    ax.set_rlabel_position(180)
    ax.tick_params(axis="y", colors=theme["text_color"], labelsize=r_size)
    ax.spines["polar"].set_color(theme["spine_color"])
    ax.spines["polar"].set_linewidth(theme["spine_width"])


def setup_polar_plot(theme, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(theme["bg_canvas"])
    ax = fig.add_subplot(111, projection="polar")
    style_polar_ax(ax, theme, x_pad=12)
    return fig, ax


def add_terminal_marker(ax, theta, values, theme):
    """Dot plus text label at the latest reported week."""
    if len(theta) == 0:
        return
    t_last, v_last = theta[-1], values[-1]
    color = theme["c_current"]

    ax.plot(t_last, v_last, marker="o", markersize=8, color=color,
            markeredgecolor="white", markeredgewidth=2, zorder=15)

    left = 0 <= t_last < np.pi
    ax.annotate(
        f"Week {len(values)}\n{int(v_last):,} deaths",
        xy=(t_last, v_last),
        xytext=(160 if left else -160, 0),
        textcoords="offset points",
        ha="left" if left else "right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor=color,
                  alpha=0.95, linewidth=1.2),
        zorder=16,
    )


def plot_history(ax, curves, highlighted, color, **line_kw):
    """Thin background lines for every year that is not highlighted."""
    for year, (theta, values) in curves.items():
        if year not in highlighted:
            ax.plot(theta, values, color=color, **line_kw)


def plot_highlights(ax, curves, theme, current_year, *, covid_lw, covid_alpha, prev_lw,
                    current_lw, labels=True):
    """COVID years, the two previous years and the current year. Returns the current curve."""

    def label(text):
        return text if labels else None

    alphas = covid_alpha if isinstance(covid_alpha, tuple) else (covid_alpha,) * len(COVID_YEARS)
    for year, alpha in zip(COVID_YEARS, alphas):
        if year in curves:
            ax.plot(*curves[year], label=label(theme["covid_labels"][year]),
                    color=theme["covid_colors"][year], linewidth=covid_lw, alpha=alpha)

    for offset, style, color, lw in ((2, "dotted", theme["c_prev2"], prev_lw[0]),
                                     (1, "solid", theme["c_prev1"], prev_lw[1])):
        year = current_year - offset
        if year in curves:
            ax.plot(*curves[year], label=label(theme["prev_label"].format(year=year)),
                    linestyle=style, color=color, linewidth=lw)

    current = curves.get(current_year)
    if current is not None:
        ax.plot(*current, label=label(theme["current_label"].format(year=current_year)),
                color=theme["c_current"], linewidth=current_lw, zorder=10)
    return current


def finish_single_figure(fig, ax, theme, spec, basename):
    """Legend, titles and PNG/SVG export shared by the single-panel charts."""
    legend = ax.legend(loc="lower right", bbox_to_anchor=(spec["legend_x"], -0.05), frameon=True,
                       facecolor=theme["bg_face"], edgecolor=theme["legend_edge"])
    if theme["legend_text"]:
        for text in legend.get_texts():
            text.set(**theme["legend_text"])

    fig.suptitle(spec["suptitle"], fontsize=theme["title_size"], fontweight="bold",
                 color=theme["title_color"], y=theme["title_y"])
    ax.set_title(spec["title"].format(sex=TOTAL_SEX, age=TOTAL_AGE), fontsize=10,
                 color=theme["subtitle_color"], y=1.1)
    save_figure(fig, f"{basename}{theme['file_suffix']}", theme)


def save_figure(fig, basename, theme, dpi=None):
    kw = dict(dpi=dpi or theme["dpi"], bbox_inches="tight", facecolor=theme["bg_canvas"])
    fig.savefig(f"{basename}.png", **kw)
    fig.savefig(f"{basename}.svg", metadata={"Date": None}, **kw)  # no timestamp: stable diffs
    plt.close(fig)


# ==========================================
# 3. OVERLAY PLOT (all years)
# ==========================================
def plot_overlay(theme, curves, highlighted, start_year, current_year):
    spec = theme["overlay"]
    fig, ax = setup_polar_plot(theme)

    plot_history(ax, curves, highlighted, theme["history_color"], **spec["history"])
    ax.plot([], [], color=theme["history_color"], alpha=spec["history_legend"]["alpha"], linewidth=1,
            label=spec["history_legend"]["label"].format(first=start_year, last=current_year - 3))

    current = plot_highlights(ax, curves, theme, current_year, covid_lw=spec["covid_lw"],
                              covid_alpha=spec["covid_alpha"], prev_lw=spec["prev_lw"],
                              current_lw=spec["current_lw"])
    add_terminal_marker(ax, *current, theme)
    finish_single_figure(fig, ax, theme, spec, "sterfte_perjaar")


# ==========================================
# 4. HISTORICAL MEDIAN & SPREAD PLOT
# ==========================================
def historical_stats(pivot, current_year):
    """Per-week statistics over all completed years, closed into a loop (53 points)."""
    hist = pivot.drop(columns=[current_year], errors="ignore").reindex(range(1, 53))
    hist = hist.interpolate(axis=0)

    def closed(series):
        arr = series.to_numpy()
        return np.append(arr, arr[0])

    return {name: closed(getattr(hist, name)(axis=1))
            for name in ("mean", "median", "min", "max", "std")}


def plot_median(theme, curves, stats, current_year):
    spec = theme["median"]
    fig, ax = setup_polar_plot(theme)

    ax.fill_between(THETA_FULL, stats["mean"] + stats["std"], stats["mean"] - stats["std"],
                    **spec["sd"])
    ax.fill_between(THETA_FULL, stats["min"], stats["max"], **spec["range"])
    ax.plot(THETA_FULL, stats["median"], linestyle="dashed", **spec["median"])

    current = plot_highlights(ax, curves, theme, current_year, covid_lw=spec["covid_lw"],
                              covid_alpha=spec["covid_alpha"], prev_lw=spec["prev_lw"],
                              current_lw=spec["current_lw"])
    add_terminal_marker(ax, *current, theme)
    finish_single_figure(fig, ax, theme, spec, "sterfte_median")


# ==========================================
# 5. ANIMATED TRACK (blog theme)
# ==========================================
def render_animation(theme, curves, current_year, *, n_years=26, frame_ms=200, pause_ms=3000,
                     dpi=150, filename="sterfte_anim_blog.gif"):
    """One frame per year; the final frame is held for `pause_ms` extra."""
    fig, ax = setup_polar_plot(theme)
    center_label = ax.text(0, 0, "", ha="center", va="center", fontsize=16, fontweight="bold",
                           color=theme["title_color"])
    covid_alpha = {2020: 0.8, 2021: 0.8, 2022: 0.7}

    frames = []
    for year in range(current_year - n_years + 1, current_year + 1):
        theta, values = curves[year]
        if year == current_year:
            ax.plot(theta, values, color=theme["c_current"], linewidth=3.5, zorder=10)
            add_terminal_marker(ax, theta, values, theme)
        elif year in COVID_YEARS:
            ax.plot(theta, values, color=theme["covid_colors"][year], linewidth=1.4,
                    alpha=covid_alpha[year])
        else:
            ax.plot(theta, values, color=theme["history_color"], alpha=0.4, linewidth=0.5)
        center_label.set_text(f"{year}")

        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", dpi=dpi, facecolor=theme["bg_canvas"])
        size = tuple(int(round(v * dpi)) for v in fig.get_size_inches())
        frames.append(Image.frombuffer("RGBA", size, buf.getbuffer(), "raw", "RGBA", 0, 1)
                      .convert("RGB"))
    plt.close(fig)

    # Hold the last frame through its duration instead of rendering duplicate frames.
    durations = [frame_ms] * (len(frames) - 1) + [frame_ms + pause_ms]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=durations, loop=0)


# ==========================================
# 6. DEMOGRAPHIC FACET GRID (SEX x AGE)
# ==========================================
def plot_facets(theme, facet_curves, highlighted, start_year, current_year):
    print(f"Generating demographic breakdown grids for: {theme['name']}...")
    fig, axes = plt.subplots(nrows=len(FACET_SEXES), ncols=len(FACET_AGES), figsize=(16, 11),
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(theme["bg_canvas"])

    for row, gender in enumerate(FACET_SEXES):
        for col, age in enumerate(FACET_AGES):
            ax = axes[row, col]
            curves = facet_curves[gender, age]
            style_polar_ax(ax, theme, tick_size=8.5, grid_width=0.5, r_size=7.5)

            plot_history(ax, curves, highlighted, theme["history_color"], linewidth=0.4, alpha=0.4)
            current = plot_highlights(ax, curves, theme, current_year, covid_lw=1.3,
                                      covid_alpha=(0.85, 0.85, 0.8), prev_lw=(1.3, 1.5),
                                      current_lw=2.5, labels=False)
            if current is not None and len(current[0]):
                ax.plot(current[0][-1], current[1][-1], marker="o", markersize=5,
                        color=theme["c_current"], markeredgecolor=theme["bg_face"],
                        markeredgewidth=1, zorder=15)

            ax.set_title(f"{gender} | {age}", fontsize=11, fontweight="bold",
                         color=theme["title_color"], pad=14)

    covid_labels = {**theme["covid_labels"], **theme.get("facet_covid_labels", {})}
    handles = [plt.Line2D([0], [0], color=theme["history_color"], linewidth=1,
                          label=f"Historical Baseline ({start_year}-{current_year - 3})")]
    handles += [plt.Line2D([0], [0], color=theme["covid_colors"][y], linewidth=1.5,
                           label=covid_labels[y]) for y in COVID_YEARS]
    handles += [
        plt.Line2D([0], [0], color=theme["c_prev2"], linestyle="dotted", linewidth=1.5,
                   label=theme["prev_label"].format(year=current_year - 2)),
        plt.Line2D([0], [0], color=theme["c_prev1"], linestyle="solid", linewidth=1.5,
                   label=theme["prev_label"].format(year=current_year - 1)),
        plt.Line2D([0], [0], color=theme["c_current"], linewidth=2.5,
                   label=theme["current_label"].format(year=current_year)),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               facecolor=theme["bg_face"], edgecolor=theme["grid_color"], fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Weekly Mortality Breakdown Across Demographic Groups", fontsize=18,
                 fontweight="bold", color=theme["title_color"], y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.25, hspace=0.3)
    save_figure(fig, f"naar_Geslacht_leeftijd{theme['file_suffix']}", theme, dpi=150)


# ==========================================
# MAIN
# ==========================================
def main():
    df = load_deaths()
    current_year = int(df["year"].max())
    start_year = int(df["year"].min())
    print(f"Analyzing complete historical range: {start_year} to {current_year}")

    highlighted = {*COVID_YEARS, current_year - 2, current_year - 1, current_year}

    # All curves are computed once and shared by every chart and theme.
    total_pivot = week_year_pivot(df, TOTAL_SEX, TOTAL_AGE)
    total_curves = all_curves(total_pivot, current_year)
    stats = historical_stats(total_pivot, current_year)
    facet_curves = {
        (gender, age): all_curves(week_year_pivot(df, gender, age), current_year)
        for gender in FACET_SEXES for age in FACET_AGES
    }

    for theme in (SCIENTIFIC, BLOG):
        print(f"Generating overlay and median charts for: {theme['name']}...")
        plot_overlay(theme, total_curves, highlighted, start_year, current_year)
        plot_median(theme, total_curves, stats, current_year)

    print("Rendering animation...")
    render_animation(BLOG, total_curves, current_year)

    for theme in (SCIENTIFIC, BLOG):
        plot_facets(theme, facet_curves, highlighted, start_year, current_year)

    print("All charts generated successfully!")


if __name__ == "__main__":
    main()
