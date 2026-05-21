import numpy as np
import pandas as pd
import cbsodata
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Fetch and Clean Data
data = pd.DataFrame(cbsodata.get_data('70895ned'))
data.dropna(subset=["Overledenen_1"], inplace=True)
df_clean = data[data.Perioden.str.contains('week')].copy()
df_clean = df_clean[~df_clean.Perioden.str.contains('1995 week 0')].reset_index(drop=True)

df_clean = df_clean.drop(columns=['ID'])
df_clean['to_first_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.str.contains('week 1')
df_clean['to_last_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.shift(-1).str.contains('week 0')
df_clean['partial_week'] = df_clean.Perioden.str.contains('dag')

df_clean.loc[df_clean['to_first_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(+1) + df_clean['Overledenen_1']
df_clean.loc[df_clean['to_last_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(-1) + df_clean['Overledenen_1']
df_clean.loc[df_clean['partial_week'] == False, 'deaths'] = df_clean['Overledenen_1']
df_clean = df_clean.dropna(subset=["deaths"]).reset_index(drop=True)

df_clean[['year', 'week']] = df_clean.Perioden.str.split("week", expand=True)
df_clean['week'] = df_clean.week.str.extract('(\d+)').astype(int)
df_clean['year'] = df_clean.year.str.extract('(\d+)').astype(int)
df_clean['deaths'] = df_clean['deaths'].astype(int)

df_clean = df_clean.drop(columns=['Overledenen_1', 'to_first_week', 'to_last_week', 'partial_week'])
df_clean = df_clean.rename(columns={"LeeftijdOp31December": "age", "Geslacht": "gender"})
df_clean = df_clean[['Perioden', 'gender', 'age', 'year', 'week', 'deaths']]

current_year = int(df_clean['year'].max())
df_clean = df_clean[df_clean['year'] >= (current_year - 10)].reset_index(drop=True)

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

# 2. FacetGrid Plot
g = sns.FacetGrid(df_clean, col="gender", hue="year", row='age', aspect=2, sharey=False)
g.map_dataframe(sns.lineplot, x='week', y='deaths', alpha=.7, estimator='mean', errorbar='sd')
g.set(xlabel="month", ylabel="deaths per week", xticks=np.arange(1, 53, (53/12)), xticklabels=months)
g.add_legend(title='')

for suffix in ['png', 'svg']:
    g.savefig(f'naar_Geslacht_leeftijd.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Prepare Pivot Table for Polar Plots
leeftijd = 'Totaal leeftijd'
sex = 'Totaal mannen en vrouwen'
df_circle = df_clean[(df_clean.age == leeftijd) & (df_clean.gender == sex)]
df_circle = df_circle.pivot(index='week', columns='year', values='deaths')

def data_for_year(y):
    """Safely extracts data and calculates theta angles for full or partial years."""
    year_data = df_circle[y].dropna().to_numpy()
    num_weeks = len(year_data)

    # If it's the current year and incomplete, map it proportionally to the completed weeks
    if y == current_year and num_weeks < 52:
        day_of_the_year = num_weeks * 7 + 3
        theta = np.linspace(0, (day_of_the_year / 365) * 2 * np.pi, num_weeks)
    else:
        # FIX: Check if next year exists in columns before trying to fetch its first week
        if (y + 1) in df_circle.columns and 1 in df_circle.index:
            next_week_1 = df_circle.loc[1, y + 1]
            if not pd.isna(next_week_1):
                year_data = np.append(year_data, next_week_1)
        else:
            # Fallback: link back to its own week 1 to close the circle loop cleanly
            year_data = np.append(year_data, year_data[0])

        theta = np.linspace(0, 2 * np.pi, len(year_data))

    return theta, year_data

def plot_year(ax, y, **kwargs):
    theta, values = data_for_year(y)
    ax.plot(theta, values, label=f"{y}", **kwargs)

def setup_polar_plot(figsize=(8, 6), constrained_layout=True):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels(months)
    ax.set_rlabel_position(180)
    return fig, ax

# --- Plot 1: Historic Overlay (The Spiral) ---
fig, ax = setup_polar_plot()

# Dynamic loop instead of repeating lines manually
start_year = current_year - 10
for i, y in enumerate(range(start_year, current_year - 2)):
    # Steadily scale line width from 0.1 up to 0.8 over the years
    lw = 0.1 + (i * (0.7 / 7))
    plot_year(ax, y, linewidth=lw, color='gray', alpha=0.5)

# Highlight recent history and current tracking
plot_year(ax, current_year - 2, color='tab:red', linewidth=1, linestyle='dotted')
plot_year(ax, current_year - 1, color='tab:orange', linewidth=2)
plot_year(ax, current_year, color='tab:green', linewidth=3)

fig.legend(loc='lower right')
fig.suptitle('Deaths in the Netherlands per week', fontsize=14, y=1.04)
ax.set_title(f"{sex}, {leeftijd}", fontsize=10, y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_perjaar.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# --- Plot 2: Statistics & Deviations ---
# Drop the current year so historical averages aren't skewed by incomplete data
years_historical = df_circle.drop(columns=[current_year], errors='ignore')

# Compute stats directly as numpy arrays or wrap them cleanly
def close_loop(series):
    """Appends the first element to the end to seamlessly close the polar chart."""
    arr = series.to_numpy()
    return np.append(arr, arr[0])

theta_stat = np.linspace(0, 2 * np.pi, len(years_historical) + 1)

mean_val   = close_loop(years_historical.mean(axis=1))
median_val = close_loop(years_historical.median(axis=1))
min_val    = close_loop(years_historical.min(axis=1))
max_val    = close_loop(years_historical.max(axis=1))
sd_val     = close_loop(years_historical.std(axis=1))

fig, ax = setup_polar_plot()

# Standard deviation band
ax.fill_between(theta_stat, mean_val + sd_val, mean_val - sd_val, alpha=0.3, label="SD", color='tab:blue')
# Min/Max range envelope
ax.fill_between(theta_stat, min_val, max_val, alpha=0.2, label="Min/Max", color='gray')
# Historical Median baseline
ax.plot(theta_stat, median_val, label="Median", linestyle='dashed', color='black')

# Overlay target years
plot_year(ax, current_year - 2, color='tab:orange', linewidth=1.5, linestyle='dotted')
plot_year(ax, current_year - 1, color='tab:orange', linewidth=1.5)
plot_year(ax, current_year, color='tab:green', linewidth=2)

fig.legend(loc='lower right')
fig.suptitle(f"Deaths per week in the Netherlands (since {current_year - 10})", fontsize=14, y=1.04)
ax.set_title(f"{sex}, {leeftijd}, median", fontsize=10, y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_median.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
