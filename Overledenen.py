import numpy as np
import pandas as pd
import cbsodata
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. DATA ACQUISITION & TRANSITIONAL CLEANING
# ==========================================
print("Fetching data from CBS OData...")
data = pd.DataFrame(cbsodata.get_data('70895ned'))
data.dropna(subset=["Overledenen_1"], inplace=True)

df_clean = data[data.Perioden.str.contains('week')].copy()
df_clean = df_clean[~df_clean.Perioden.str.contains('1995 week 0')].reset_index(drop=True)
df_clean = df_clean.drop(columns=['ID'], errors='ignore')

# --- PRESERVED: YOUR ORIGINAL TRANSITIONAL WEEK LOGIC ---
df_clean['to_first_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.str.contains('week 1')
df_clean['to_last_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.shift(-1).str.contains('week 0')
df_clean['partial_week'] = df_clean.Perioden.str.contains('dag')

df_clean.loc[df_clean['to_first_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(+1) + df_clean['Overledenen_1']
df_clean.loc[df_clean['to_last_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(-1) + df_clean['Overledenen_1']
df_clean.loc[df_clean['partial_week'] == False, 'deaths'] = df_clean['Overledenen_1']
df_clean = df_clean.dropna(subset=["deaths"]).reset_index(drop=True)
# --------------------------------------------------------

# Clean up date structures via regex capture groups
parsed_dates = df_clean['Perioden'].str.extract(r'(?P<year_str>\d{4})\s+week\s+(?P<week_str>\d+)')
df_clean['year'] = pd.to_numeric(parsed_dates['year_str'])
df_clean['week'] = pd.to_numeric(parsed_dates['week_str'])
df_clean['deaths'] = pd.to_numeric(df_clean['deaths']).astype(int)

df_clean = df_clean.drop(columns=['Overledenen_1', 'to_first_week', 'to_last_week', 'partial_week'])
df_clean = df_clean.rename(columns={"LeeftijdOp31December": "age", "Geslacht": "gender"})
df_clean = df_clean[['Perioden', 'gender', 'age', 'year', 'week', 'deaths']]

# --- FIX: REMOVED THE 10-YEAR FILTER TO KEEP ALL AVAILABLE DATA ---
current_year = int(df_clean['year'].max())
start_year = int(df_clean['year'].min())
print(f"Analyzing complete historical range: {start_year} to {current_year}")

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

# Pivot targeted demographic categories to matrix columns
leeftijd = 'Totaal leeftijd'
sex = 'Totaal mannen en vrouwen'
df_circle = df_clean[(df_clean.age == leeftijd) & (df_clean.gender == sex)]

# Find the absolute latest valid data week for the current year BEFORE pivoting
latest_active_week = df_circle[df_circle['year'] == current_year]['week'].max()
print(f"Current year detected: {current_year}. Latest updated week: {latest_active_week}")

df_circle = df_circle.pivot(index='week', columns='year', values='deaths')

# ==========================================
# 2. POLAR ROUTINES & MATHEMATICAL BRIDGING
# ==========================================
def data_for_year(y):
    """
    Extracts data frames strictly mapped out over a true 52-week calendar distribution.
    """
    if y == current_year:
        weeks = np.arange(1, latest_active_week + 1)
        values = np.array([df_circle.loc[w, y] for w in weeks])
        theta = ((weeks - 1) / 52) * 2 * np.pi
        return theta, values

    weeks = np.arange(1, 53)
    values = np.array([df_circle.loc[w, y] for w in weeks])

    if pd.Series(values).isna().any():
        values = pd.Series(values).interpolate(method='linear').to_numpy()

    next_year = y + 1
    if next_year in df_circle.columns and 1 in df_circle.index:
        next_val = df_circle.loc[1, next_year]
        if not pd.isna(next_val):
            theta = np.linspace(0, 2 * np.pi, 53)
            values = np.append(values, next_val)
            return theta, values

    theta = np.linspace(0, 2 * np.pi, 53)
    values = np.append(values, values[0])
    return theta, values

def setup_polar_plot(figsize=(10, 8), constrained_layout=True):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels(months, fontsize=9, color='#333333')
    ax.tick_params(axis='x', pad=12)

    ax.grid(True, color='#E5E5E5', linestyle='--', linewidth=0.5)
    ax.set_rlabel_position(180)
    ax.spines['polar'].set_color('#888888')
    ax.spines['polar'].set_linewidth(0.8)
    return fig, ax

def add_terminal_marker(ax, theta, values):
    """Places a distinct tracking point and metadata annotation at the leading edge."""
    if len(theta) == 0 or len(values) == 0:
        return
    t_last = theta[-1]
    v_last = values[-1]

    ax.plot(t_last, v_last, marker='o', markersize=7, color='#D32F2F', markeredgecolor='white', markeredgewidth=1.5, zorder=15)

    text_padding = 150
    ha = 'left' if 0 <= t_last < np.pi else 'right'

    ax.annotate(
        f"W{len(values)}: {int(v_last):,}",
        xy=(t_last, v_last),
        xytext=(text_padding if ha == 'left' else -text_padding, 0),
        textcoords='offset points',
        ha=ha,
        va='center',
        fontsize=9,
        fontweight='bold',
        color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D32F2F', alpha=0.9, linewidth=0.8),
        zorder=16
    )

# ==========================================
# 3. STATIC HISTORIC OVERLAY PLOT
# ==========================================
print("Generating historic overlay polar chart...")
fig, ax = setup_polar_plot()

# 1. Background Historical Base Layer (Faint Gray for all standard historical years)
# Excludes highlighted pandemic years, recent baselines, and the current year
exclude_years = [2020, 2021, 2022, current_year - 2, current_year - 1, current_year]
for y in range(start_year, current_year):
    if y in exclude_years:
        continue
    t, v = data_for_year(y)
    ax.plot(t, v, linewidth=0.4, color='#CFD8DC', alpha=0.3)

# Add a single placeholder line to generate a generic history label in the legend
ax.plot([], [], color='#CFD8DC', alpha=0.6, linewidth=1, label=f"History ({start_year}-{current_year-3})")

# 2. Highlight Pandemic Epoch (COVID Years in distinct high-contrast jewel tones)
t_2020, v_2020 = data_for_year(2020)
ax.plot(t_2020, v_2020, label="2020 (COVID)", color='#00838F', linewidth=1.5, linestyle='solid')

t_2021, v_2021 = data_for_year(2021)
ax.plot(t_2021, v_2021, label="2021 (COVID)", color='#00695C', linewidth=1.5, linestyle='solid')

t_2022, v_2022 = data_for_year(2022)
ax.plot(t_2022, v_2022, label="2022 (COVID)", color='#2E7D32', linewidth=1.5, linestyle='solid')

# 3. Recent Dynamic Baselines (Deep Slate/Charcoal shades)
t_2024, v_2024 = data_for_year(current_year - 2)
ax.plot(t_2024, v_2024, label=f"{current_year - 2}", linestyle='dotted', color='#78909C', linewidth=1.5)

t_2025, v_2025 = data_for_year(current_year - 1)
ax.plot(t_2025, v_2025, label=f"{current_year - 1}", linestyle='solid', color='#37474F', linewidth=1.8)

# 4. Active Year Trace (Vibrant Crimson)
t_curr, v_curr = data_for_year(current_year)
ax.plot(t_curr, v_curr, label=f"{current_year}", color='#D32F2F', linewidth=3.2, zorder=10)

# Attach endpoint annotation label
add_terminal_marker(ax, t_curr, v_curr)

ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), frameon=True, facecolor='white', edgecolor='#E0E0E0')
fig.suptitle('Deaths in the Netherlands per week', fontsize=14, fontweight='bold', color='#212121', y=1.04)
ax.set_title(f"{sex}, {leeftijd}\nHighlighting Pandemic anomalies across complete dataset", fontsize=10, color='#616161', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_perjaar.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ==========================================
# 4. HISTORICAL MEDIAN & SPREAD PLOT
# ==========================================
print("Generating statistical threshold chart...")
# Spread aggregates are calculated across the entire history up to the current year
years_historical = df_circle.drop(columns=[current_year], errors='ignore').reindex(range(1, 53))
years_historical = years_historical.interpolate(axis=0)

def close_loop(series):
    arr = series.to_numpy()
    return np.append(arr, arr[0])

theta_stat = np.linspace(0, 2 * np.pi, 53)

mean_val = close_loop(years_historical.mean(axis=1))
median_val = close_loop(years_historical.median(axis=1))
min_val = close_loop(years_historical.min(axis=1))
max_val = close_loop(years_historical.max(axis=1))
sd_val = close_loop(years_historical.std(axis=1))

fig, ax = setup_polar_plot()

# Smooth statistical envelopes over the entire dataset footprint
ax.fill_between(theta_stat, mean_val + sd_val, mean_val - sd_val, alpha=0.25, label="Standard Deviation", color='#90A4AE')
ax.fill_between(theta_stat, min_val, max_val, alpha=0.10, label="Historical Min/Max range", color='#CFD8DC')
ax.plot(theta_stat, median_val, label="Historical Median", linestyle='dashed', color='#455A64', linewidth=1.2)

# Re-overlay specific target highlights for comparison against the total historic spread
ax.plot(t_2020, v_2020, label="2020 (COVID)", color='#00838F', linewidth=1.3, alpha=0.85)
ax.plot(t_2021, v_2021, label="2021 (COVID)", color='#00695C', linewidth=1.3, alpha=0.85)
ax.plot(t_2022, v_2022, label="2022 (COVID)", color='#2E7D32', linewidth=1.3, alpha=0.85)
ax.plot(t_2024, v_2024, label=f"{current_year - 2}", linestyle='dotted', color='#78909C', linewidth=1.2)
ax.plot(t_2025, v_2025, label=f"{current_year - 1}", linestyle='solid', color='#37474F', linewidth=1.5)
ax.plot(t_curr, v_curr, label=f"{current_year}", color='#D32F2F', linewidth=2.5, zorder=10)

# Render marker details on statistical sheet
add_terminal_marker(ax, t_curr, v_curr)

ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), frameon=True, facecolor='white', edgecolor='#E0E0E0')
fig.suptitle(f"Deaths per week in the Netherlands (Full History)", fontsize=14, fontweight='bold', color='#212121', y=1.04)
ax.set_title(f"{sex}, {leeftijd}, historical spread bounds", fontsize=10, color='#616161', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_median.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Execution completed successfully!")
