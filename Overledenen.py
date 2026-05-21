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
df_circle = df_circle.pivot(index='week', columns='year', values='deaths')


# ==========================================
# 2. POLAR ROUTINES & MATHEMATICAL BRIDGING
# ==========================================
def data_for_year(y):
    """
    Extracts data frames strictly mapped out over a true 52-week calendar distribution.
    Prevents artificial drops or forward predicting timeline leakage.
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


def setup_polar_plot(figsize=(10, 8), blog_mode=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Modern Blog Light Theme vs Standard Academic Viewport
    bg_canvas = '#F7FAFC' if blog_mode else 'white'
    bg_face = '#FFFFFF' if blog_mode else 'white'
    grid_color = '#E2E8F0' if blog_mode else '#E5E5E5'
    text_color = '#4A5568' if blog_mode else '#333333'
    spine_color = '#CBD5E0' if blog_mode else '#888888'

    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_face)

    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels(months, fontsize=10, color=text_color, fontweight='bold' if blog_mode else 'normal')
    ax.tick_params(axis='x', pad=12)

    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.6)
    ax.set_rlabel_position(180)
    ax.tick_params(axis='y', colors=text_color)

    ax.spines['polar'].set_color(spine_color)
    ax.spines['polar'].set_linewidth(1.0 if blog_mode else 0.8)
    return fig, ax


def add_terminal_marker(ax, theta, values, blog_mode=False):
    """Places a precise data node point and explicit text overlay label at the leading timeline edge."""
    if len(theta) == 0 or len(values) == 0:
        return
    t_last = theta[-1]
    v_last = values[-1]

    brand_color = '#FF3366' if blog_mode else '#D32F2F'
    bg_box = '#FFFFFF'

    ax.plot(t_last, v_last, marker='o', markersize=8, color=brand_color, markeredgecolor='white', markeredgewidth=2, zorder=15)

    text_padding = 160
    ha = 'left' if 0 <= t_last < np.pi else 'right'

    ax.annotate(
        f"Week {len(values)}\n{int(v_last):,} deaths",
        xy=(t_last, v_last),
        xytext=(text_padding if ha == 'left' else -text_padding, 0),
        textcoords='offset points',
        ha=ha,
        va='center',
        fontsize=9,
        fontweight='bold',
        color=brand_color,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_box, edgecolor=brand_color, alpha=0.95, linewidth=1.2),
        zorder=16
    )


# pre-calculate shared arrays to streamline runtime execution
t_2020, v_2020 = data_for_year(2020)
t_2021, v_2021 = data_for_year(2021)
t_2022, v_2022 = data_for_year(2022)
t_2024, v_2024 = data_for_year(current_year - 2)
t_2025, v_2025 = data_for_year(current_year - 1)
t_curr, v_curr = data_for_year(current_year)


# ==========================================
# 3. ORIGINAL SCIENTIFIC OVERLAY PLOTS
# ==========================================
print("Generating clean scientific overlay charts...")
fig, ax = setup_polar_plot(blog_mode=False)

exclude_years = [2020, 2021, 2022, current_year - 2, current_year - 1, current_year]
for y in range(start_year, current_year):
    if y in exclude_years:
        continue
    t, v = data_for_year(y)
    ax.plot(t, v, linewidth=0.4, color='#CFD8DC', alpha=0.4)

ax.plot([], [], color='#CFD8DC', alpha=0.7, linewidth=1, label=f"History ({start_year}-{current_year-3})")

# COVID highlights (Academic deep jewel tones)
ax.plot(t_2020, v_2020, label="2020 (COVID)", color='#00838F', linewidth=1.5)
ax.plot(t_2021, v_2021, label="2021 (COVID)", color='#00695C', linewidth=1.5)
ax.plot(t_2022, v_2022, label="2022 (COVID)", color='#2E7D32', linewidth=1.5)

# Recent Tracking Context Lines
ax.plot(t_2024, v_2024, label=f"{current_year - 2}", linestyle='dotted', color='#78909C', linewidth=1.5)
ax.plot(t_2025, v_2025, label=f"{current_year - 1}", linestyle='solid', color='#37474F', linewidth=1.8)
ax.plot(t_curr, v_curr, label=f"{current_year}", color='#D32F2F', linewidth=3.2, zorder=10)

add_terminal_marker(ax, t_curr, v_curr, blog_mode=False)

ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), frameon=True, facecolor='white', edgecolor='#E0E0E0')
fig.suptitle('Deaths in the Netherlands per week', fontsize=14, fontweight='bold', color='#212121', y=1.04)
ax.set_title(f"{sex}, {leeftijd}\nHighlighting Pandemic anomalies across complete historical record", fontsize=10, color='#616161', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_perjaar.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# ==========================================
# 4. HISTORICAL MEDIAN & SPREAD PLOT
# ==========================================
print("Generating statistical spread thresholds...")
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

fig, ax = setup_polar_plot(blog_mode=False)
ax.fill_between(theta_stat, mean_val + sd_val, mean_val - sd_val, alpha=0.25, label="Standard Deviation", color='#90A4AE')
ax.fill_between(theta_stat, min_val, max_val, alpha=0.10, label="Historical Range Bound", color='#CFD8DC')
ax.plot(theta_stat, median_val, label="Historical Median", linestyle='dashed', color='#455A64', linewidth=1.2)

ax.plot(t_2020, v_2020, label="2020 (COVID)", color='#00838F', linewidth=1.3, alpha=0.85)
ax.plot(t_2021, v_2021, label="2021 (COVID)", color='#00695C', linewidth=1.3, alpha=0.85)
ax.plot(t_2022, v_2022, label="2022 (COVID)", color='#2E7D32', linewidth=1.3, alpha=0.85)
ax.plot(t_2024, v_2024, label=f"{current_year - 2}", linestyle='dotted', color='#78909C', linewidth=1.2)
ax.plot(t_2025, v_2025, label=f"{current_year - 1}", linestyle='solid', color='#37474F', linewidth=1.5)
ax.plot(t_curr, v_curr, label=f"{current_year}", color='#D32F2F', linewidth=2.5, zorder=10)

add_terminal_marker(ax, t_curr, v_curr, blog_mode=False)

ax.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), frameon=True, facecolor='white', edgecolor='#E0E0E0')
fig.suptitle(f"Deaths per week in the Netherlands (Full Dataset)", fontsize=14, fontweight='bold', color='#212121', y=1.04)
ax.set_title(f"{sex}, {leeftijd}, complete spread distribution analysis", fontsize=10, color='#616161', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_median.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# ==========================================
# 5. BLOG VERSIONS (MODERN LIGHT EDITORIAL)
# ==========================================
print("Generating light-themed blog assets...")

# --- BLOG GRAPHIC 1: OVERLAY HISTORIC TRENDS ---
fig, ax = setup_polar_plot(blog_mode=True)

# 1. Base Layer History (Clean muted slate grays)
for y in range(start_year, current_year):
    if y in exclude_years:
        continue
    t, v = data_for_year(y)
    ax.plot(t, v, linewidth=0.5, color='#CBD5E0', alpha=0.5)
ax.plot([], [], color='#CBD5E0', alpha=0.8, linewidth=1, label=f"Historical Baseline ({start_year}-{current_year-3})")

# 2. Pandemic Waves (Hip, vivid editorial jewel tones)
ax.plot(t_2020, v_2020, label="2020 Pandemic Outbreak", color='#0EA5E9', linewidth=1.6) # Vibrant Light Blue
ax.plot(t_2021, v_2021, label="2021 Delta Wave", color='#10B981', linewidth=1.6)       # Vibrant Emerald Green
ax.plot(t_2022, v_2022, label="2022 Omicron Structural Shift", color='#6366F1', linewidth=1.6) # Indigo Purple

# 3. Recent Comparisons
ax.plot(t_2024, v_2024, label=f"{current_year - 2} Trend", linestyle='dotted', color='#718096', linewidth=1.5)
ax.plot(t_2025, v_2025, label=f"{current_year - 1} Trend", linestyle='solid', color='#4A5568', linewidth=1.8)

# 4. Highlight Active Tracking Line (Ultra Neon Hot Coral Pink)
ax.plot(t_curr, v_curr, label=f"Current Tracker ({current_year})", color='#FF3366', linewidth=3.5, zorder=10)
add_terminal_marker(ax, t_curr, v_curr, blog_mode=True)

legend = ax.legend(loc='lower right', bbox_to_anchor=(1.32, -0.05), frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')
for text in legend.get_texts():
    text.set_color('#2D3748')
    text.set_fontsize(9.5)

fig.suptitle('Tracking Mortality Trends in the Netherlands', fontsize=16, fontweight='bold', color='#1A202C', y=1.05)
ax.set_title(f"Demographic Filters: {sex} | {leeftijd}", fontsize=10, color='#718096', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_perjaar_blog.{suffix}', dpi=300, bbox_inches='tight', facecolor='#F7FAFC')
plt.close()


# --- BLOG GRAPHIC 2: MEDIAN AND SHADED ENVELOPE ---
print("Generating light-themed statistical median blog assets...")
fig, ax = setup_polar_plot(blog_mode=True)

# Clean, soft pastel editorial band distributions
ax.fill_between(theta_stat, mean_val + sd_val, mean_val - sd_val, alpha=0.35, label="Standard Deviation Bounds", color='#CBD5E0')
ax.fill_between(theta_stat, min_val, max_val, alpha=0.15, label="Historical Minimum / Maximum", color='#E2E8F0')
ax.plot(theta_stat, median_val, label="Historical Median Trend", linestyle='dashed', color='#2D3748', linewidth=1.5)

# Re-overlay key timeline indicators
ax.plot(t_2020, v_2020, label="2020 Pandemic Outbreak", color='#0EA5E9', linewidth=1.2, alpha=0.7)
ax.plot(t_2021, v_2021, label="2021 Delta Wave", color='#10B981', linewidth=1.2, alpha=0.7)
ax.plot(t_2022, v_2022, label="2022 Omicron Structural Shift", color='#6366F1', linewidth=1.2, alpha=0.7)
ax.plot(t_2024, v_2024, label=f"{current_year - 2} Trend", linestyle='dotted', color='#718096', linewidth=1.2)
ax.plot(t_2025, v_2025, label=f"{current_year - 1} Trend", linestyle='solid', color='#4A5568', linewidth=1.5)
ax.plot(t_curr, v_curr, label=f"Current Tracker ({current_year})", color='#FF3366', linewidth=3.0, zorder=10)
add_terminal_marker(ax, t_curr, v_curr, blog_mode=True)

legend = ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05), frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')
for text in legend.get_texts():
    text.set_color('#2D3748')
    text.set_fontsize(9.5)

fig.suptitle('Historical Mortality Spread Context', fontsize=16, fontweight='bold', color='#1A202C', y=1.05)
ax.set_title(f"Demographic Filters: {sex} | {leeftijd}", fontsize=10, color='#718096', y=1.1)

for suffix in ['png', 'svg']:
    plt.savefig(f'sterfte_median_blog.{suffix}', dpi=300, bbox_inches='tight', facecolor='#F7FAFC')
plt.close()

print("Execution completed successfully! Regular and light blog variants are saved.")
