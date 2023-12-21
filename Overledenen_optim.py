import numpy as np
import pandas as pd
import cbsodata
import matplotlib.pyplot as plt

# Data Retrieval and Cleaning
data = pd.DataFrame(cbsodata.get_data('70895ned'))
data.dropna(subset=["Overledenen_1"], inplace=True)

# Filter and clean data
df_clean = (
    data[data.Perioden.str.contains('week')]
    .loc[~data.Perioden.str.contains('1995 week 0')]
    .drop(columns=['ID'])
    .reset_index(drop=True)
)

# Calculate weekly death counts using numpy vectorized operations
df_clean['to_first_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.str.contains('week 1')
df_clean['to_last_week'] = df_clean.Perioden.str.contains('dag') & df_clean.Perioden.shift(-1).str.contains('week 0')
df_clean['partial_week'] = df_clean.Perioden.str.contains('dag')
df_clean['deaths'] = (
    np.where(df_clean['to_first_week'], df_clean['Overledenen_1'].shift(+1) + df_clean['Overledenen_1'], 0) +
    np.where(df_clean['to_last_week'], df_clean['Overledenen_1'].shift(-1) + df_clean['Overledenen_1'], 0) +
    np.where(~df_clean['to_first_week'] & ~df_clean['to_last_week'] & ~df_clean['partial_week'], df_clean['Overledenen_1'], 0)
)

# Filter and rename columns
df_clean = (
    df_clean.dropna(subset=["deaths"])
    .assign(year=df_clean.Perioden.str.extract('(\d+) week (\d+)')[0].astype(int),
            week=df_clean.Perioden.str.extract('(\d+) week (\d+)')[1].astype(int))
    .drop(columns=['Overledenen_1', 'to_first_week', 'to_last_week', 'partial_week'])
    .rename(columns={"LeeftijdOp31December": "age", "Geslacht": "gender"})
    .loc[df_clean.Perioden >= '2010']
    .reset_index(drop=True)
)

# Categorize data based on COVID-19 years
df_clean['covid_year'] = np.where(df_clean['year'] >= 2020, df_clean['year'], '2010-2019 +/- SD')

# Visualization using Seaborn and Matplotlib
current_year = df_clean['year'].max()

# Polar Plot for Total Age and Total Gender
leeftijd = 'Totaal leeftijd'
sex = 'Totaal mannen en vrouwen'
df_circle = df_clean[(df_clean.age == leeftijd) & (df_clean.gender == sex)]
df_circle = df_circle.pivot(index='week', columns='year', values='deaths')

# Statistical Analysis and Visualization on Polar Plot
years_except_current = df_circle.loc[:, ~df_circle.columns.isin([current_year])]

# Calculate statistical measures
median = years_except_current.median(skipna=True, axis=1)
median[53] = median[1]

# Setup polar plot for statistical measures
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))

# Plot median as dashed line
ax.plot(np.linspace(0, 2 * np.pi, len(median)), median, label="Median", linestyle='dashed')

# Plot individual years
years_to_plot = [2020, 2021, 2022, current_year]
for i in years_to_plot:
    year_data = df_circle[i].dropna().to_numpy()
    ax.plot(np.linspace(0, (len(year_data) / 52) * 2 * np.pi, len(year_data)), year_data,
            label=f"{i}", linewidth=i - (int(current_year) - 2) if i >= int(current_year) - 2 else 2, linestyle='dotted')

# Add legend and titles
ax.legend(loc='lower right')
fig.suptitle(f"Deaths per week in the Netherlands (since 2010)", fontsize=14, y=1.04)
ax.set_title(f"{sex}, {leeftijd}, median", fontsize=10, y=1.1)

# Save the polar plot with statistical measures
for suffix in 'png svg'.split():
    plt.savefig('sterfte_median_optim.' + suffix, dpi=200, bbox_inches='tight')
