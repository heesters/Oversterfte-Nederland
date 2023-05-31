import numpy as np
import pandas as pd
import cbsodata
import seaborn as sns
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

data = pd.DataFrame(cbsodata.get_data('70895ned'))
data.dropna(subset = ["Overledenen_1"], inplace=True)
df_clean=data[data.Perioden.str.contains('week')]
df_clean=df_clean[df_clean.Perioden.str.contains('1995 week 0')==False].reset_index(drop=True)

df_clean = df_clean.drop(columns = ['ID'])
df_clean['to_first_week']=df_clean.Perioden.str.contains('dag') & df_clean.Perioden.str.contains('week 1')
df_clean['to_last_week']=df_clean.Perioden.str.contains('dag') & df_clean.Perioden.shift(-1).str.contains('week 0')
df_clean['partial_week']=df_clean.Perioden.str.contains('dag')
df_clean.loc[df_clean['to_first_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(+1) + df_clean['Overledenen_1'] 
df_clean.loc[df_clean['to_last_week'] == True, 'deaths'] = df_clean['Overledenen_1'].shift(-1) + df_clean['Overledenen_1'] 
df_clean.loc[df_clean['partial_week'] == False, 'deaths'] = df_clean['Overledenen_1']
df_clean = df_clean.dropna(subset = ["deaths"]).reset_index(drop=True)
df_clean[['year','week']] = df_clean.Perioden.str.split("week",expand=True)
df_clean['week'] = df_clean.week.str.extract('(\d+)')
df_clean['year'] = df_clean.year.str.extract('(\d+)')
df_clean['week'] =pd.to_numeric(df_clean['week']).values.astype(int)
df_clean['year'] =pd.to_numeric(df_clean['year']).values.astype(int)
df_clean['deaths'] =pd.to_numeric(df_clean['deaths']).values.astype(int)
df_clean = df_clean.drop(columns = ['Overledenen_1','to_first_week','to_last_week','partial_week'])
df_clean = df_clean.rename(columns={"LeeftijdOp31December": "age", "Geslacht": "gender"})
df_clean = df_clean[['Perioden','gender','age','year','week','deaths']]

df_clean=df_clean[df_clean.Perioden >= '2010'].reset_index(drop=True)

df_clean['covid_year']=df_clean['year'] >= 2020
df_clean.loc[df_clean['covid_year'] == False, 'covid_year'] = '2010-2019 +/- SD'
df_clean.loc[df_clean['covid_year'] == True, 'covid_year'] = df_clean['year']

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
current_year = df_clean['year'].max()

g = sns.FacetGrid(df_clean, col="gender", hue="covid_year", palette=["dodgerblue","red","orange", "green"], row='age', aspect=2,sharey=False)
g.map(sns.lineplot, 'week', 'deaths', alpha=.7, estimator='mean', errorbar='sd')
g.set(xlabel="month", ylabel = "deaths per week", xticks=np.arange(1, 53,(53/12) ), xticklabels=months)
g.add_legend(title = '')
for suffix in 'png svg'.split():
    g.savefig('naar_Geslacht_leeftijd.'+suffix, dpi=200, bbox_inches='tight', facecolor='white')

leeftijd='Totaal leeftijd'
sex='Totaal mannen en vrouwen'
df_circle=df_clean[(df_clean.age == leeftijd) & (df_clean.gender == sex)]
df_circle=df_circle.pivot(index='week', columns='year', values='deaths')

def data_for_year(y):
    year = df_circle[y].dropna().to_numpy()
    if y == int(current_year):
        num_weeks = len(year)
        day_of_the_year = num_weeks*7 + 3 # ex. week 46 -> november 15 -> day 319
        theta = np.linspace(0, (day_of_the_year/365)*2*np.pi, num_weeks)
    else:
        # append first week of next year for correct radial plotting
        year = np.append(year, df_circle.loc[1, y+1])
        theta = np.linspace(0, 2*np.pi, len(year))
    return (theta, year)

def plot_year(ax, y, **kwargs):
    ax.plot(*data_for_year(y), label=f"{y}", **kwargs)

def setup_polar_plot(figsize=(8, 6), constrained_layout=True):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(0, 2*np.pi, np.pi/6))
    ax.set_xticklabels(months)

    ax.set_rlabel_position(180)
#   ax.set_yticklabels(['1000', '2000', '3000', '4000', '5000', ''])

    return fig, ax

fig, ax = setup_polar_plot()
plot_year(ax, int(current_year)-10, linewidth=0.1)
plot_year(ax, int(current_year)-9, linewidth=0.2)
plot_year(ax, int(current_year)-8, linewidth=0.3)
plot_year(ax, int(current_year)-7, linewidth=0.4)
plot_year(ax, int(current_year)-6, linewidth=0.5)
plot_year(ax, int(current_year)-5, linewidth=0.6)
plot_year(ax, int(current_year)-4, linewidth=0.7)
plot_year(ax, int(current_year)-3, linewidth=0.8)
plot_year(ax, int(current_year)-2, color='tab:red', linewidth=1, linestyle='dotted')
plot_year(ax, int(current_year)-1, color='tab:orange', linewidth=2)
plot_year(ax, int(current_year), color='tab:green', linewidth=3)

fig.legend(loc='lower right')
fig.suptitle('Deaths in the Netherlands per week', fontsize=14, y=1.04)
ax.set_title(f"{sex}, {leeftijd}", fontsize=10, y=1.1)
for suffix in 'png svg'.split():
    plt.savefig('sterfte_perjaar.'+suffix, dpi=200, bbox_inches='tight', facecolor='white')

years = df_circle.loc[:, ~df_circle.columns.isin([2020, 2021, 2022, int(current_year)])] # excluding corona years and current year (for now)

mean = years.mean(skipna=True,axis=1)
mean[53] = mean[1]

median = years.median(skipna=True,axis=1)
median[53] = median[1]

min = years.min(axis=1)
min[53] = min[1]

max = years.max(axis=1)
max[53] = max[1]

sd = years.std(axis=1)
sd[53] = sd[1]

q25 = df_circle.astype(float).quantile(0.25, axis=1)
q25[53] = q25[1]

q75 = df_circle.astype(float).quantile(0.75, axis=1)
q75[53] = q75[1]

fig, ax = setup_polar_plot()

ax.fill_between(np.linspace(0, 2*np.pi, len(min)), mean+sd, mean-sd, alpha=0.3, label="SD", color='tab:blue')

ax.fill_between(np.linspace(0, 2*np.pi, len(min)), min, max, alpha=0.2, label="Min/Max")

ax.plot(np.linspace(0, 2*np.pi, len(median)), median, label="Median", linestyle='dashed')
plot_year(ax, int(current_year)-3, color='tab:red', linewidth=2, linestyle='dotted')
plot_year(ax, int(current_year)-2, color='tab:orange', linewidth=2, linestyle='dotted')
plot_year(ax, int(current_year)-1, color='tab:orange', linewidth=2)
plot_year(ax, int(current_year), color='tab:green', linewidth=3)

fig.legend(loc='lower right')
fig.suptitle(f"Deaths per week in the Netherlands (since 2010)", fontsize=14, y=1.04)
ax.set_title(f"{sex}, {leeftijd}, median excludes 2020-2023", fontsize=10, y=1.1)
for suffix in 'png svg'.split():
    plt.savefig('sterfte_median.'+suffix, dpi=200, bbox_inches='tight', facecolor='white')


start_year = 2010

# Function to setup polar plot
def setup_polar_plot(figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    pos = ax.get_position()
    pos.y0 -= 0.05
    pos.y1 -= 0.05
    pos.x0 -= 0.012
    pos.x1 -= 0.012
    ax.set_position(pos)
    fig.suptitle("Deaths per week in the Netherlands (since 2010)", fontsize=14)
    ax.set_title(f"{sex}, {leeftijd}", fontsize=10, y=1.1)
    return fig, ax

# Dummy functions for missing data_for_year
def data_for_year(year):
    return np.random.rand(12), np.random.rand(12)

# Initialize plot elements
def init():
    old.set_data([], []) 
    prev.set_data([], []) 
    current.set_data([], []) 
    center.set_text("")
    return old, prev, current, center

# Define helper functions
def year_and_week_for_index(i):
    y = start_year
    while True:
        len_year = len(df_circle[y].dropna()) + 1
        if len_year > i:
            return (y, i+1)
        else:
            y += 1
            i -= (len_year-1)

def data_for_index(i):
    y, w = year_and_week_for_index(i)
    theta, year = data_for_year(y)
    return theta[:w], year[:w]

# Animate function
def animate(i):
    y = year_and_week_for_index(i)[0]

    if y > start_year:
        old_theta = np.concatenate([data_for_year(year)[0] for year in range(start_year, y-1)])
        old_data = np.concatenate([data_for_year(year)[1] for year in range(start_year, y-1)])
        old.set_data(old_theta, old_data)
        prev.set_data(*data_for_year(y-1))

    current.set_data(*data_for_index(i))
    center.set_text(f"{y}")
    return old, prev, current, center

# Create animation
fig, ax = setup_polar_plot(figsize=(6, 6.2))

old, = ax.plot([], [], color='tab:blue', linewidth=0.5, linestyle='dotted', label="2010-2019")
prev, = ax.plot([], [], color='tab:orange', label=int(current_year)-1)
current, = ax.plot([], [], color='tab:green', linewidth=3, label=int(current_year))
center = ax.text(0, 25, "5000", horizontalalignment='center', fontsize=18)
ax.set_rmax(5500)

# Number of frames
num_frames = len(df_circle)

# Create animation
fig.tight_layout()
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True) 
anim.save('sterfte_anim.gif', writer='pillow', fps=50, dpi=72)
anim.save('sterfte_anim.mp4', writer='ffmpeg', dpi=300, extra_args=['-vf', 'tpad=stop_mode=clone:stop_duration=5'])
