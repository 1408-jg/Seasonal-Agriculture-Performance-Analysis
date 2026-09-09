
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Display / plotting settings
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

df = pd.read_csv('agriculture_data.csv')
print("Dataset shape:", df.shape)
df.head()


# Explore structure, data types, summary statistics and missing values before doing any analysis.

# %%
df.info()


# %%
df.describe(include='all').T


# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df[missing_df['missing_count'] > 0]


# %%
print("Duplicate rows:", df.duplicated().sum())
print("\nUnique Seasons:", df['Season'].unique())
print("Unique States:", df['State'].nunique())
print("Unique Crops:", df['Crop'].unique())
print("Unique Irrigation Methods:", df['Irrigation_Method'].unique())


# %% [markdown]
# ## 3. Data Cleaning & Preparation
#
# Missing values exist in `Rainfall_mm`, `Soil_Moisture_pct` and `Yield_Tonnes_Ha`.
# Instead of a blanket fill, missing values are imputed using the **median of the same
# Season + Crop group**, which preserves seasonal/crop-specific patterns better than a
# single global median. We also check for outliers and negative/invalid values.

# %%
df_clean = df.copy()

cols_to_impute = ['Rainfall_mm', 'Soil_Moisture_pct', 'Yield_Tonnes_Ha']

for col in cols_to_impute:
    df_clean[col] = df_clean.groupby(['Season', 'Crop'])[col]\
                             .transform(lambda x: x.fillna(x.median()))
    # fallback: if a Season+Crop group was entirely NaN, use overall median
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Remaining missing values after imputation:")
print(df_clean[cols_to_impute].isnull().sum())


# %%
# Outlier check using IQR on key performance columns
key_cols = ['Yield_Tonnes_Ha', 'Profit_INR', 'Water_Efficiency_t_per_1000m3']

for col in key_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    print(f"{col}: {len(outliers)} potential outliers ({len(outliers)/len(df_clean)*100:.1f}%) "
          f"| valid range approx [{lower:.1f}, {upper:.1f}]")

# Outliers are kept (agriculture data legitimately varies widely with real crop failures/
# bumper yields), but flagged for awareness. Extreme negative profits are expected since
# Total_Cost_INR can exceed Revenue_INR for a genuinely loss-making season/crop combination.


# %%
# Derived / helper columns useful for the analysis
df_clean['Profit_Margin_pct'] = (df_clean['Profit_INR'] / df_clean['Revenue_INR'].replace(0, np.nan)) * 100
df_clean['Cost_per_Tonne'] = df_clean['Total_Cost_INR'] / df_clean['Production_Tonnes'].replace(0, np.nan)

season_order = ['Kharif', 'Rabi', 'Zaid']
df_clean['Season'] = pd.Categorical(df_clean['Season'], categories=season_order, ordered=True)

df_clean.to_csv('agriculture_data_cleaned.csv', index=False)
print("Cleaned dataset saved. Shape:", df_clean.shape)
df_clean.head()


# %% [markdown]
# ## 4. Exploratory Data Analysis (Univariate)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

df_clean['Season'].value_counts().reindex(season_order).plot(kind='bar', ax=axes[0], color='#4C8C4A')
axes[0].set_title('Number of Farm Records per Season')
axes[0].set_xlabel('Season'); axes[0].set_ylabel('Count')

df_clean['Crop'].value_counts().plot(kind='bar', ax=axes[1], color='#D98E04')
axes[1].set_title('Number of Records per Crop')
axes[1].set_xlabel('Crop'); axes[1].set_ylabel('Count')

df_clean['Irrigation_Method'].value_counts().plot(kind='bar', ax=axes[2], color='#3E7CB1')
axes[2].set_title('Irrigation Method Usage')
axes[2].set_xlabel('Method'); axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('01_univariate_overview.png', dpi=120)
plt.show()


# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
num_cols = ['Rainfall_mm', 'Avg_Temperature_C', 'Yield_Tonnes_Ha',
            'Profit_INR', 'Water_Efficiency_t_per_1000m3', 'Disease_Pest_Risk_pct']

for ax, col in zip(axes.flatten(), num_cols):
    sns.histplot(df_clean[col], kde=True, ax=ax, color='#4C8C4A')
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('02_numeric_distributions.png', dpi=120)
plt.show()


# %% [markdown]
# ## 5. Seasonal Performance Comparison
#
# Aggregate key environmental, resource and economic indicators by season to answer:
# *"How does agricultural performance vary across seasons?"*

# %%
agg_cols = ['Rainfall_mm', 'Avg_Temperature_C', 'Humidity_pct', 'Soil_Moisture_pct',
            'Fertilizer_kg_ha', 'Pesticide_Litre_ha', 'Water_Used_m3',
            'Yield_Tonnes_Ha', 'Production_Tonnes', 'Total_Cost_INR', 'Revenue_INR',
            'Profit_INR', 'Water_Efficiency_t_per_1000m3', 'Disease_Pest_Risk_pct']

seasonal_summary = df_clean.groupby('Season', observed=True)[agg_cols].mean().round(2)
seasonal_summary


# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

seasonal_summary['Yield_Tonnes_Ha'].plot(kind='bar', ax=axes[0,0], color='#4C8C4A')
axes[0,0].set_title('Average Yield (Tonnes/Ha) by Season'); axes[0,0].set_ylabel('Tonnes/Ha')

seasonal_summary['Profit_INR'].plot(kind='bar', ax=axes[0,1], color='#3E7CB1')
axes[0,1].set_title('Average Profit (INR) by Season'); axes[0,1].set_ylabel('INR')

seasonal_summary['Water_Efficiency_t_per_1000m3'].plot(kind='bar', ax=axes[1,0], color='#0E7C7B')
axes[1,0].set_title('Average Water Efficiency by Season'); axes[1,0].set_ylabel('t / 1000 m3')

seasonal_summary['Disease_Pest_Risk_pct'].plot(kind='bar', ax=axes[1,1], color='#B23A48')
axes[1,1].set_title('Average Disease/Pest Risk (%) by Season'); axes[1,1].set_ylabel('%')

for ax in axes.flatten():
    ax.set_xlabel('Season')
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('03_seasonal_key_metrics.png', dpi=120)
plt.show()


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.boxplot(data=df_clean, x='Season', y='Yield_Tonnes_Ha', order=season_order, ax=axes[0], hue='Season', palette='Greens', legend=False)
axes[0].set_title('Yield Distribution by Season')

sns.boxplot(data=df_clean, x='Season', y='Profit_INR', order=season_order, ax=axes[1], hue='Season', palette='Blues', legend=False)
axes[1].set_title('Profit Distribution by Season')

sns.boxplot(data=df_clean, x='Season', y='Water_Efficiency_t_per_1000m3', order=season_order, ax=axes[2], hue='Season', palette='YlOrBr', legend=False)
axes[2].set_title('Water Efficiency Distribution by Season')

plt.tight_layout()
plt.savefig('04_seasonal_boxplots.png', dpi=120)
plt.show()


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.boxplot(data=df_clean, x='Season', y='Rainfall_mm', order=season_order, ax=axes[0], hue='Season', palette='PuBu', legend=False)
axes[0].set_title('Rainfall by Season')

sns.boxplot(data=df_clean, x='Season', y='Avg_Temperature_C', order=season_order, ax=axes[1], hue='Season', palette='OrRd', legend=False)
axes[1].set_title('Temperature by Season')

sns.boxplot(data=df_clean, x='Season', y='Humidity_pct', order=season_order, ax=axes[2], hue='Season', palette='BuGn', legend=False)
axes[2].set_title('Humidity by Season')

plt.tight_layout()
plt.savefig('05_seasonal_environment.png', dpi=120)
plt.show()


# %% [markdown]
# ## 6. Statistical Testing: Are Seasonal Differences Significant?
#
# A one-way **ANOVA** tests whether the mean of a metric differs significantly across the
# three seasons. Where ANOVA is significant (p < 0.05), a **Tukey HSD** post-hoc test
# identifies which specific season pairs differ.

# %%
def run_anova(metric):
    groups = [df_clean.loc[df_clean['Season'] == s, metric].dropna() for s in season_order]
    f_stat, p_val = stats.f_oneway(*groups)
    sig = "Significant" if p_val < 0.05 else "Not significant"
    print(f"{metric:35s} F = {f_stat:8.2f}   p = {p_val:.4f}   -> {sig} (alpha=0.05)")
    return p_val

print("ANOVA: Does the metric differ significantly across seasons?\n")
anova_metrics = ['Yield_Tonnes_Ha', 'Profit_INR', 'Water_Efficiency_t_per_1000m3',
                  'Disease_Pest_Risk_pct', 'Fertilizer_kg_ha', 'Rainfall_mm']
p_values = {m: run_anova(m) for m in anova_metrics}


# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd

for metric, p in p_values.items():
    if p < 0.05:
        print(f"\nTukey HSD post-hoc test for: {metric}")
        tukey = pairwise_tukeyhsd(endog=df_clean[metric], groups=df_clean['Season'], alpha=0.05)
        print(tukey)


# %% [markdown]
# ## 7. Correlation & Relationship Analysis
#
# Examine relationships between environmental/resource conditions and agricultural outcomes
# (yield, profit), and whether these relationships hold consistently across seasons.

# %%
corr_cols = ['Rainfall_mm', 'Avg_Temperature_C', 'Humidity_pct', 'Sunlight_Hours_Day',
             'Soil_pH', 'Soil_Moisture_pct', 'Nitrogen_kg_ha', 'Phosphorus_kg_ha',
             'Potassium_kg_ha', 'Fertilizer_kg_ha', 'Pesticide_Litre_ha', 'Seed_Quality_Score',
             'Water_Used_m3', 'Yield_Tonnes_Ha', 'Profit_INR', 'Water_Efficiency_t_per_1000m3',
             'Disease_Pest_Risk_pct']

corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(13, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, cbar_kws={'label': 'Correlation coefficient'})
plt.title('Correlation Heatmap: Environmental, Resource & Outcome Variables')
plt.tight_layout()
plt.savefig('06_correlation_heatmap.png', dpi=120)
plt.show()


# %%
print("Top correlations with Yield_Tonnes_Ha:")
print(corr_matrix['Yield_Tonnes_Ha'].drop('Yield_Tonnes_Ha').sort_values(ascending=False))

print("\nTop correlations with Profit_INR:")
print(corr_matrix['Profit_INR'].drop('Profit_INR').sort_values(ascending=False))


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, season, color in zip(axes, season_order, ['#4C8C4A', '#3E7CB1', '#D98E04']):
    sub = df_clean[df_clean['Season'] == season]
    sns.regplot(data=sub, x='Rainfall_mm', y='Yield_Tonnes_Ha', ax=ax,
                scatter_kws={'alpha':0.3, 's':15, 'color': color}, line_kws={'color':'black'})
    ax.set_title(f'Rainfall vs Yield — {season}')

plt.tight_layout()
plt.savefig('07_rainfall_vs_yield_by_season.png', dpi=120)
plt.show()


# %% [markdown]
# ## 8. Crop-wise & Region-wise Seasonal Patterns
#
# Check whether seasonal patterns are consistent across different crops and states, or
# whether some crops/regions behave differently.

# %%
crop_season_yield = df_clean.pivot_table(values='Yield_Tonnes_Ha', index='Crop',
                                          columns='Season', aggfunc='mean', observed=True).round(2)
crop_season_yield


# %%
plt.figure(figsize=(12, 7))
sns.heatmap(crop_season_yield, annot=True, fmt='.2f', cmap='YlGn', linewidths=0.5,
            cbar_kws={'label': 'Avg Yield (Tonnes/Ha)'})
plt.title('Average Yield by Crop and Season')
plt.tight_layout()
plt.savefig('08_crop_season_yield_heatmap.png', dpi=120)
plt.show()


# %%
crop_season_profit = df_clean.pivot_table(values='Profit_INR', index='Crop',
                                           columns='Season', aggfunc='mean', observed=True).round(0)

plt.figure(figsize=(12, 7))
sns.heatmap(crop_season_profit, annot=True, fmt='.0f', cmap='RdYlGn', center=0, linewidths=0.5,
            cbar_kws={'label': 'Avg Profit (INR)'})
plt.title('Average Profit by Crop and Season')
plt.tight_layout()
plt.savefig('09_crop_season_profit_heatmap.png', dpi=120)
plt.show()


# %%
top_states = df_clean['State'].value_counts().head(8).index
state_season = df_clean[df_clean['State'].isin(top_states)].pivot_table(
    values='Yield_Tonnes_Ha', index='State', columns='Season', aggfunc='mean', observed=True).round(2)

state_season.plot(kind='bar', figsize=(13, 6), color=['#4C8C4A', '#3E7CB1', '#D98E04'])
plt.title('Average Yield by State (Top 8 States) Across Seasons')
plt.ylabel('Avg Yield (Tonnes/Ha)')
plt.xlabel('State')
plt.legend(title='Season')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('10_state_season_yield.png', dpi=120)
plt.show()


# %% [markdown]
# ## 9. Resource Usage & Efficiency Analysis
#
# Compare fertilizer, pesticide and water usage across seasons, and how efficiently that
# usage converts into yield/profit.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=df_clean, x='Season', y='Fertilizer_kg_ha', order=season_order, ax=axes[0],
            estimator=np.mean, hue='Season', palette='YlOrBr', legend=False)
axes[0].set_title('Avg Fertilizer Use by Season')

sns.barplot(data=df_clean, x='Season', y='Pesticide_Litre_ha', order=season_order, ax=axes[1],
            estimator=np.mean, hue='Season', palette='RdPu', legend=False)
axes[1].set_title('Avg Pesticide Use by Season')

sns.barplot(data=df_clean, x='Season', y='Water_Used_m3', order=season_order, ax=axes[2],
            estimator=np.mean, hue='Season', palette='PuBu', legend=False)
axes[2].set_title('Avg Water Used by Season')

plt.tight_layout()
plt.savefig('11_resource_usage_by_season.png', dpi=120)
plt.show()


# %%
irrigation_season = pd.crosstab(df_clean['Season'], df_clean['Irrigation_Method'], normalize='index') * 100
irrigation_season = irrigation_season.round(1)
print("Irrigation method share (%) within each season:")
irrigation_season


# %%
irrigation_season.plot(kind='bar', stacked=True, figsize=(10, 6),
                        colormap='Set2')
plt.title('Irrigation Method Mix by Season')
plt.ylabel('% of farms')
plt.xlabel('Season')
plt.xticks(rotation=0)
plt.legend(title='Irrigation Method', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('12_irrigation_mix_by_season.png', dpi=120)
plt.show()


# %%
irrigation_efficiency = df_clean.groupby(['Season', 'Irrigation_Method'], observed=True)[
    'Water_Efficiency_t_per_1000m3'].mean().unstack().round(2)
irrigation_efficiency


