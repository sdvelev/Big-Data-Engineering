import pandas as pd
import pyspark 
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from os.path import join

spark = SparkSession.builder.appName("Coursework 1 - Socioeconomic Analysis").master("local[*]").getOrCreate()
sc = spark.sparkContext

path_data_1 = "hdfs:///coursework-1/happiness-cantril-ladder.csv"
df_life_satisfaction = spark.read.format("csv") \
  .option("sep", ",") \
  .option("inferSchema", "true") \
  .option("header", "true") \
  .load(path_data_1)

df_life_satisfaction.show()
df_life_satisfaction.printSchema()

print("The number of rows in the life satisfaction data frame is:", df_life_satisfaction.count())

path_data_2 = "hdfs:///coursework-1/extreme-poverty-headcount-ratio-vs-life-expectancy-at-birth.csv"
df_extreme_poverty_life_expectancy = spark.read.format("csv") \
  .option("sep", ",") \
  .option("inferSchema", "true") \
  .option("header", "true") \
  .load(path_data_2)
  
df_extreme_poverty_life_expectancy.show()
df_extreme_poverty_life_expectancy.printSchema()

print("The number of rows in the extreme poverty vs. life expectancy data frame is:", df_extreme_poverty_life_expectancy.count())

print(df_extreme_poverty_life_expectancy.columns)

path_data_3 = "hdfs:///coursework-1/political-corruption-index.csv"
df_political_corruption = spark.read.format("csv") \
  .option("sep", ",") \
  .option("inferSchema", "true") \
  .option("header", "true") \
  .load(path_data_3)

df_political_corruption.show()
df_political_corruption.printSchema()

print("The number of rows in the political corruption data frame is:", df_political_corruption.count())

df_life_satisfaction = (
    df_life_satisfaction
    .select('Entity', 'Year', 'Cantril ladder score')
    .withColumnRenamed('Entity', 'Country')
    .filter(col('Year') == 2021)
    .na.drop()
)

unwanted_countries = [
    'High-income countries',
    'Low-income countries',
    'Lower-middle-income countries',
    'Upper-middle-income countries',
    'World'
]

df_life_satisfaction = df_life_satisfaction.filter(~col('Country').isin(unwanted_countries))

df_life_satisfaction.show()

df_extreme_poverty_life_expectancy = df_extreme_poverty_life_expectancy.select(
    col('Entity'),
    col('Year'),
    col('Life expectancy - Sex: all - Age: 0 - Variant: estimates'),
    col('`$2.15 a day - Share of population in poverty`'),
    col('Population (historical)')
).withColumnRenamed('Entity', 'Country') \
 .withColumnRenamed('Life expectancy - Sex: all - Age: 0 - Variant: estimates', 'Life expectancy') \
 .withColumnRenamed('$2.15 a day - Share of population in poverty', 'Share in extreme poverty') \
 .withColumnRenamed('Population (historical)', 'Population') \
 .filter(col('Year') == 2021) \
 .na.drop()
 
# Remove not-country-specific entries
df_extreme_poverty_life_expectancy = df_extreme_poverty_life_expectancy.filter(col('Country') != 'World')

# Correct the Central African Republic life expectancy according the World Health Organization Data for 2021
# Source: https://data.who.int/countries/140
df_extreme_poverty_life_expectancy = df_extreme_poverty_life_expectancy.withColumn(
    "Life expectancy",
    when(col("Country") == "Central African Republic", 52.31).otherwise(col("Life expectancy"))
)

df_extreme_poverty_life_expectancy.show()

df_political_corruption = df_political_corruption.select(
    col('Entity'),
    col('Year'),
    col('Political corruption index (best estimate, aggregate: average)')
).withColumnRenamed('Entity', 'Country') \
 .withColumnRenamed('Political corruption index (best estimate, aggregate: average)', 'Political corruption index') \
 .filter(col('Year') == 2021) \
 .na.drop()
 
# Remove not-country-specific entries
df_political_corruption = df_political_corruption.filter(col('Country') != 'World')

df_political_corruption.show()

continents = ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America']

# Extract only continent entries
df_life_satisfaction_continents = df_life_satisfaction.filter(col('Country').isin(continents))

# Remove continent entries (but keep 'Australia' as it is a country too)
df_life_satisfaction = df_life_satisfaction.filter(
    (~col('Country').isin(continents)) | (col('Country') == 'Australia')
)

df_life_satisfaction.show()
df_life_satisfaction.count()

df_life_satisfaction_continents.show()

# Extract only continent entries
df_political_corruption_continents = df_political_corruption.filter(col('Country').isin(continents))

# Remove continent entries (but keep 'Australia')
df_political_corruption = df_political_corruption.filter(
    (~col('Country').isin(continents)) | (col('Country') == 'Australia')
)

df_political_corruption.show()
df_political_corruption.count()

df_political_corruption_continents.show()

df_extreme_poverty_life_expectancy.summary().show()

# Convert PySpark DataFrame to pandas DataFrame
df_extreme_poverty_life_expectancy = df_extreme_poverty_life_expectancy.toPandas()

df_extreme_poverty_life_expectancy.plot.scatter(x='Share in extreme poverty', y='Life expectancy')
plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (world countries) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
plt.scatter(df_extreme_poverty_life_expectancy['Share in extreme poverty'], df_extreme_poverty_life_expectancy['Life expectancy'], color='blue', s = df_extreme_poverty_life_expectancy['Population']/1000000, alpha=0.5)

for i, country in enumerate(df_extreme_poverty_life_expectancy['Country']):
    plt.text(df_extreme_poverty_life_expectancy['Share in extreme poverty'].iloc[i], df_extreme_poverty_life_expectancy['Life expectancy'].iloc[i], df_extreme_poverty_life_expectancy['Country'].iloc[i], fontsize=6, ha='right')

plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (world countries) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
df_low_extreme_poverty = df_extreme_poverty_life_expectancy[df_extreme_poverty_life_expectancy['Share in extreme poverty'] <= 0.5]
plt.scatter(df_low_extreme_poverty['Share in extreme poverty'], df_low_extreme_poverty['Life expectancy'], color='blue', s = df_low_extreme_poverty['Population']/200000, alpha=0.5)

for i, country in enumerate(df_low_extreme_poverty['Country']):
    plt.text(df_low_extreme_poverty['Share in extreme poverty'].iloc[i], df_low_extreme_poverty['Life expectancy'].iloc[i], df_low_extreme_poverty['Country'].iloc[i], fontsize=7, ha='right')

plt.yticks(range(69, 85))

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with share in extreme poverty less than 0.5%) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
df_middle_extreme_poverty = df_extreme_poverty_life_expectancy[(df_extreme_poverty_life_expectancy['Share in extreme poverty'] >= 0.5) & (df_extreme_poverty_life_expectancy['Share in extreme poverty'] <= 5)]
plt.scatter(df_middle_extreme_poverty['Share in extreme poverty'], df_middle_extreme_poverty['Life expectancy'], color='blue', s = df_middle_extreme_poverty['Population']/200000, alpha=0.5)

for i, country in enumerate(df_middle_extreme_poverty['Country']):
    plt.text(df_middle_extreme_poverty['Share in extreme poverty'].iloc[i], df_middle_extreme_poverty['Life expectancy'].iloc[i], df_middle_extreme_poverty['Country'].iloc[i], fontsize=7, ha='right')

plt.yticks(range(60, 85, 2))

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with share in extreme poverty between 0.5% and 5%) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
df_high_extreme_poverty = df_extreme_poverty_life_expectancy[df_extreme_poverty_life_expectancy['Share in extreme poverty'] >= 5]
plt.scatter(df_high_extreme_poverty['Share in extreme poverty'], df_high_extreme_poverty['Life expectancy'], color='blue', s = df_high_extreme_poverty['Population']/200000, alpha=0.5)

for i, country in enumerate(df_high_extreme_poverty['Country']):
    plt.text(df_high_extreme_poverty['Share in extreme poverty'].iloc[i], df_high_extreme_poverty['Life expectancy'].iloc[i], df_high_extreme_poverty['Country'].iloc[i], fontsize=7, ha='right')

plt.xticks(range(5, 70, 5))
plt.yticks(range(50, 75, 2))

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with share in extreme poverty more than 5%) for 2021')
plt.show()

# List countries with high life expectancy but higher share in extreme poverty using medians
median_life_expectancy = np.median(df_extreme_poverty_life_expectancy['Life expectancy'])
print('Median life expectancy is', median_life_expectancy)
median_extreme_poverty = np.median(df_extreme_poverty_life_expectancy['Share in extreme poverty'])
print('Median share in extreme poverty is', median_extreme_poverty)
df_higher_life_expectancy_higher_extreme_poverty_medians = df_extreme_poverty_life_expectancy[(df_extreme_poverty_life_expectancy['Life expectancy'] > median_life_expectancy) & (df_extreme_poverty_life_expectancy['Share in extreme poverty'] > median_extreme_poverty)]
print('The number of countries with high life expectancy but higher share in extreme poverty using medians is', len(df_higher_life_expectancy_higher_extreme_poverty_medians))
print('The list of countries with high life expectancy but higher share in extreme poverty using medians is:', ', '.join(df_higher_life_expectancy_higher_extreme_poverty_medians['Country']))

plt.figure(figsize=(22, 11))
plt.scatter(df_higher_life_expectancy_higher_extreme_poverty_medians['Share in extreme poverty'], df_higher_life_expectancy_higher_extreme_poverty_medians['Life expectancy'], color='blue', s = df_higher_life_expectancy_higher_extreme_poverty_medians['Population']/200000, alpha=0.5)

for i, country in enumerate(df_higher_life_expectancy_higher_extreme_poverty_medians['Country']):
    plt.text(df_higher_life_expectancy_higher_extreme_poverty_medians['Share in extreme poverty'].iloc[i], df_higher_life_expectancy_higher_extreme_poverty_medians['Life expectancy'].iloc[i], df_higher_life_expectancy_higher_extreme_poverty_medians['Country'].iloc[i], fontsize=7, ha='right')

plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])
plt.yticks(range(73, 85))

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with high life expectancy but higher share in extreme poverty using medians as criteria) for 2021')
plt.show()

# List countries with high life expectancy but higher share in extreme poverty using means
mean_life_expectancy = np.mean(df_extreme_poverty_life_expectancy['Life expectancy'])
print('Mean life expectancy is', mean_life_expectancy)
mean_extreme_poverty = np.mean(df_extreme_poverty_life_expectancy['Share in extreme poverty'])
print('Mean share in extreme poverty is', mean_extreme_poverty)
df_higher_life_expectancy_higher_extreme_poverty_means = df_extreme_poverty_life_expectancy[(df_extreme_poverty_life_expectancy['Life expectancy'] > mean_life_expectancy) & (df_extreme_poverty_life_expectancy['Share in extreme poverty'] > mean_extreme_poverty)]
print('The number of countries with high life expectancy but higher share in extreme poverty using means is', len(df_higher_life_expectancy_higher_extreme_poverty_means))

upper_boundary_low_extreme_poverty = df_extreme_poverty_life_expectancy['Share in extreme poverty'].quantile(0.25)
print('The upper boundary for low share in extreme poverty is', upper_boundary_low_extreme_poverty)
df_lower_extreme_poverty = df_extreme_poverty_life_expectancy[df_extreme_poverty_life_expectancy['Share in extreme poverty'] <= upper_boundary_low_extreme_poverty]
print('The number of countries with lower index of extreme poverty using 25th percentile is', len(df_lower_extreme_poverty))
print('The list of countries with lower index of extreme poverty using 25th percentile is:', ', '.join(df_lower_extreme_poverty['Country']))

plt.figure(figsize=(22, 11))
plt.scatter(df_lower_extreme_poverty['Share in extreme poverty'], df_lower_extreme_poverty['Life expectancy'], color='blue', s = df_lower_extreme_poverty['Population']/200000, alpha=0.5)

for i, country in enumerate(df_lower_extreme_poverty['Country']):
    plt.text(df_lower_extreme_poverty['Share in extreme poverty'].iloc[i], df_lower_extreme_poverty['Life expectancy'].iloc[i], df_lower_extreme_poverty['Country'].iloc[i], fontsize=7, ha='right')

plt.xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
plt.yticks(range(68, 86, 2))

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with lower index of extreme poverty using 25th percentile as criteria) for 2021')
plt.xlim(left=-0.02)
plt.show()

lower_boundary_high_life_expectancy = np.quantile(df_extreme_poverty_life_expectancy['Life expectancy'], 0.75)
print('75th Percentile life expectancy is', lower_boundary_high_life_expectancy)
print('Mean life expectancy is', mean_life_expectancy)
upper_boundary_low_life_expectancy = np.quantile(df_extreme_poverty_life_expectancy['Life expectancy'], 0.25)
print('25th Percentile life expectancy is', upper_boundary_low_life_expectancy)

plt.figure(figsize=(22, 11))
plt.scatter(df_lower_extreme_poverty['Share in extreme poverty'], df_lower_extreme_poverty['Life expectancy'], color='blue', s = df_lower_extreme_poverty['Population']/200000, alpha=0.5)

for i, country in enumerate(df_lower_extreme_poverty['Country']):
    plt.text(df_lower_extreme_poverty['Share in extreme poverty'].iloc[i], df_lower_extreme_poverty['Life expectancy'].iloc[i], df_lower_extreme_poverty['Country'].iloc[i], fontsize=7, ha='right')

plt.axhline(y=lower_boundary_high_life_expectancy, color='green', linestyle='--', label='75th Percentile Life Expectancy')

plt.axhline(y=mean_life_expectancy, color='red', linestyle='--', label='Mean Life Expectancy')

plt.axhline(y=upper_boundary_low_life_expectancy, color='blue', linestyle='--', label='25th Percentile Life Expectancy')

plt.xlabel('Share in Extreme Poverty (percents)')
plt.ylabel('Life Expectancy (years)')
plt.title('Share in Extreme Poverty vs Life Expectancy (countries with lower index of extreme poverty using 25th percentile as criteria) for 2021')
plt.legend()
plt.xlim(left=-0.02)
plt.show()

df_life_satisfaction.summary().show()

df_political_corruption.summary().show()

# Merge the above two data frames using inner join on 'Çountry' column
df_life_satisfaction_political_corruption = df_life_satisfaction.join(df_political_corruption, df_life_satisfaction.Country == df_political_corruption.Country, "inner")

df_life_satisfaction_political_corruption.show()

# Convert PySpark DataFrame to pandas DataFrame
df_life_satisfaction_political_corruption = df_life_satisfaction_political_corruption.toPandas()

df_life_satisfaction_political_corruption = df_life_satisfaction_political_corruption.loc[:, ~df_life_satisfaction_political_corruption.columns.duplicated()]

df_life_satisfaction_political_corruption.head()

df_life_satisfaction_political_corruption.plot.scatter(x='Political corruption index', y='Cantril ladder score')
plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst; 10=best)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (world countries) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
plt.scatter(df_life_satisfaction_political_corruption['Political corruption index'], df_life_satisfaction_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_life_satisfaction_political_corruption['Country']):
    plt.text(df_life_satisfaction_political_corruption['Political corruption index'].iloc[i], df_life_satisfaction_political_corruption['Cantril ladder score'].iloc[i], df_life_satisfaction_political_corruption['Country'].iloc[i], fontsize=6, ha='right')

plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.yticks(range(0, 9))

plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (world countries) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
df_low_political_corruption = df_life_satisfaction_political_corruption[df_life_satisfaction_political_corruption['Political corruption index'] <= 0.3]
plt.scatter(df_low_political_corruption['Political corruption index'], df_low_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_low_political_corruption['Country']):
    plt.text(df_low_political_corruption['Political corruption index'].iloc[i], df_low_political_corruption['Cantril ladder score'].iloc[i], df_low_political_corruption['Country'].iloc[i], fontsize=7, ha='right')

plt.yticks([3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])

plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (countries with political corruption index less than 0.3) for 2021')
plt.show()

plt.figure(figsize=(22, 11))
df_high_political_corruption = df_life_satisfaction_political_corruption[df_life_satisfaction_political_corruption['Political corruption index'] >= 0.3]
plt.scatter(df_high_political_corruption['Political corruption index'], df_high_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_high_political_corruption['Country']):
    plt.text(df_high_political_corruption['Political corruption index'].iloc[i], df_high_political_corruption['Cantril ladder score'].iloc[i], df_high_political_corruption['Country'].iloc[i], fontsize=7, ha='right')

plt.yticks([2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])

plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (countries with political corruption index higher than 0.3) for 2021')
plt.show()

top_10_happiest_countries = df_life_satisfaction_political_corruption.sort_values(by='Cantril ladder score', ascending=False).head(10)

plt.figure(figsize=(9, 7))
plt.barh(top_10_happiest_countries['Country'], top_10_happiest_countries['Cantril ladder score'], color='lightgreen')
plt.xlabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.ylabel('Country')
plt.title('Top 10 Happiest Countries (Based on Cantril Ladder Score) for 2021')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

top_20_most_politically_corrupted_countries = df_life_satisfaction_political_corruption.sort_values(by='Political corruption index', ascending=False).head(20)

plt.figure(figsize=(9, 7))
plt.barh(top_20_most_politically_corrupted_countries['Country'], top_20_most_politically_corrupted_countries['Political corruption index'], color='crimson')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Country')
plt.title('Top 20 Most Politically Corrupted Countries for 2021')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

upper_boundary_low_political_corruption = np.quantile(df_life_satisfaction_political_corruption['Political corruption index'], 0.25)
print('25th Percentile political corruption index is', upper_boundary_low_political_corruption)

df_lower_political_corruption = df_life_satisfaction_political_corruption[
    df_life_satisfaction_political_corruption['Political corruption index'] <= upper_boundary_low_political_corruption]
print('The number of countries with lower political corruption index using 25th percentile is', len(df_lower_political_corruption))
print('The list of countries with lower political corruption index using 25th percentile is:', ', '.join(df_lower_political_corruption['Country']))

plt.figure(figsize=(18, 8))
plt.scatter(df_lower_political_corruption['Political corruption index'], df_lower_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_lower_political_corruption['Country']):
    plt.text(df_lower_political_corruption['Political corruption index'].iloc[i], df_lower_political_corruption['Cantril ladder score'].iloc[i], df_lower_political_corruption['Country'].iloc[i], fontsize=6, ha='right')

plt.yticks([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (countries with lower political corruption index using 25th percentile) for 2021')
plt.show()

lower_boundary_high_life_satisfaction = np.quantile(df_life_satisfaction_political_corruption['Cantril ladder score'], 0.75)
print('75th Percentile Cantril ladder score is', lower_boundary_high_life_satisfaction)
mean_life_satisfaction = np.mean(df_life_satisfaction_political_corruption['Cantril ladder score'])
print('Mean Cantril ladder score is', mean_life_satisfaction)
upper_boundary_low_life_satisfaction = np.quantile(df_life_satisfaction_political_corruption['Cantril ladder score'], 0.25)
print('25th Percentile Cantril ladder score is', upper_boundary_low_life_satisfaction)

plt.figure(figsize=(22, 11))
plt.scatter(df_lower_political_corruption['Political corruption index'], df_lower_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_lower_political_corruption['Country']):
    plt.text(df_lower_political_corruption['Political corruption index'].iloc[i], df_lower_political_corruption['Cantril ladder score'].iloc[i], df_lower_political_corruption['Country'].iloc[i], fontsize=7, ha='right')

plt.yticks([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
plt.axhline(y=lower_boundary_high_life_satisfaction, color='green', linestyle='--', label='75th Percentile Cantril ladder score')
plt.axhline(y=mean_life_satisfaction, color='red', linestyle='--', label='Mean Cantril ladder score')
plt.axhline(y=upper_boundary_low_life_satisfaction, color='blue', linestyle='--', label='25th Percentile Cantril ladder score')

plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (countries with lower political corruption index using 25th percentile as criteria) for 2021')
plt.legend()
plt.show()

lower_boundary_high_political_corruption = np.quantile(df_life_satisfaction_political_corruption['Political corruption index'], 0.75)
print('75th Percentile political corruption index is', lower_boundary_high_political_corruption)

df_higher_political_corruption = df_life_satisfaction_political_corruption[
    df_life_satisfaction_political_corruption['Political corruption index'] >= lower_boundary_high_political_corruption]
print('The number of countries with higher political corruption index using 75th percentile is', len(df_higher_political_corruption))
print('The list of countries with higher political corruption index using 75th percentile is:', ', '.join(df_higher_political_corruption['Country']))

df_higher_political_corruption_higher_life_satisfaction = df_higher_political_corruption[df_higher_political_corruption['Cantril ladder score'] > mean_life_satisfaction]
df_higher_political_corruption_higher_life_satisfaction.head(9)

plt.figure(figsize=(22, 11))
plt.scatter(df_higher_political_corruption['Political corruption index'], df_higher_political_corruption['Cantril ladder score'], color='green', alpha=0.6)

for i, country in enumerate(df_higher_political_corruption['Country']):
    plt.text(df_higher_political_corruption['Political corruption index'].iloc[i], df_higher_political_corruption['Cantril ladder score'].iloc[i], df_higher_political_corruption['Country'].iloc[i], fontsize=7, ha='right')

plt.axhline(y=lower_boundary_high_life_satisfaction, color='green', linestyle='--', label='75th Percentile Cantril ladder score')
plt.axhline(y=mean_life_satisfaction, color='red', linestyle='--', label='Mean Cantril ladder score')
plt.axhline(y=upper_boundary_low_life_satisfaction, color='blue', linestyle='--', label='25th Percentile Cantril ladder score')

plt.xlabel('Political corruption index (0=low; 1=high)')
plt.ylabel('Cantril ladder score (0=worst possible life; 10=best possible life)')
plt.title('Political Corruption Index vs Self-Reported Life Satisfaction (Cantril Ladder Score) (countries with higher political corruption index using 75th percentile) for 2021')
plt.legend(loc="lower right")
plt.show()

df_political_corruption_continents.show()
df_life_satisfaction_continents.show()

df_political_corruption_continents = df_political_corruption_continents.toPandas()
df_life_satisfaction_continents = df_life_satisfaction_continents.toPandas()

x = np.arange(len(continents))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, df_life_satisfaction_continents['Cantril ladder score'], width, label='Cantril ladder score (0=worst possible life; 10=best possible life)', color='lightgreen')
bars2 = ax.bar(x + width/2, df_political_corruption_continents['Political corruption index'] * 10, width, label='Political corruption index (rescaled 0=low; 10=high)', color='crimson')

ax.set_xlabel('Populated Continents')
ax.set_ylabel('Index Scores (0-10)')
ax.set_title('Life Satisfaction (Cantril Ladder Score) and Political Corruption Index by Continents for 2021')
ax.set_xticks(x)
ax.set_xticklabels(continents)
ax.legend()
plt.tight_layout()
plt.show()