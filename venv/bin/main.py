import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


path_name = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"

df = pd.read_csv(path_name)
print(df.head())

# First, let's see what data types we are dealing with
print(df.dtypes)

# Looks correctly typed. No wrangling needed for types.

#I'm curious about the correlation between bore, stroke, compression ratio, and horsepower. Let's check that
df_corr = df[["bore", "stroke", "compression-ratio", "horsepower"]]

print(df_corr.corr())

# Correlation between bore and price

print(df[["bore", "price"]].corr())

# Positive but seems pretty weak. Lets check with a regplot
sns.regplot(x="bore", y="price", data = df)
plt.ylim(0,)
plt.show()

# Yeah looks pretty weak
# Let's look at some categorical variables now

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()

# Looks like this could possibly be a predictor of price.
# Interesting to note that correlation does not incidcate causation here
# Since RWD vehicles are actually the worst for handling and usually a lower package
# than 4-wheel drive. However, RWD vehicles tend to be sporty, resulting in a higher
# price. 4WD is still of more value than RWD, but RWD does still predict a higher cost
# in a general sample set.

# Lets look at some value counts

counts = df["drive-wheels"].value_counts().to_frame()
counts.rename(columns= {"drive-wheels" : "value_counts" },inplace=True)
counts.index.name = "drive-wheels"
print(counts)

# Based on these counts, drive-wheels might not be accurate. n=8 for 4wd cars, which could
# explain the counterintuitive impliciation above.

# Nonetheless, lets run some more tests

df_group_one = df[["drive-wheels", "body-style", "price"]]
df_test_one = df_group_one.groupby(["drive-wheels", "body-style"], as_index = False).mean()
print(df_test_one)
df_test_one_pivot = df_test_one.pivot(index="drive-wheels", columns="body-style")
print(df_test_one_pivot)

# Nice. Not super helpful though.

# Maybe body-style alone is better to visualize

df_group_two = df[["body-style", "price"]].groupby(["body-style"], as_index = False).mean()
print(df_group_two)

# Lets heatmap with the drive-wheels and body style. Best solution.
plt.pcolor(df_test_one_pivot, cmap="RdBu")
plt.colorbar()
plt.show()
# That sucks - no labels?

figure, axis = plt.subplots()
image = axis.pcolor(df_test_one_pivot, cmap="RdBu")

# Add labels
axis.set_xticklabels(df_test_one_pivot.columns.levels[1], minor=False)
axis.set_yticklabels(df_test_one_pivot.index, minor=False)

#move ticks and labels to the center
axis.set_xticks(np.arange(df_test_one_pivot.shape[1]) + 0.5, minor=False)
axis.set_yticks(np.arange(df_test_one_pivot.shape[0]) + 0.5, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

figure.colorbar(image)
plt.show()

# Cool!

# Lets look at Pearson Correlation for the numerical vars

# statistically significant, weak linear correlation
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# statistically significant, strong positive correlation
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
#...
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value )

# ANOVA

# Question: does the different types of drive wheels impact price

grouped_test_two = df[["drive-wheels", "price"]].groupby("drive-wheels")
print(grouped_test_two.head())

#ANOVA TEST
f_val, p_val = stats.f_oneway(grouped_test_two.get_group("fwd")["price"], grouped_test_two.get_group("rwd")["price"], grouped_test_two.get_group("4wd")["price"])

print(f_val, p_val)

# This looks like a very strong correlation! p<<.0001 and F score is large.
# But are all groups correlated? We need to look more closely

f_val, p_val = stats.f_oneway(grouped_test_two.get_group('fwd')['price'], grouped_test_two.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test_two.get_group('4wd')['price'], grouped_test_two.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped_test_two.get_group('fwd')['price'], grouped_test_two.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)







