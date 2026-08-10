import seaborn as sns

from sklearn.preprocessing import StandardScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

df = sns.load_dataset('titanic')

print(df.head(50))

print(df.info())

print(df.describe())

print(df.shape)


percentage_null_values = (df.isnull().sum()/df.shape[0])*100

print(percentage_null_values)

folder = os.path.dirname(os.path.abspath(__file__))

t_data_set = os.path.join(folder, "titanic.csv")

df.to_csv(t_data_set, index=False)

df_new = pd.read_csv(t_data_set)

df_new = df.drop_duplicates()

print(df_new.head(50))

print(df_new.info())

print(df_new.describe())

print(df_new.shape)

# dropping the emabarked NA rows as it has less than 1 % of data

df_new = df_new.dropna(subset=["embarked","embark_town"])

# imputing the age rows with mean keeps the overall column average unchnaged and also as it has more than 5% of data

df_new["age"] = df_new["age"].fillna(df_new["age"].mean())

# droping the desk column , imputing with the mode will bring bias

df_new = df_new.drop(columns="deck")

#dropping these colums as it share duplication
# who and adult_male is same as sex so we kept sex
# class and pclass share same so kept pclass
# embarked and embark_town share same data with emabark_town have good info so kept the same

df_new = df_new.drop(columns=["who","adult_male","class","alive","embarked"])

print(df_new)


t_data_set2 = os.path.join(folder, "titanic_cleaned.csv")

df_new.to_csv(t_data_set2, index=False)



# Generate two normal distributions
dist1 =  df_new["age"]
# dist2 = range(df_new["age"])

plt.hist(dist1, bins=20)

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('range')

plt.show()



plt.boxplot(dist1)

plt.xlabel('Age')

plt.title('Age Outliners')

plt.show()


age_q1 = np.percentile(dist1, 25)
age_q3 = np.percentile(dist1,75)
age_IQR = age_q3-age_q1
age_lower = age_q1 - (1.5* age_IQR)
age_higher = age_q3 + (1.5* age_IQR)



age_outliners = dist1[(dist1 < age_lower) | (dist1 > age_higher)]

num_age_outliners = len(age_outliners)

print(f"number of age outliner are {num_age_outliners}")

Mean_age = dist1.mean()
Median_age = dist1.median()
Mode_age = dist1.mode()[0]

print(f"{Mean_age,Median_age,Mode_age}")

if Mean_age > Median_age > Mode_age:
    print("Age distribution is right-skewed")
elif Mean_age < Median_age < Mode_age:
    print("Age distribution is left-skewed")
else:
    print("Age distribution is approximately symmetric")


## The Age distribution is symemetric because the mean , median and mode are almost equal.

# Generate two normal distributions
dist2 =  df_new["fare"]
# dist2 = range(df_new["age"])

plt.hist(dist2, bins=200)

plt.title('Fare Distribution')

plt.xlabel('Fare')

plt.ylabel('range')

plt.show()


plt.boxplot(dist2)

plt.xlabel('Fare')

plt.title('Fare Outliners')

plt.show()


fare_q1 = np.percentile(dist2, 25)
fare_q3 = np.percentile(dist2,75)
fare_IQR = fare_q3-fare_q1
fare_lower = fare_q1 - (1.5* fare_IQR)
fare_higher = fare_q3 + (1.5* fare_IQR)


fare_outliners = dist2[(dist2 < fare_lower)  | (dist2 > fare_higher)]

num_fare_outliners = len(fare_outliners)

print(f"number of fare outliner are {num_fare_outliners}")

Mean_fare = dist2.mean()
Median_fare = dist2.median()
Mode_fare = dist2.mode()[0]

print(Mean_fare,Median_fare,Mode_fare)

if Mean_fare > Median_fare > Mode_fare:
    print("Fare distribution is right-skewed")
elif Mean_fare < Median_fare < Mode_fare:
    print("Fare distribution is left-skewed")
else:
    print("Fare distribution is approximately symmetric")

##The Fare distribution is right-skewed because the mean is greater than the median which is greater than the mode.

male = df_new[df_new["sex"] == "male"]
female = df_new[df_new["sex"] == "female"]

female_survival = female["survived"].mean()
male_survival = male["survived"].mean()

print(male_survival, female_survival)


f_class = df_new[df_new["pclass"] == 1]
s_class = df_new[df_new["pclass"] == 2]
t_class = df_new[df_new["pclass"] == 3]

f_class_sur = f_class["survived"].mean()
s_class_sur = s_class["survived"].mean()
t_class_sur = t_class["survived"].mean()

print(f_class_sur,s_class_sur,t_class_sur)


f_class_male = df_new[(df_new["pclass"] == 1) & (df_new["sex"] == "male")]
f_class_female = df_new[(df_new["pclass"] == 1) & (df_new["sex"] == "female")]
s_class_male = df_new[(df_new["pclass"] == 2) & (df_new["sex"] == "male")]
s_class_female = df_new[(df_new["pclass"] == 2) & (df_new["sex"] == "female")]
t_class_male = df_new[(df_new["pclass"] == 3) & (df_new["sex"] == "male")]
t_class_female = df_new[(df_new["pclass"] == 3) & (df_new["sex"] == "female")]

f_class_male_sur = f_class_male["survived"].mean()
f_class_female_sur = f_class_female["survived"].mean()
s_class_male_sur = s_class_male["survived"].mean()
s_class_female_sur = s_class_female["survived"].mean()
t_class_male_sur = t_class_male["survived"].mean()
t_class_female_sur = t_class_female["survived"].mean()

print(f_class_male_sur,f_class_female_sur,s_class_male_sur,s_class_female_sur,t_class_male_sur,t_class_female_sur)

df_cor = df_new.drop(columns=["sex","embark_town","alone"])

correlation = df_cor.corr()

print(correlation)


sns.heatmap(correlation, annot=True)
plt.title("Correlation of the 6 columns")
plt.show()



absl = correlation.abs()

print(absl)

top = absl.unstack().sort_values(ascending=False)

top_two = top.iloc[6:10]

print(top_two)

# The two strongest correlations are between pclass and fare is - 0.55 and between sibsp and parch + 0.41. 
# The negative correlation between pclass and fare indicates that higher class numbers are generally associated with lower fares.
# The positive correlation between sibsp and parch indicates that passengers travelling with siblings or spouses were also more likely to be travelling with parents or children.


plt.bar(["Male", "Female"], [male_survival,female_survival])

plt.title("Survival of male vs female")

plt.xlabel("male and female")

plt.ylabel("Survival Rate")

plt.show()

## Women had a higher survival rate than men. 
## This suggests that sex was an important factor associated with survival.

plt.bar(["1st class", "2nd class",  "3rd class"], [f_class_sur,s_class_sur, t_class_sur])

plt.title("Survival of 1st , 2nd and 3rd class")

plt.xlabel("1st , 2nd and 3rd class")

plt.ylabel("Survival Rate")

plt.show()

## Passengers in first class had a higher survival rate than passengers in second and third class.
## Socioeconomic state played important role.

plt.bar(["1st class male", "1st class female", "2nd class male","2nd class female", "3rd class male" , "3rd class female"], [f_class_male_sur , f_class_female_sur,s_class_male_sur,s_class_female_sur, t_class_male_sur,t_class_female_sur])

plt.title("Survival of 1st , 2nd and 3rd class male and females")

plt.ylabel("Survival Rate")

plt.show()

#Female passengers generally had higher survival rates than male passengers within the same passenger class.
#The combination of sex and passenger class therefore provides a clearer picture of survival differences than either variable alone.


sur = df_new[df_new["survived"] == 1]
nonsur = df_new[df_new["survived"] == 0]

sur_age = sur["age"]
nonsur_age = nonsur["age"]

plt.boxplot([sur_age , nonsur_age])

plt.xticks([1, 2], ["Survived", "Not Survived"])

plt.xlabel('Fare')

plt.title('Fare Outliners')

plt.show()

#The ages of survivors and non-survivors overlap, but their median ages are different.
#This shows that age may have affected survival, but sex and passenger class had a stronger effect.

sep_age = df_new[["age"]]
sep_fare = df_new[["fare"]]


stdsc = StandardScaler()

new_age = stdsc.fit_transform(sep_age)

new_fare = stdsc.fit_transform(sep_fare)

sep_age_mean = sep_age["age"].mean()
sep_fare_mean = sep_fare["fare"].mean()
sep_age_std = sep_age["age"].std()
sep_fare_std = sep_fare["fare"].std()
new_age_mean = new_age.mean()
new_fare_mean = new_fare.mean()
new_age_std = new_age.std()
new_fare_std = new_fare.std()

print(f"{'Feature':<10} {'Before Mean':>15} {'Before Std':>15} {'After Mean':>15} {'After Std':>15}")

print(f"{'Age':<10} {sep_age_mean:>15.2f} {sep_age_std:>15.2f} "
      f"{new_age_mean:>15.2f} {new_age_std:>15.2f}")

print(f"{'Fare':<10} {sep_fare_mean:>15.2f} {sep_fare_std:>15.2f} "
      f"{new_fare_mean:>15.2f} {new_fare_std:>15.2f}")


