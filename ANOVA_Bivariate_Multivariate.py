#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 07:23:58 2018

@author: danielsalmond
"""
# %%

import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
#import seaborn as sns; sns.set(color_codes=True)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


file = "./A4/train.csv"

# Creates a DataFrame from the csv file
train_data2 = pd.read_csv(file, converters={'Sex':lambda s: ['male','female'].index(s)})
train_data1 = pd.read_csv(file, converters={'Sex':lambda s: ['female','male'].index(s)})
train_data = pd.read_csv(file)




###########################################################################################################
#                                   ANOVA's
###########################################################################################################

print(train_data2)
#    Performed an ANOVA on the 'Sex' column using 'Survived' as the independent variable. 


#for the anova we need to create our one-way model:
model = ols('Sex ~ Survived', data=train_data2).fit()

#actually run the anova:
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA 1 results:")
print(aov_table)

#So, we will compare the three groups against each other:
print("\nTukey HSD results:")
mc = MultiComparison(train_data2['Sex'], train_data2['Survived'])
result = mc.tukeyhsd()

print(result)

print("There is a correlation to sex to survival. The female's had a higher chance of surviving.")


#    Perform a similiar ANOVA on PClass using 'Survived' as the independent variable. 

#for the anova we need to create our one-way model:
model = ols('Pclass ~ Survived', data=train_data2).fit()

#actually run the anova:
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA 2 results:")
print(aov_table)

#So, we will compare the three groups against each other:
print("\nTukey HSD results:")
mc = MultiComparison(train_data['Pclass'], train_data2['Survived'])
result = mc.tukeyhsd()

print(result)

print("There is a correlation to higher class to survival, the upper class survived.")

# %%


############################################################################################################
##                           Seperate 'Sex' Column
############################################################################################################

#    Seperating all the data pertaining to female, and another pertaining to male.
df3, df4 = [x for _, x in train_data2.groupby(train_data2['Sex'] == 0)]

df3.columns  = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Female', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df4.columns  = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Male', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


#    Visualizing with linear regression to find the correlation of 'female' to survived.
print("\nCorrelation table of survived, female, and male.\n")
femaleToSurvived = train_data[['Sex', 'Survived']]
femaleToSurvived = femaleToSurvived.loc[femaleToSurvived['Sex'].isin(('female', 'male'))]
dummies = pd.get_dummies(femaleToSurvived['Sex'])
femaleToSurvived = pd.concat([femaleToSurvived, dummies], axis=1)
femaleToSurvived = femaleToSurvived.drop(['Sex'], axis=1)
print(femaleToSurvived.corr())


x = np.array(df3['Female'])
y = np.array(df3['Survived'])
fitme = np.polyfit(x,y,1)
fitme_fn = np.poly1d(fitme)

plt.plot(x,y, 'yo', x, fitme_fn(x), '--k')
plt.title("Correlation between the Female and Survived")
plt.xlabel("Female")
plt.ylabel("Survived")
plt.show()


#    Visualizing with linear regression to find the correlation of 'male' to survived.

fig, ax = plt.subplots()
fit = np.polyfit(train_data2['Sex'], train_data2['Survived'], deg=1)
#ax.plt(train_data2['Sex'], fit[0]*train_data2['Sex'] + fit[1], color="red")
plt.scatter(train_data2['Sex'], train_data2['Survived'])
plt.xlabel("Male")
plt.ylabel("Survived (0 is dead, 1 is alive)")
plt.title("Linear Correlation of Males")
plt.show()


x = np.array(femaleToSurvived['female'])
y = np.array(femaleToSurvived['Survived'])
fitme = np.polyfit(x,y,1)
fitme_fn = np.poly1d(fitme)

plt.plot(x,y, 'yo', x, fitme_fn(x), '--k')
plt.title("Correlation between Female/Male and Survived")
plt.xlabel("Male (0.0), Female (1.0)")
plt.ylabel("Survived (0 is dead, 1 is alive)")
print("")
plt.show()

femaleToSurvived1 = train_data2[['Sex', 'Survived']]
sex = femaleToSurvived1['Sex']
survived = femaleToSurvived1['Survived']
correlation1 = sex.corr(survived)
print("\nCorrelation of 'female' to survived: ",correlation1)


#   Seeing if the 'Age' and 'Fare' columns are normal distributions.

age = train_data.copy(True)
age['Age'] = age['Age'].fillna(200)
sns.distplot(age['Age'])
plt.show()

fare = train_data2[(train_data2['Fare'] > 0.0)]
data_fare = pd.DataFrame(fare)
fareVariable = np.log(data_fare['Fare'])
sns.distplot(fareVariable)


# %%


###########################################################################################################
#                            Bivariate Visualizations
###########################################################################################################

#    Create a bivariate visualization (e.g. scatterplot) between the 'Survived' column and another.

train_data.plot(kind="scatter",     # Create a scatterplot
              x="Survived",          # Put Survived on the x axis
              y="Age",          # Put Age on the y axis
              figsize=(7,7),
              ylim=(0,100)) 

train_data.plot(kind="scatter",     # Create a scatterplot
              x="Survived",          # Put Survived on the x axis
              y="Fare",          # Put Age on the y axis
              figsize=(7,7),
              ylim=(0,600))


train_data.plot(kind="scatter",     # Create a scatterplot
              x="Survived",          # Put Survived on the x axis
              y="Parch",        # Put Age on the y axis
              figsize=(7,7),
              ylim=(0,10))

# %%

###########################################################################################################
#                           Multivariate Visualizations
###########################################################################################################

g = sns.pairplot(train_data2, diag_kind="kde")
print(g)

# %%

###########################################################################################################
#                                  Small Report
###########################################################################################################

#   In the end, write a small report (hardcode it in your python code) where you indicate the most 
#   important columns. Explain why you think they are the most important columns (or subset of columns).

print("\n\nSmall Report: The most important columns in my mind would be the Sex, Survived, and Pclass. With those columns you can determine who survived and why. You find out that the rich, 1st class people had a higher chance of survival. The female 1st class people had a high survival rate. So basically, if you where rich and female you survived the sinking of the titanic. The sex column, I seperated it into female and male so I could analyse the data better. The Survived column will tell you who survived and not survived, which would be an important column because the Titanic sank.")




