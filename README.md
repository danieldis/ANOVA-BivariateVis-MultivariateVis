# ANOVA, Bivariate Vis & Multivariate Vis
The goal is to get one step closer to predictive analysis.

Before I can start predicting data I need to be able to visualize and identify the most likely candidates in the data 
that will be of the most predictive power. In other words, which columns (or subset) of data will help me predict the answer?

### Data for this assignment:

I'm using the Titanic data set: https://www.kaggle.com/c/titanic/data (Links to an external site.)

The file I will be using is the "train.csv" file: [train.csv](</downloads/train.csv "Title">).

The idea behind the data is that "train.csv" file helps me know who did and who did not survive. The "test.csv" file is used to 
actually test my model. However, I'm not quite ready for the actual prediction.

I simply asked: What is/are the data with the most predictive power?

I will do EDA (Exploratory Data Analysis) to figure that out.


### ANOVA's:

* I performed an ANOVA on the 'Sex' column using 'Survived' as the independent variable.
* Performed a similiar ANOVA on PClass using 'Survived' as the independent variable.

### Separate 'Sex' Column:

* I separated the data based on the sex of the passenger.
* I visualized the correlation of 'female' to survived with the corresponding 
linear regression.
* I Visualized the correlation of 'male' to survived with the corresponding linear 
regression.
* I picked two different columns to determine if they are normal distributions. If they are not normal then I will transform it and revisualize it. 

### Bivariate Visualizations:

* I created 3 different bivariate visualizations between the 'Survived' column and another column.
* I separated the columns and visualized the subsets of the data.

### Multivariate Visualization:

* I used a multivariate visualization to see the interaction between multiple variables. 



