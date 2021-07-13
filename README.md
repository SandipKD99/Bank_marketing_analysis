## Predict if a customer if going to make term deposit or not

## Description about the data set.


 
The dataset we are using has total 16 columns and 11162 rows.

The columns of the dataset are:-

1 - age: (numeric)


2 - job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')


3 - marital: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)


4 - education: (categorical: primary, secondary, tertiary and unknown)


5 - default: has credit in default? (categorical: 'no','yes','unknown')


6 - housing: has housing loan? (categorical: 'no','yes','unknown')


7 - loan: has personal loan? (categorical: 'no','yes','unknown')


8 - contact: contact communication type (categorical: 'cellular','telephone')


9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')


10 - day: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')


11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no')


12 - balance: Balance of the individual.

13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)


14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)


15 - previous: number of contacts performed before this campaign and for this client (numeric)


16 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')


17 - Output variable (desired target):
 - y - has the client subscribed a term deposit? (binary: 'yes','no')

!pip install plotly

# importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier

import pickle

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
%matplotlib inline



ds = pd.read_csv('bank.csv') #reading the csv file

ds.head() #checking the top 5 records

ds.info() # getting more idea aboout data

ds.describe() # checking some statistical information for our data

 Now Let's see how many sub categories are present in each categorical column

# creating a list which only contains categorical data

cat = [i for i in ds.columns if ds[i].dtype=='O'] 

# method for printing the sub-categoires present

def get_cat(cat):
    for i in cat:
        a = ds[i].unique()
        print('unique values in'+' '+i+' '+'are:-',a)

get_cat(cat) # calling the method

From the above output we can observe that there are many sub categories or unique values in each categorical caloumn.

## Now lets check how many null values are present on our data

ds.isnull().sum() # checking the total number of nan values

## Data Visualization

In this part we will plot some simple graphs so that we understand the data in a much better way.

ds.head() 

As there are two classes in the target varibale so let's check if the proportion of each class is almost equal or not.

fig = px.pie(ds, names='deposit',width=600, height=400 )
fig.show()

The above plot says that the proportion of the deposits which is 'Yes' is almost equal to class 'No', so the classes are balanced.

Now we will check the data distribution of the numerical columns using histogram.

#sns.histplot(ds['age']) # ploting histogram

fig = px.histogram(ds, x="age",width=800, height=400)
fig.show()

From the above plot we can see that the age column is rightly skewed which means there is maximium amout of data present between the age 20-60 and above that the density decreases.

#sns.histplot(ds['balance'], kde = True)
fig = px.histogram(ds, x="balance",width=800, height=400, nbins = 80)
fig.show()

Same case is for the column 'balance' where is rightly skewed and along with some outliers.

fig = px.histogram(ds, x="duration",width=800, height=400, nbins = 80)
fig.show()

The above distribution also has some outliers as most of the data lies between the range 0 to around 500 but there are extreme outliers present at 4000.

Lets check the relation between the capital gain and cpatial loss using scatter plot.

# ploting scatter plot
fig = px.scatter(ds, x="age",y = 'balance', width=800, height=400)
fig.show()
#sns.scatterplot(data=ds, x="age", y="balance") 

The above scatter plot says that there is no significant relation between age and balance as many people between the age 20 and 90 have almost the same balance that is less than 20,000 onyly few of them are having the balance more than or equal to 20,000.

#sns.scatterplot(data=ds, x="age", y="duration") 
fig = px.scatter(ds, x="age",y = 'duration', width=800, height=400)
fig.show()

We can observe from the above distribuiton is that people who are aged between 20-60 have received calls whos duration was less than or equal to 2 minutes. Ony some of them have calls whos duration more than 2 minutes.

fig = px.scatter_3d(ds, x='age', y='balance', z='duration',color = 'deposit')
fig.show()

visualizing the same plots in 3d.

##### visualization of  the categorical variables

ds.head()

sns.countplot(x="job", data=ds, palette="Set3")
plt.xticks(rotation=90)

plt.figure(figsize=(50,50))
a = sns.catplot(x="job", col="deposit",
                data=ds, kind="count")
#plt.xticks(rotation=90)
a.set_xticklabels(rotation=90)

plt.figure(figsize=(50,50))
a = sns.catplot(x="education", col="deposit",
                data=ds, kind="count")
#plt.xticks(rotation=90)
a.set_xticklabels(rotation=90)

plt.figure(figsize=(50,50))
a = sns.catplot(x="job", col="loan",
                data=ds, kind="count")
#plt.xticks(rotation=90)
a.set_xticklabels(rotation=90)

sns.countplot(x="month", hue="deposit", data=ds, palette="Set2")
plt.xticks(rotation=90)

In May we got maximum transaction

plt.figure(figsize=(50,50))
a = sns.catplot(x="contact", col="deposit",
                data=ds, kind="count")
#plt.xticks(rotation=90)
a.set_xticklabels(rotation=90)

sns.countplot(x="poutcome", data=ds, palette="Set1")
plt.xticks(rotation=90)

ds['poutcome'].unique()

The above plot is interesting because the campaigns which are unknown, other and successful have maximum term deposits whereas the failed campaigns have equal response.

#### Now we will check the outliers using boxplots.

ds.head()

fig = px.box(ds, x="age", width = 600, height = 400,notched=True)
fig.show()

In the age column the outliers range between 75 to 90.

fig = px.box(ds, x="balance", width = 600, height = 400,notched=True)
fig.show()

In the balance column there are lots of outliers from 5k to 40k and some extreme outliers beyond 40k.

fig = px.box(ds, x="duration", width = 600, height = 400,notched=True)
fig.show()

In the duration column,most of outliers lie in range of 1000 to 2093 and some extreme outliers beyond 2093k.

fig = px.box(ds, x="pdays", width = 600, height = 400,notched=True)
fig.show()

pdays column also has maximum outliers between the range 60 to 561 and byond that there are xtreme values till 854.

#### Correlation

Lastly we will also use corelation matrix to check which columns are co-relatred to each other.

before that what is corelations?..It is the association between the columns which say how much they are depended or change or move in relation to another varibale/column.

It gives us the idea about the degree of the relationship of the two variables.

Types of Corelation:

A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that when one variable’s value increases, the other variables’ values decrease. Correlation can also be neutral or zero, meaning that the variables are unrelated.

Positive Correlation: both variables change in the same direction.

Neutral Correlation: No relationship in the change of the variables.

Negative Correlation: variables change in opposite directions.

sns.heatmap(ds.corr(),annot = True) # heatmap for corelation
#fig = px.imshow(ds.corr())
#fig.show()

The above corelation matrix says that there is no signigicant relation between the columns. You can see the color map at the right side where the corelation value is high the color gets darker.

### Feature Engineering

In this part we will remove the outliers and scale the data points down if required.

Now lets start by removing the outliers, for that we will be using IQR to remove them.

Steps to remove outliers using IQR

1) Sort the data in ascending order

2) find the first quantile(Q1) and the third quantile(Q3) values

3) Find it's IQR by substracting (Q1-Q2)

4) Set the lower bound i.e Q1*1.5

5) Set the upper bound i.e Q3*1.5

ds1 = ds.copy() # making a copy of the orignal data

ds1.head()

# method for removing outliers

def remove_outliers(lis):
    for i in lis:
        q1, q3 = np.percentile(ds[i],[25,75])
        iqr = q3-q1
        lowb = q1-(1.5*iqr)
        upb = q3+(1.5*iqr)
        print('upper bound for'+ ' '+i, upb)
        print('lower bound for'+' '+i, lowb)
        ds1.loc[ds1[i]>=upb,i] = upb
        ds1.loc[ds1[i]<=lowb,i] = lowb
    print('Removed!')


nums = [i for i in ds1.columns if ds1[i].dtype != 'O']
nums.remove('day')
nums.remove('campaign')
nums.remove('previous')
print(nums)
remove_outliers(nums)

fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,5))

fig.suptitle('After containing outliers')


sns.boxplot(ax=axes[0,0], x = ds1['age'])


sns.boxplot(ax=axes[0,1], x = ds1['balance'])


sns.boxplot(ax=axes[1,0], x = ds1['duration'])

sns.boxplot(ax=axes[1,1], x = ds1['pdays'])




So as we have removed the outliers successfully, we will proceed towards encoding the categorical features.



## Encoding

What is encoding?....Encoding is process of converting the categorical data into numerical representations. There are multiple encoding techniques that are available out of which we will be using target-guided-encoding for ordial features and we can use label encoding for the nomial features.

ds1.head()

min(ds1['balance'])

Let's start by encoding nomial features.

nominal = ['default', 'housing', 'loan', 'deposit']

le = LabelEncoder() # creating the object

def label_encode(lit):
    for i in lit:
        ds1[i] = le.fit_transform(ds1[i])   #  applying the encoder

label_encode(nominal)

ds1.head()

It's time to encode the categorical feautres using target guided encoding.

What is target-guided encdoing?....Its a type of an encdoing where encoding is done based on the target column in our case its 'income'. So whichever sub-categories contribute more to the target based on that the values will be assigned. If contribution is more than higher value will be assigned.

# method for performing target encoding

def ordi():
    k = ['job','marital','education','contact','month','poutcome']
    for l in k:
        ov = ds1.groupby([l])['deposit'].mean().sort_values().index
        print(ov)
        work_ol = {n:p for p,n in enumerate(ov,1)}
        print(work_ol)
        ds1['ordinal_'+l] = ds1[l].map(work_ol)

ordi() # calling the method



ds1.head() # checking the results

Once encoded now we can drop the original columns as we don't need it anymore.

# dropping the duplicate columns

ds1.drop(['job','marital','education','contact','month','poutcome'],axis = 1, inplace = True)

ds1.head() # checking the results

As we can see above the duration column is in seconds we need to convert it into minutes.

# method to convert seconds to minutes
mint =  []
def sec2min():
    for j in ds1['duration']:
        sec_value = j % (24 * 3600)
        sec_value %= 3600
        min_value = sec_value // 60
        sec_value %= 60
        #print(min_value)
        mint.append(int(min_value))
        

sec2min()

ds1['duration(in minutes)'] = mint # adding the new column to the dataframe

ds1.drop(['duration'],axis = 1, inplace = True) # dropping the duration column

ds1.head() # checking the results

One more thing we can do is that, scale down the values of 'balance' column using standard scaler so that it will be normally distributed.

sc = StandardScaler()
ds1['balance'] = sc.fit_transform(ds1[['balance']]) #  applying standard scaler

ds1.head() # checking the results

max(ds1['balance'])

#ds1.to_csv('BankAnalysis.csv')

x = ds1.drop('deposit',axis = 1) 
y = ds1['deposit']

ec = ExtraTreesClassifier()
ec.fit(x,y) # applying the classifier

# plotting the scores and values

f_s = pd.Series(ec.feature_importances_, index = x.columns, name = 'features')
fs = f_s.nlargest(10).plot(kind = 'barh')

fss = f_s.nlargest(10) # copying the series
ds2 = ds1[fss.index] # making a data frame of only selected features
ds2.head() # seeing the results

# separating x and y
X = ds2[:] 
Y = ds1['deposit']

# splitting data in train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def svmm(x_train, y_train, x_test, y_test):

    sv = SVC()
    m_sv = sv.fit(x_train, y_train)
    y_hat = m_sv.predict(x_test)
    y_pred_train = m_sv.predict(x_train)
    print('accuracy of SVM is:', accuracy_score(y_test, y_hat))
    print('accuracy of SVM is:', accuracy_score(y_train, y_pred_train))
    print(classification_report(y_test, y_hat))
    cm = confusion_matrix(y_test, y_hat)
    print(cm)

svmm(x_train,y_train,x_test,y_test) # calling svm


def dt(x_train,y_train, x_test, y_test):

    dct = DecisionTreeClassifier()
    m_sv = dct.fit(x_train,y_train)
    y_hat = m_sv.predict(x_test)
    print('accuracy of decison tree is:',accuracy_score(y_test, y_hat))
    print(classification_report(y_test, y_hat))
    cm = confusion_matrix(y_test, y_hat)
    print(cm)


dt(x_train,y_train,x_test,y_test)

def rf(x_train, y_train, x_test, y_test):

    rff = RandomForestClassifier()
    rf1 = rff.fit(x_train,y_train)
    y_hat = rf1.predict(x_test)
    print('accuracy of random forest is',accuracy_score(y_test, y_hat))
    print(classification_report(y_test, y_hat))
    cm = confusion_matrix(y_test, y_hat)
    return rf1

rf(x_train,y_train,x_test,y_test)

with open('saved_rf.pkl', 'wb') as wr:
   pickle.dump(rf, wr)
