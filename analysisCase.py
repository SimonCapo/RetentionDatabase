
#performance

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm


os.chdir(r'C:\Users\simon\Documents')
retention = pd.read_excel('RetentionExcel.xlsx', dtype={'RecordDate':'datetime64'})

dataset =  pd.read_excel('DataSetABS.xlsx')


#profit vs. cust satisfaction
X = dataset['customer_satisfaction'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = dataset['profit'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
np.nan_to_num(X)

np.nan_to_num(Y)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')

plt.ylabel('profit')
plt.xlabel('customer_satisfaction')
plt.axis([0, 10, -20000, 20000])
plt.show()


#engagement vs. cust satisfaction
X = dataset['engagement'].values.reshape(-1, 1)  # values converts it into a numpy array
Y = dataset['customer_satisfaction'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
np.nan_to_num(X)

np.nan_to_num(Y)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')

plt.ylabel('overall_engagement_stores')
plt.xlabel('customer satisfaction')
plt.axis([0, 10, 0, 10])
plt.show()


#gendergap
dataset = pd.get_dummies(dataset, columns = ['gender'])

X = dataset[['gender_M','employee_grade']]
Y = dataset['base_salary']

X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)







#high potentials are leaving (1/2)
retention = pd.get_dummies(retention, columns = ['talent_status'])
df = retention.groupby(['RecordDate'],as_index=False).sum()
X = df['RecordDate']
Y = df[['talent_status_Talent']]
plt.plot(X, Y, color='red')
plt.xticks(rotation=90)
plt.ylabel('talents leaving per month')
plt.xlabel('period')
plt.show()


#high potentials are leaving (2/2)
retention = pd.get_dummies(retention, columns = ['potential'])
df = retention.groupby(['RecordDate'],as_index=False).sum()
X = df['RecordDate']
Y = df[['potential_High']]
plt.plot(X, Y, color='red')
plt.xticks(rotation=90)
plt.ylabel('high potentials leaving per month')
plt.xlabel('period')
plt.show()


#are leavers engaged?
df = retention.groupby(['RecordDate'],as_index=False).mean()
X = df['RecordDate']
Y = df[['eng_emp_sat','eng_leadership' , 'eng_person_org_fit'  ]]
plt.plot(X, Y)
plt.legend(['Employee Satisfaction','Engaged With Leadership', 'Organisational Fit'])
plt.xticks(rotation=90)
plt.ylabel('average engagement of leavers')
plt.xlabel('period')
plt.show()


#how many leavers?
df = retention.groupby(['RecordDate'],as_index=False).count()
X = df['RecordDate']
Y = df['Source.Name']
plt.plot(X, Y, color='red')
plt.xticks(rotation=90)
plt.ylabel('total leavers per month')
plt.xlabel('period')
plt.show()

#Reason people leave
df = retention.groupby(['retention_risk_reason'],as_index=False).count()
X = df['retention_risk_reason']
Y = df['Source.Name']
xAxis = [i + 0.5 for i, _ in enumerate(X)]
plt.bar(xAxis, Y, color='teal')
plt.xlabel('Exit Reason', fontsize=14)
plt.ylabel('#Leavers', fontsize=14)
plt.xticks([i + 0.5 for i, _ in enumerate(X)], X)
plt.xticks(rotation=90)
plt.show()



#what drives engagement
dataset['Record Date']= pd.to_datetime(dataset['Record Date'])
dataset['date_in_service']=pd.to_datetime(dataset['date_in_service'])
dataset['YearsWorked'] = ((dataset['Record Date'] - dataset['date_in_service']).dt.days)/365
dataset = pd.get_dummies(dataset, columns = ['performance_status'])
X = dataset[['relative_salary_position',  'performance_status_Low', 'YearsWorked', 'customer_satisfaction']] 
Y = dataset['eng_emp_sat']

X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

