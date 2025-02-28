# Assessing-Customer-Churn-Using-Machine-Learning
![IMG_8811](https://github.com/user-attachments/assets/313ebfd4-537f-4887-90c0-40a3d19ab830)
The telecommunications (telecom) sector in India is rapidly changing, with more and more telecom businesses being created and many customers deciding to switch between providers. "Churn" refers to the process where customers or subscribers stop using a company's services or products. Understanding the factors that influence keeping a customer as a client in predicting churn is crucial for telecom companies to enhance their service quality and customer satisfaction. As the data scientist on this project, you aim to explore the intricate dynamics of customer behavior and demographics in the Indian telecom sector in predicting customer churn, utilizing two comprehensive datasets from four major telecom partners: Airtel, Reliance Jio, Vodafone, and BSNL:

telecom_demographics.csv contains information related to Indian customer demographics:
Variable	Description
customer_id 	Unique identifier for each customer.
telecom_partner 	The telecom partner associated with the customer.
gender 	The gender of the customer.
age 	The age of the customer.
state	The Indian state in which the customer is located.
city	The city in which the customer is located.
pincode	The pincode of the customer's location.
registration_event	When the customer registered with the telecom partner.
num_dependents	The number of dependents (e.g., children) the customer has.
estimated_salary	The customer's estimated salary.
telecom_usage contains information about the usage patterns of Indian customers:
Variable	Description
customer_id	Unique identifier for each customer.
calls_made	The number of calls made by the customer.
sms_sent	The number of SMS messages sent by the customer.
data_used	The amount of data used by the customer.
churn	Binary variable indicating whether the customer has churned or not (1 = churned, 0 = not churned).



# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Start your code here!

#Created the dataframes
df1=pd.read_csv('telecom_demographics.csv')
df2=pd.read_csv('telecom_usage.csv')

#Merge the dataframes
churn_df = df1.merge(df2, how='inner', on='customer_id')
churn_df.head()

#Calculate churn rate
churn_rate=(churn_df['churn'].value_counts())/len(churn_df)
print("Churn Rate:\n", churn_rate)

churn_df.info()


# Encoding categorical features
cat_var=['telecom_partner', 'gender', 'state', 'city', 'registration_event']
churn_df=pd.get_dummies(churn_df, columns=cat_var)

#Scaling data using StandardScaler
scaler = StandardScaler()

#X and y variable, we have taken features variable to do feature scaling
features = churn_df.drop(['customer_id', 'churn'], axis=1)
features_scaled = scaler.fit_transform(features)
target=churn_df['churn']

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

## Initializing Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

## Initializing RandomForest model

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred=rf.predict(X_test)


# Logistic Regression evaluation
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

#Random forest evaluation

print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


## Which accuracy score is higher? Ridge or RandomForest
higher_accuracy = "RandomForest"
