import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

def shapImp(f_train, o_train):
    print("Wait for some time...")

    model = RandomForestRegressor()
    model = model.fit(f_train, o_train.values.ravel())

    explainer = shap.Explainer(model) 
    shap_values = explainer(f_train)   
    shap.plots.bar(shap_values)

    shap_imp = shap_values.values  
    abs_imp = np.abs(shap_imp)  
    mean_imp = abs_imp.mean(axis=0)  
    shap_imp = pd.DataFrame({'Feature': f_train.columns, 'Importance': mean_imp})
    sorted_values = shap_imp.sort_values('Importance', ascending=False)

    print(sorted_values)
    total = sorted_values['Importance'].sum()
    sorted_values['Cumulative'] = sorted_values['Importance'].cumsum() / total
    selected = []
    for i, row in sorted_values.iterrows():
        if row['Cumulative'] <= 0.95:
            selected.append(row['Feature'])
    print("Selected Features are:", selected)
    return selected

def log_reg(f_train,f_test,o_train,o_test):
    # model = LogisticRegressionCV(penalty='l1', solver='saga', max_iter=10000)
    model = LogisticRegression()
    model.fit(f_train,o_train)
    o_pred = model.predict(f_test)
    acc = accuracy_score(o_test,o_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(o_test, o_pred))


def split(feature,output):
    #features 
    train_data = []
    test_data = []
    for index,row in feature.iterrows():
        if row["Year"] < 2013:
            train_data.append(row.values)
        else:
            test_data.append(row.values)          
    #output 
    train_out = []
    test_out = []
    for index,row in output.iterrows():
        if row["Year"] < 2013:
            train_out.append(row.values)
        else:
            test_out.append(row.values)    

    f_train = pd.DataFrame(train_data, columns=feature.columns)
    f_train = f_train.drop(columns=["Year"])
    f_test = pd.DataFrame(test_data, columns=feature.columns)
    f_test = f_test.drop(columns=["Year"])
    o_train = pd.DataFrame(train_out, columns=output.columns)
    o_train = o_train.drop(columns=["Year"])
    o_test = pd.DataFrame(test_out, columns=output.columns)
    o_test = o_test.drop(columns=["Year"])
    return f_train,f_test,o_train,o_test

dataset = pd.read_csv("Life-Expectancy-Data-Updated.csv")

#for developed country
feature = dataset.drop(columns = ['Economy_status_Developed','Economy_status_Developing'])
output = dataset[['Economy_status_Developed','Year']]
# feature = pd.get_dummies(feature, columns=['Country', 'Region'], drop_first=True)
# print(feature)
# print(feature.head())
# print(feature.iloc[0])
# print(f"Number of features: {feature.shape[1]}")
feature = feature.drop(columns=['Country','Region'])
# print(f"Number of features: {feature.shape[1]}")

x_train,x_test,y_train,y_test = split(feature,output)
pol = PolynomialFeatures(degree=2, include_bias=False)

feature = feature.drop(columns=['Year'])
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train,columns = feature.columns)
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test,columns = feature.columns)

slcted_features = shapImp(x_train,y_train)
x_train = x_train[slcted_features] 
x_test = x_test[slcted_features]
print(", ".join(slcted_features))

x_train_pol = pol.fit_transform(x_train)
x_test_pol = pol.transform(x_test)


print("For developed Country")
print("Multiple Logistic Regression")
log_reg(x_train,x_test,y_train,y_test)
print("Polynomial Logistic Regression")
log_reg(x_train_pol,x_test_pol,y_train,y_test)

#for developing country
feature = dataset.drop(columns = ['Economy_status_Developing','Economy_status_Developed'])
output = dataset[['Economy_status_Developing','Year']]
# feature = pd.get_dummies(feature, columns=['Country', 'Region'], drop_first=True)
# print(feature)
# print(feature.head())
# print(feature.iloc[0])
# print(f"Number of features: {feature.shape[1]}")
feature = feature.drop(columns=['Country','Region'])
# print(f"Number of features: {feature.shape[1]}")

x_train,x_test,y_train,y_test = split(feature,output)

feature = feature.drop(columns=['Year'])
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train,columns = feature.columns)
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test,columns = feature.columns)

slcted_features = shapImp(x_train,y_train)
x_train = x_train[slcted_features] 
x_test = x_test[slcted_features]
print(", ".join(slcted_features))
x_train_pol = pol.fit_transform(x_train)
x_test_pol = pol.transform(x_test)

# print("For developing Country")
print("Multiple Logistic Regression")
log_reg(x_train,x_test,y_train,y_test)
print("Polynomial Logistic Regression")
log_reg(x_train_pol,x_test_pol,y_train,y_test)





