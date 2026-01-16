import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
import math
from gd import *
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
# from sklearn.tree import DecisionTreeRegressor
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def shapImp(f_train, o_train):
    print("Wait for some time...")

    model = RandomForestRegressor()
    model = model.fit(f_train, o_train.values.ravel())

    explainer = shap.Explainer(model) 
    shap_values = explainer(f_train)   
    shap.plots.bar(shap_values,color='skyblue')

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
    

def featureImp(f_train,o_train):
    res = DecisionTreeRegressor()
    res.fit(f_train, o_train)
    feature_importances = res.feature_importances_
    feature_df = pd.DataFrame({"Feature": f_train.columns, "Importance": feature_importances})
    sorted_values = feature_df.sort_values(by="Importance", ascending=False)
    print(sorted_values)
    plt.figure(figsize=(10, 5))
    plt.barh(feature_df["Feature"], feature_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance from Decision Tree")
    plt.show()

def plotPred(y_test,y_pred):
        # Create figure
    plt.figure(figsize=(10, 6))

    # Plot actual vs. predicted values
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
    plt.scatter(y_test,y_test,alpha =0.5,label = 'Perfect Model')

    # Customize plot
 
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

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
# feature = dataset.drop(columns=["Life_expectancy","Country","Region","Population_mln","Economy_status_Developing","Economy_status_Developed"])
feature = dataset.drop(columns=["Life_expectancy","Country","Region"])
# model = LinearRegression() 
print(feature.columns)
output = dataset[["Life_expectancy","Year"]]
# y_norm = scaler.fit_transform(output)

x_train,x_test,y_train,y_test = split(feature,output)
feature = feature.drop(columns=["Year"])
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train,columns = feature.columns)

x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test,columns = feature.columns)

# print("x-norm",x_norm)
# x_norm["Year"] = output["Year"] # year normalize na ho 
# featureImp(x_train,y_train)
slcted_features = shapImp(x_train,y_train)
x_train = x_train[slcted_features] 
x_test = x_test[slcted_features]

print("after feature selection : ", x_train)

######## gd without using lib #########
# initial_w = np.ones(x_train.shape[1])  
# initial_b = 0.0
# w, b, cost_hist, w_hist = gradient_descent(
#     x_train, 
#     y_train.values.ravel() if hasattr(y_train, 'values') else y_train.ravel(),
#     initial_w,
#     initial_b,
#     compute_cost,
#     compute_gradient,
#     alpha=0.01,
#     num_iters=5000
# )
# print("Wj",w)
# print("b",b)
# y_pred = np.dot(x_test, w) + b
# plotPred(y_test.values,y_pred)
# cost = mean_squared_error(y_test, y_pred)
# print(f"Final cost (MSE): {cost:.2f}")

# ######## gd using lib #########
# gd = SGDRegressor(max_iter=5000)
# gd.fit(x_train, y_train)
# w = gd.coef_  
# b = gd.intercept_[0]
# print("Wj",w)
# print("b",b)
# y_pred = gd.predict(x_test)
# plotPred(y_test.values,y_pred)
# cost = mean_squared_error(y_test, y_pred)
# print(f"Final cost (MSE): {cost:.2f}")

######## linear regression model(skl) #########
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
plotPred(y_test.values,y_pred)
cost = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")
print(f"Final cost (MSE): {cost:.2f}")


######## polynomial regression #########
poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(x_train.values)
model = Lasso(alpha=0.01) 
model.fit(X_train_poly, y_train)
X_test_poly = poly.transform(x_test.values)  
y_pred = model.predict(X_test_poly)
plotPred(y_test.values, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")

cost = mean_squared_error(y_test, y_pred)
print(f"Final cost (MSE): {cost:.2f}")


