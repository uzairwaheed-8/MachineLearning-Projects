import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


def preprocessing(dataset): 
    dataset.drop(columns=["id"], inplace=True)
    dataset["bmi"].fillna(dataset["bmi"].median(), inplace=True)
    categorical_columns = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()     #text -> number
        dataset[col] = le.fit_transform(dataset[col]) # unique value identify then replace category with integer
        label_encoders[col] = le # save encoder object to decode 
    return dataset 

def KNN(features,out):
    f_train, f_test, o_train, o_test = train_test_split(features, out, test_size=0.3, stratify=out)
    acc_arr =[]
    k=[]
    for i in range(23):
        if(i%2 != 0):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(f_train, o_train)
            o_pred = knn.predict(f_test)
            acc=accuracy_score(o_test, o_pred)
            print(f"K={i} ",acc)
            acc_arr.append(acc)

    m = max(acc_arr)
    ind = acc_arr.index(m)
    k_value = 2*ind +1  //kynke k value odd ho gi 
    k= list(range(1, 23, 2))
    plt.xticks(k)  //k  ki original value plot ho 
    plt.scatter(k,acc_arr,color='b')
    plt.xlabel("K-value")
    plt.ylabel("Accuracy")
    plt.title("k-values VS Accuracy")
    plt.show()
    print("Best K-value is ",k_value)                 
    return m

def plot(feature_df):
    plt.figure(figsize=(10, 5))
    plt.barh(feature_df["Feature"], feature_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance from Decision Tree")
    plt.show()


def feature_selection(feature,output,dataset):
    print("FEATURE SELECTION: ")
    #using decision tree to check the importance of features 
    f_train, f_test, o_train, o_test = train_test_split(feature, output, test_size=0.3, stratify=output)#stratify-> for same proportion
    acc10 = KNN(feature,output)
    print("Accurracy with 10-f",acc10)
    res = DecisionTreeClassifier()
    res.fit(f_train, o_train)
    feature_importances = res.feature_importances_
    feature_df = pd.DataFrame({"Feature": f_train.columns, "Importance": feature_importances})
    sorted_values = feature_df.sort_values(by="Importance", ascending=False)
    print(sorted_values)
    plot(sorted_values)

    features=dataset.drop(columns=[sorted_values.Feature[9],sorted_values.Feature[8],sorted_values.Feature[7]])
    out = dataset["stroke"]
    acc = KNN(features,out)
    print("Accurracy with 7-f",acc)

    features=dataset.drop(columns=[sorted_values.Feature[9],sorted_values.Feature[8],sorted_values.Feature[7],
    sorted_values.Feature[6],sorted_values.Feature[5]])
    out = dataset["stroke"]
    acc = KNN(features,out)
    print("Accurracy with 5-f",acc)
    
    features=dataset[[sorted_values.Feature[0], sorted_values.Feature[1], sorted_values.Feature[2]]]
    out = dataset["stroke"]
    acc = KNN(features,out)
    print("Accurracy with 3-f",acc)


def Pca(feature,output):
    print("PCA : ")
    acc_pca = []
    for i in range(1,11): 
        pca = PCA(n_components=i) 
        pca_features = pca.fit_transform(feature) #learns patterns from dataset and transformed into new reduced dimensions
        acc = KNN(pca_features,output) 
        print(f"Accuracy with PCA{i}:",acc)
        acc_pca.append(acc)
    m = max(acc_pca) 
    ind = acc_pca.index(m)
    dim = ind + 1  #where return tuple use index 0 for array 
    print("Selected-PCA Dimensions are ",dim)
    return m    

def Lda(feature,output):
    print("LDA:")
    lda = LinearDiscriminantAnalysis(n_components=1) 
    lda_features = lda.fit_transform(feature,output) #learns best linear discrimiant that separate output and then projects it to LDA axis
    acc = KNN(lda_features,output) 
    print("Accuracy with LDA:",acc)

def tSNE(feature,output):
    print("t_SNE: ")
    tsne = TSNE(n_components=2, perplexity=30) 
    tsne_feature = tsne.fit_transform(feature)
    acc = KNN(tsne_feature,output)
    print("Accuracy with tSNE: ",acc)
    
#main
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")
dataset = preprocessing(dataset)
feature = dataset.drop(columns=["stroke"])
output = dataset["stroke"]
feature_selection(feature,output,dataset)
Pca(feature,output)
Lda(feature,output)
tSNE(feature,output)
# print(dataset)



