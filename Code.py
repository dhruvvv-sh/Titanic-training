import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the training dataset
df = pd.read_csv('train.csv')
#filling all the remaining rows with the median 
df["Age"] = df["Age"].fillna(df["Age"].median())
#converting the sex column rows into int64 data type
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
#creating a family Size to get a better survivability 
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
X = df[["Pclass", "Sex", "Age", "FamilySize"]]
y = df["Survived"]
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
#Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict on validation set
y_pred = model.predict(X_val)
# printing accuracy
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
#loading test data set
test_df = pd.read_csv("test.csv")
#Fill missing Age and Fare values with median
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
# Convert Sex column to numeric
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
#Create FamilySize feature
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
# Drop unnecessary columns
test_df.drop(columns=["Name", "Ticket", "Cabin", "Embarked", "SibSp", "Parch"], inplace=True)
X_test = test_df[["Pclass", "Sex", "Age", "FamilySize"]]
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.head(100)
