import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function for feature engineering
def feature_engineering(df):
    # Convert categorical variables to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    return df

# Function for model training and evaluation
def train_evaluate_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=45)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=66)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict survival on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Create a DataFrame to compare predicted survival with original features
    results_df = pd.DataFrame({'PassengerId': X_test.index, 'Survived': y_test, 'Predicted_Survived': y_pred})

    return results_df

# Read the Titanic dataset from the CSV file
df = pd.read_csv('titanic_data.csv')

# Feature engineering
df = feature_engineering(df)

# Select features and target variable
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Train and evaluate the model
results_df = train_evaluate_model(X, y)

# Save the results to a CSV file
results_df.to_csv('predicted_survival_random_forest.csv', index=False)
