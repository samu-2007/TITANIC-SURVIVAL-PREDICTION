import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------- Title ----------------------
st.title("Titanic Survival Prediction")

# ---------------------- File Upload ----------------------
file = st.file_uploader("Upload Titanic CSV File", type="csv")

if file:

    # ---------------------- Load Dataset ----------------------
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # ---------------------- Data Cleaning / Filtering ----------------------

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop rows where Embarked is missing
    df.dropna(subset=['Embarked'], inplace=True)

    # Convert categorical variables to numeric
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # ---------------------- Feature Selection ----------------------
    features = [
        'Pclass',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Sex_male',
        'Embarked_Q',
        'Embarked_S'
    ]

    X = df[features]
    y = df['Survived']

    # ---------------------- Train Test Split ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------- Scaling ----------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------- Model ----------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ---------------------- Prediction ----------------------
    y_pred = model.predict(X_test)

    # ---------------------- Accuracy ----------------------
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    # ---------------------- Confusion Matrix ----------------------
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    st.pyplot(fig)
