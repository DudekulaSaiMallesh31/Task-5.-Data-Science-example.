# Task-5.-Data-Science-example.
Consumer Complaints Text Classification
This project classifies consumer complaints into categories such as Credit Reporting, Debt Collection, Consumer Loan, and Mortgage based on the complaint text using various machine learning models. It uses a dataset containing consumer complaints from the Consumer Financial Protection Bureau (CFPB).

Requirements
Before running the code, ensure you have the following Python libraries installed:

pandas
numpy
scikit-learn
re (built-in Python module)
string (built-in Python module)
matplotlib
seaborn
You can install the required libraries using pip:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
Project Structure
Data Loading and Preprocessing:
![image](https://github.com/user-attachments/assets/66809f96-f446-4d85-9672-7fda619a73c7)


Loads the dataset of consumer complaints from a zip file.
Selects relevant columns (Product, Consumer complaint narrative) and cleans the data.
Maps categories to numerical values and applies text preprocessing.
Text Vectorization:

Converts text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
Model Training:

Trains and evaluates multiple machine learning models:
Naïve Bayes
Logistic Regression
Support Vector Machine (SVM)
Random Forest
Model Evaluation:

For each model, the script calculates and prints the accuracy score and classification report, which includes precision, recall, and F1-score.
Prediction Function:

A function that takes a complaint text and predicts its category using the trained model.
Steps to Run
Step 1: Load and Preprocess Data
python
Copy
Edit
import pandas as pd

# Load dataset
url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
df = pd.read_csv(url, compression='zip', low_memory=False)

# Selecting required columns
columns_needed = ["Product", "Consumer complaint narrative"]
df = df[columns_needed].dropna()
df.columns = ["Category", "Complaint"]

# Check the first few rows to ensure correct loading
print(df.head())
Step 2: Clean Data and Map Categories
![image](https://github.com/user-attachments/assets/ee89d3a0-afc1-4f0d-8761-a7f35eff6136)

python
Copy
Edit
import re
import string

# Mapping categories to numerical values
category_map = {
    "Credit reporting, repair, or other": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}
df = df[df["Category"].isin(category_map.keys())]
df["Category"] = df["Category"].map(category_map)

# Text Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\d+", "", text)  # Remove numbers
    return text

# Clean the complaint text
df["Complaint"] = df["Complaint"].apply(clean_text)

# Check the first few rows to ensure preprocessing is done correctly
print(df.head())
Step 3: Split Data into Train and Test
![image](https://github.com/user-attachments/assets/cea3721a-63d2-4fa3-bdc2-4bc9017b8736)

python
Copy
Edit
from sklearn.model_selection import train_test_split

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["Complaint"], df["Category"], test_size=0.2, random_state=42, stratify=df["Category"]
)

# Check the shape of train and test splits
print(f"Training Data Size: {X_train.shape}")
print(f"Test Data Size: {X_test.shape}")
Step 4: Convert Text to TF-IDF Features
python
Copy
Edit
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical features using TF-IDF
![image](https://github.com/user-attachments/assets/009ee759-64fc-4732-8ae3-09ba42f9ce8f)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Check the shape of the resulting TF-IDF matrices
print(f"TF-IDF Training Matrix Shape: {X_train_tfidf.shape}")
print(f"TF-IDF Test Matrix Shape: {X_test_tfidf.shape}")
Step 5: Train Naïve Bayes Model
python
Copy
Edit
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train Naïve Bayes model
![image](https://github.com/user-attachments/assets/18d702a9-fb60-44f0-92c8-04616dbaba08)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naïve Bayes Model Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
Step 6: Train Logistic Regression Model
python
Copy
Edit
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
![image](https://github.com/user-attachments/assets/9ae92ed9-826a-46e4-8103-b6062cbc0ade)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
Step 7: Train Support Vector Machine (SVM) Model
python
Copy
Edit
from sklearn.svm import SVC

# Train Support Vector Machine model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
Step 8: Train Random Forest Model
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred_rf = rf_model.predict(X_test_tfidf)
print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
Step 9: Prediction Function
python
Copy
Edit
# Prediction function
def predict_complaint(text, model):
    text_tfidf = vectorizer.transform([clean_text(text)])
    category_idx = model.predict(text_tfidf)[0]
    category = [k for k, v in category_map.items() if v == category_idx][0]
    return category

# Example prediction using Logistic Regression
example_text = "My credit report has incorrect information and they won't fix it."
print("Predicted Category:", predict_complaint(example_text, lr_model))
Results
Each model's evaluation metrics will be printed in the console, including accuracy, precision, recall, and F1-score. The models are evaluated using various metrics provided by classification_report.

License
This project is licensed under the MIT License - see the LICENSE file for details.
