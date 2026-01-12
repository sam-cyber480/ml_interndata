import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


data = pd.read_csv("internship_data.csv")


X = data[['Python', 'Java', 'SQL', 'HTML_CSS', 'ML']]
y = data['Internship']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))


print("\nEnter your skills (1 = Yes, 0 = No)")
python = int(input("Python: "))
java = int(input("Java: "))
sql = int(input("SQL: "))
html = int(input("HTML/CSS: "))
ml = int(input("ML: "))

user_data = [[python, java, sql, html, ml]]
predicted = model.predict(user_data)

print("Recommended Internship:",
      encoder.inverse_transform(predicted)[0])
