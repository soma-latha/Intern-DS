import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import nltk

nltk.download('vader_lexicon')

try:
    data = pd.read_csv("/content/drive/MyDrive/INTERN-DS/imdb_top_1000.csv")
    print(data.head())
except FileNotFoundError:
    print("Error: 'imdb_top_1000.csv' not found.")
    exit()

data = data.dropna(subset=['Genre', 'IMDB_Rating', 'Gross', 'Overview'])
data['Gross'] = data['Gross'].replace(r'[\$,]', '', regex=True)
data['Gross'] = pd.to_numeric(data['Gross'], errors='coerce')
data['No_of_Votes'] = data['No_of_Votes'].replace(',', '', regex=True)
data['No_of_Votes'] = pd.to_numeric(data['No_of_Votes'], errors='coerce')
data = data.dropna(subset=['Gross', 'No_of_Votes'])

vader = SentimentIntensityAnalyzer()
data['compound'] = data['Overview'].apply(lambda x: vader.polarity_scores(str(x))['compound'])
data['sentiment_label'] = data['compound'].apply(lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral'))

data['Primary_Genre'] = data['Genre'].apply(lambda x: x.split(',')[0].strip())
data = pd.get_dummies(data, columns=['Primary_Genre'], drop_first=True)

X = data[['IMDB_Rating', 'No_of_Votes', 'compound'] + [col for col in data.columns if col.startswith('Primary_Genre_')]]
y = data['Gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Linear Regression R²:", round(r2_score(y_test, y_pred_lr), 3))
print("Linear Regression MSE:", round(mean_squared_error(y_test, y_pred_lr), 3))
print("Random Forest R²:", round(r2_score(y_test, y_pred_rf), 3))
print("Random Forest MSE:", round(mean_squared_error(y_test, y_pred_rf), 3))

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=data, hue='sentiment_label', palette='Set2', legend=False)
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Movies")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, color='teal', alpha=0.7)
plt.xlabel("Actual Revenue (Gross)")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Box Office (Random Forest)")
plt.show()

results = pd.DataFrame({
    'Title': data['Series_Title'],
    'Predicted_Revenue': rf.predict(X),
    'Sentiment': data['sentiment_label']
})
results.to_excel("movie_success_predictions.xlsx", index=False)
print("Results saved successfully to 'movie_success_predictions.xlsx'!")



