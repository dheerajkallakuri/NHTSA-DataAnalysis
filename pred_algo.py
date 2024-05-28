import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pickle
import json
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

# Read the CSV file into a DataFrame
df = pd.read_csv('YOUR_DATA_FILE.csv') #ADS/ADAS

# Data preprocessing
categorical_cols = ['Make','Mileage', 'State', 'Roadway Type', 'Roadway Surface', 'Roadway Description',
                    'Posted Speed Limit (MPH)', 'Lighting', 'Weather - Clear', 'Weather - Snow', 'Weather - Severe Wind', 'Weather - Unknown', 'Weather - Other',
                    'Weather - Cloudy', 'Weather - Rain', 'Weather - Fog/Smoke', 
                    'SV Pre-Crash Movement','CP Pre-Crash Movement','SV Precrash Speed (MPH)', 'Incident Time (24:00)',
                    'Crash With']

df = df[categorical_cols]
# Identify columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()].tolist()
# Impute missing values for numerical columns
numerical_cols = df.select_dtypes(include='number').columns
for col in numerical_cols:
    if col in columns_with_missing_values:
        imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]])
df[numerical_cols] = df[numerical_cols].fillna(0)

df['Incident Time (24:00)']= df['Incident Time (24:00)'].replace(' ','00:00')
df['Incident Time (24:00)'] = pd.to_datetime(df['Incident Time (24:00)'], format='%H:%M')
df['Incident Time (24:00)'] = df['Incident Time (24:00)'].dt.hour * 60 + df['Incident Time (24:00)'].dt.minute
df['Incident Time (24:00)'] = df['Incident Time (24:00)'].fillna(0)

df['CP Pre-Crash Movement']= df['CP Pre-Crash Movement'].fillna('Unknown')
df['Mileage'] = df['Mileage'].fillna(0)
df['Posted Speed Limit (MPH)'] = df['Posted Speed Limit (MPH)'].fillna(0)
df['Weather - Clear'] = df['Weather - Clear'].replace(' ','N')
df['Weather - Snow'] = df['Weather - Snow'].replace(' ','N')
df['Weather - Cloudy'] = df['Weather - Cloudy'].replace(' ','N')
df['Weather - Rain'] = df['Weather - Rain'].replace(' ','N')
df['Weather - Fog/Smoke'] = df['Weather - Fog/Smoke'].replace(' ','N')
df['Weather - Severe Wind'] = df['Weather - Severe Wind'].replace(' ','N')
df['Weather - Unknown'] = df['Weather - Unknown'].replace(' ','N')
df['Weather - Other'] =df['Weather - Other'].replace(' ','N')
df['SV Precrash Speed (MPH)'] = df['SV Precrash Speed (MPH)'].fillna(0)
                   
# Initialize an empty dictionary to store unique values
unique_values_dict = {}
# Iterate over each column in the DataFrame
for column in categorical_cols:
    # Check if the column contains integer values
    if df[column].dtype == 'object':
        # Get unique values in the column
        unique_values = df[column].unique()

        # Create a dictionary mapping each unique value to an integer
        value_mapping = {value: idx for idx, value in enumerate(unique_values)}

        # Update the main dictionary with the mapping for this column
        unique_values_dict[column] = value_mapping

# Print or use the resulting dictionary
with open('integer_dict.json', 'w') as file:
    json.dump(unique_values_dict, file, indent=2)

# Define features (X) and target variable (y)
with open('integer_dict.json', 'r') as file:
    integer_dict = json.load(file)

for column, mapping in integer_dict.items():
    df[column] = df[column].map(mapping)

X = df.drop(columns=['Crash With'])
y = df['Crash With']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Load the trained model[adas_predmodel/ads_predmodel]
with open('YOUR_MODEL_FILE', 'wb') as model_file:
    pickle.dump(model, model_file)

# Model evaluation
y_pred = model.predict(X_test)

# Calculate accuracy and other classification metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# classification_report_result = classification_report(y_test, y_pred)
# print(f'Classification Report:\n{classification_report_result}')
 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', marker='o')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

mat=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(mat, annot=True, fmt='g', cmap='Blues', xticklabels=range(15), yticklabels=range(15))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()