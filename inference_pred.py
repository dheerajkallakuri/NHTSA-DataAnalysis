import sys
import pickle
import json
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QVBoxLayout,QLabel, QComboBox, QDoubleSpinBox, QPushButton, QMessageBox, QTimeEdit
from PyQt5.QtCore import QTime
from sklearn.preprocessing import LabelEncoder

class InjurySeverityPredictor(QWidget):
    def __init__(self):
        super().__init__()

        # Load the trained model[adas_predmodel.pkl/ads_predmodel.pkl]
        with open('YOUR_MODEL_FILE', 'rb') as model_file:
            self.model = pickle.load(model_file)

        # Define the features for prediction
        self.feature_info = {
            'Make': 'categorical',
            'Mileage': 'numeric',
            'State': 'categorical',
            'Roadway Type': 'categorical',
            'Roadway Surface': 'categorical',
            'Roadway Description': 'categorical',
            'Posted Speed Limit (MPH)': 'numeric',
            'Lighting': 'categorical',
            'Weather - Clear': 'categorical',
            'Weather - Snow': 'categorical',
            'Weather - Severe Wind': 'categorical',
            'Weather - Unknown': 'categorical', 
            'Weather - Other': 'categorical',
            'Weather - Cloudy': 'categorical',
            'Weather - Rain': 'categorical',
            'Weather - Fog/Smoke': 'categorical',
            'SV Pre-Crash Movement': 'categorical',
            'CP Pre-Crash Movement': 'categorical',
            'SV Precrash Speed (MPH)': 'numeric',
            'Incident Time (24:00)': 'numeric'
        }
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Crash With?')
        self.setGeometry(100, 100, 400, 300)

        layout = QGridLayout()

        row = 0
        for feature, data_type in self.feature_info.items():
            label = QLabel(feature)
            widget = self.create_input_widget(data_type, feature)
            setattr(self, f'{feature}_widget', widget)

            layout.addWidget(label, row, 0)

            if data_type == 'numeric':
                row += 1
                layout.addWidget(widget, row, 0, 1, 4)  # span the widget across 4 columns
            else:
                layout.addWidget(widget, row, 1)

            row += 1

        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.predict)
        layout.addWidget(predict_button, row, 0, 1, 4)  # span the button across 4 columns

        self.setLayout(layout)
        self.setFixedSize(self.sizeHint())  # Make the GUI resizable
        self.setMinimumSize(400, 300)  # Set a minimum size
    
    def create_input_widget(self, data_type,feature):
        
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
        if data_type == 'categorical':
            combo_box = QComboBox()
            combo_box.addItems([''] + df[feature].unique().astype(str).tolist())
            return combo_box
        elif data_type == 'numeric' and feature != 'Incident Time (24:00)':
            spin_box = QDoubleSpinBox()
            spin_box.setRange(0, 150000)
            return spin_box
        elif feature == 'Incident Time (24:00)':
            time_picker = QTimeEdit(self)
            time_picker.setDisplayFormat("HH:mm")
            return time_picker
        
    def predict(self):
        categorical_cols = ['Make','Mileage', 'State', 'Roadway Type', 'Roadway Surface', 'Roadway Description',
                    'Posted Speed Limit (MPH)', 'Lighting', 'Weather - Clear', 'Weather - Snow', 'Weather - Severe Wind', 'Weather - Unknown', 'Weather - Other',
                    'Weather - Cloudy', 'Weather - Rain', 'Weather - Fog/Smoke', 
                    'SV Pre-Crash Movement','CP Pre-Crash Movement','SV Precrash Speed (MPH)','Incident Time (24:00)']
        input_data = {}
        for feature, data_type in self.feature_info.items():
            if feature == 'Incident Time (24:00)':
                time_widget = getattr(self, f'{feature}_widget')
                selected_time = time_widget.time()
                # input_data[feature] = selected_time.toString("HH:mm")
                input_data[feature] = selected_time.hour() * 60 + selected_time.minute()
            else:
                widget = getattr(self, f'{feature}_widget')
                input_data[feature] = widget.currentText() if data_type == 'categorical' else widget.value()

        input_df = pd.DataFrame([input_data], columns=categorical_cols)

        with open('integer_dict.json', 'r') as file:
            integer_dict = json.load(file)

        for column, mapping in integer_dict.items():
            if column != 'Crash With':
                input_df[column] = input_df[column].map(mapping)
        print(input_df)

        try:
            prediction_int = self.model.predict(input_df)[0]
            crash_with_dict = integer_dict.get('Crash With')
            prediction=[key for key, value in crash_with_dict.items() if value == prediction_int][0]
            print(prediction)
            QMessageBox.information(self, 'Prediction', f'Crash With: {prediction}')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Error during prediction: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    df = pd.read_csv('YOUR_DATA_FILE.csv') #ADS/ADAS
    categorical_cols = ['Make','Mileage', 'State', 'Roadway Type', 'Roadway Surface', 'Roadway Description',
                    'Posted Speed Limit (MPH)', 'Lighting', 'Weather - Clear', 'Weather - Snow', 'Weather - Severe Wind', 'Weather - Unknown', 'Weather - Other',
                    'Weather - Cloudy', 'Weather - Rain', 'Weather - Fog/Smoke', 
                    'SV Pre-Crash Movement','CP Pre-Crash Movement','SV Precrash Speed (MPH)','Incident Time (24:00)']

    df = df[categorical_cols]
    df['Incident Time (24:00)']= df['Incident Time (24:00)'].replace(' ','00:00')
    df['Incident Time (24:00)'] = pd.to_datetime(df['Incident Time (24:00)'], format='%H:%M')
    df['Incident Time (24:00)'] = df['Incident Time (24:00)'].dt.hour * 60 + df['Incident Time (24:00)'].dt.minute
    window = InjurySeverityPredictor()
    window.show()
    sys.exit(app.exec_())