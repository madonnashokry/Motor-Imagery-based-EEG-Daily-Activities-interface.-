# EEGModelsTest.py

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class EEG_Model_class:
    def __init__(self, data_path):
        self.data_path = data_path

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data

    def normalize_data(self,data):
        normalized_Electrodes = preprocessing.normalize(data, axis=0)
        normalized_Electrodes_copy = normalized_Electrodes.copy()
        normalized_Electrodes_copy -= np.mean(normalized_Electrodes_copy, axis=0)
        n_samples = normalized_Electrodes_copy.shape[0]
        pca = PCA()
        x = pca.fit_transform(normalized_Electrodes_copy)
        return x

    def load_data(self):
        label_mapping = {
            'right': 'drink',
            'left': 'sleep',
            'tongue': 'eat',
            'foot': 'bathroom'
        }
        data = pd.read_csv(self.data_path)
        columns = ['patient', 'label', 'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3',
                   'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12',
                   'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        data = data[columns]

        right = data[data.label == 'right']
        left = data[data.label == 'left']
        tongue = data[data.label == 'tongue']
        foot = data[data.label == 'foot']

        tongue_upsampled = resample(tongue, replace=True, n_samples=130248, random_state=123)
        foot_upsampled = resample(foot, replace=True, n_samples=130248, random_state=123)

        data_balanced = pd.concat([right, left, tongue_upsampled, foot_upsampled])
        X = data_balanced.drop(['patient', 'label'], axis=1)
        y = data_balanced['label']
        y = y.map(label_mapping)
        lowcut = 0.5
        highcut = 30
        fs = 250
        X = self.apply_bandpass_filter(X, lowcut, highcut, fs)
        X= self.normalize_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #######feature extraction
        csp = CSP(n_components=4)
        csp.fit(X_train_scaled[:, :, np.newaxis], y_train)
        X_train_csp = csp.transform(X_train_scaled[:, :, np.newaxis])
        X_test_csp = csp.transform(X_test_scaled[:, :, np.newaxis])
        X_train_csp = np.squeeze(X_train_csp)
        X_test_csp = np.squeeze(X_test_csp)
        return X_train_csp,X_test_csp ,y_train ,y_test

    def DecisionTreeClassifier(self):
        X_train_csp,X_test_csp ,y_train ,y_test=self.load_data()
        dtc_clf = DecisionTreeClassifier(max_depth=20, random_state=42)
        dtc_clf.fit(X_train_csp, y_train)
        y_pred = dtc_clf.predict(X_test_csp)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return y_pred
    def LogisticRegression_classifier(self):
        X_train_csp,X_test_csp ,y_train ,y_test=self.load_data()
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train_csp, y_train)
        y_pred = clf.predict(X_test_csp)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return y_pred
    def RandomForest_class(self):
        X_train_csp,X_test_csp ,y_train ,y_test=self.load_data()
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=6)
        rf_clf.fit(X_train_csp, y_train)
        y_pred = rf_clf.predict(X_test_csp)
        accuracy_rf = accuracy_score(y_test, y_pred)
        print(f'Random Forest - Accuracy: {accuracy_rf * 100:.2f}%')
        return y_pred

