import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit, QComboBox ,QLabel
from EEGModels import EEG_Model_class

class EEGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("EEG Motor Imagery Classifier")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout(self.central_widget)

        # Upload button
        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_button)

        # Feature selection
        self.feature_label = QLabel("Feature Selection:")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["Decision Tree", "Logistic Regression", "RandomForest"])
        self.feature_layout = QHBoxLayout()
        self.feature_layout.addWidget(self.feature_label)
        self.feature_layout.addWidget(self.feature_combo)
        self.layout.addLayout(self.feature_layout)

        # Run button
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)
        self.layout.addWidget(self.run_button)

        # Output text box
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

    def upload_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Select EEG file", "", "CSV Files (*.csv)")
        if self.filepath:
            QMessageBox.information(self, "File Selected", f"File selected: {self.filepath}")

    def run_model(self):
        if not self.filepath:
            QMessageBox.warning(self, "No File", "Please upload a file first.")
            return

        feature = self.feature_combo.currentText()
        if feature == "Decision Tree":
            model = EEG_Model_class(self.filepath)
            y_pred = model.DecisionTreeClassifier()

        elif feature == "Logistic Regression":
            model = EEG_Model_class(self.filepath)
            y_pred = model.LogisticRegression_classifier()

        elif feature == "RandomForest":
            model = EEG_Model_class(self.filepath)
            y_pred = model.RandomForest_class()

        else:
            QMessageBox.critical(error_type=QMessageBox.Warning, text="Invalid Selection", detail="Please select a valid feature.")
            return

        self.output_text.clear()
        for pred in y_pred:
            self.output_text.append(f"{pred}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGApp()
    window.show()
    sys.exit(app.exec_())


'''import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, \
    QMessageBox, QTextEdit, QComboBox, QLabel, QProgressBar
from matplotlib.backends.backend_qt import MainWindow

from EEGModels import Decisiontree_class, LogisticRegression_class, RandomForest_class


class EEGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Brain-Computer Interface")
        self.setGeometry(450, 90, 1000, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Change background color
        self.central_widget.setAutoFillBackground(True)
        p = self.central_widget.palette()
        p.setColor(self.central_widget.backgroundRole(), Qt.lightGray)
        self.central_widget.setPalette(p)

        self.layout = QVBoxLayout(self.central_widget)

        # Main label
        self.label = QLabel("Please think of an action and press 'Appear Signal'")
        font = QFont()
        font.setPointSize(16)  # Larger font size
        font.setBold(True)  # Bold font
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)  # Center alignment

        self.pbar = QProgressBar()

        # Button with larger and bold font
        buttonFont = QFont()
        buttonFont.setPointSize(12)  # Larger font size for buttons
        buttonFont.setBold(True)  # Bold font for buttons

        self.upload_button = QPushButton("Upload EEG Data")
        self.upload_button.setFont(buttonFont)
        self.qus_button = QPushButton("Appear Signal")
        self.qus_button.setFont(buttonFont)

        # Feature selection (similar to code 1's option labels)
        self.feature_label = QLabel("Feature Selection:")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["Decision Tree", "Logistic Regression", "RandomForest"])
        self.feature_layout = QHBoxLayout()
        self.feature_layout.addWidget(self.feature_label)
        self.feature_layout.addWidget(self.feature_combo)

        # Add widgets to the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.pbar)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.qus_button)
        self.layout.addLayout(self.feature_layout)  # Include feature selection

        # Adjust spacing between labels (optional)
        # ... (similar to code 1)

        # Add spacers around buttons for centering (optional)
        # ... (similar to code 1)

        self.upload_button.clicked.connect(self.upload_file)
        self.qus_button.clicked.connect(self.show_options)

    def upload_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Select EEG file", "", "CSV Files (*.csv)")
        if self.filepath:
            # Simulate processing
            for i in range(101):
                QApplication.processEvents()
                self.pbar.setValue(i)
            QMessageBox.information(self, "Success", "File Uploaded Successfully!")
            # Replace with your processing logic using Decisiontree_class or LogisticRegression_class
            # ... your processing code here ...
            return self.filepath  # Return the filepath for later use

    def show_options(self):
        # Hide initial elements (similar to code 1)
        self.label.hide()
        self.pbar.hide()
        self.upload_button.hide()
        self.qus_button.hide()

        # Message label (similar to code 1, customize message)
        self.message_label = QLabel("Brain signal received. Please choose your desired action:")
        self.message_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.message_label.setAlignment(Qt.AlignCenter)
        self.layout.insertWidget(0, self.message_label)  # Insert at the top


    def show_options(self):
        # Hide buttons, progress bar, and the initial label
        self.button.hide()
        self.qus_button.hide()
        self.pbar.hide()
        self.label.hide()

        # "Message appear here" label
        self.message_label = QLabel("Message appear here")
        self.message_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.message_label.setAlignment(Qt.AlignCenter)
        self.layout.insertWidget(0, self.message_label)  # Insert at the top

        # Show the option labels
        for label in self.option_labels:
            label.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())'''
