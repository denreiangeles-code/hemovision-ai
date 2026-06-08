import sys
import os
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QRubberBand, QHBoxLayout, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QImageReader
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, QByteArray, QBuffer, QIODevice

# NOTE: For deployment, you would import tflite_runtime.interpreter here
# import tflite_runtime.interpreter as tflite

class CroppableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.original_pixmap = None

    def set_base_image(self, pixmap):
        # Save the original so we can "reset" if the user messes up the crop
        self.original_pixmap = pixmap
        self.setPixmap(pixmap)

    def mousePressEvent(self, ev):
        # The Guard: Abort if ev is None OR if there is no image
        if ev is None or not self.original_pixmap: return
        
        self.origin = ev.pos()
        self.rubber_band.setGeometry(QRect(self.origin, QSize()))
        self.rubber_band.show()

    def mouseMoveEvent(self, ev):
        if ev is None or not self.original_pixmap: return
        
        # Draw the physical box as the user drags their mouse
        self.rubber_band.setGeometry(QRect(self.origin, ev.pos()).normalized())

    def mouseReleaseEvent(self, ev):
        if ev is None or not self.original_pixmap: return
        
        self.rubber_band.hide()
        rect = self.rubber_band.geometry()
        
        if rect.width() > 10 and rect.height() > 10:
            # THE FIX: Take a literal UI "screenshot" of the selected area
            # This completely bypasses all the complex coordinate scaling math!
            cropped_pixmap = self.grab(rect)
            self.setPixmap(cropped_pixmap)
            
    def reset_crop(self):
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap)

class HemoVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HemoVision AI - Clinical Screening")
        self.setFixedSize(600, 700)
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # THE FIX: Rename to main_layout to avoid overriding PyQt6's internal .layout() method
        self.main_layout = QVBoxLayout(central_widget)
        
        # 1. The Capture Guidelines
        self.setup_guidelines_ui()
        
        # 2. Image Display Area
        self.image_label = CroppableLabel() # Using our custom class
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background: #eee; color: #555555;")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setScaledContents(True) # Ensures the crop maps perfectly to the pixels
        self.main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 3. Controls
        btn_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.upload_btn)
        
        self.reset_btn = QPushButton("Reset Crop")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_image)
        btn_layout.addWidget(self.reset_btn)
        
        self.main_layout.addLayout(btn_layout)
        
        self.analyze_btn = QPushButton("Run AI Analysis")
        self.analyze_btn.setEnabled(False) 
        self.analyze_btn.clicked.connect(self.run_inference)
        self.main_layout.addWidget(self.analyze_btn)
        
        # 4. Results Display
        self.result_label = QLabel("Predicted Hgb: -- g/dL")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-top: 20px;")
        self.main_layout.addWidget(self.result_label)
        
        self.current_image_path = None

        # --- NEW: LOAD THE AI BRAIN ON STARTUP ---
        self.result_label.setText("Loading AI Engine...")
        QApplication.processEvents() # Force UI to show loading text
        
        # THE FIX: Calculate the path BEFORE the try block so it always exists
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'hemovision_final_model.keras')
        
        try:
            # Load using the pre-calculated absolute pathway
            self.model = tf.keras.models.load_model(model_path)
            
            self.result_label.setText("System Ready. Upload an image.")
            
        except Exception as e:
            self.result_label.setText("Error: Model not found!")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #d32f2f;")
            print(f"Model Load Error: {e}")
            # Now Pylance knows model_path is 100% guaranteed to exist here
            print(f"Attempted to look in: {model_path}")

    def setup_guidelines_ui(self):
        rules_text = """
        <b>📸 IMAGE CAPTURE RULES:</b><br>
        1. <b>Pull Down:</b> Gently pull the lower eyelid down completely.<br>
        2. <b>Lighting:</b> Use flash or bright natural daylight.<br>
        3. <b>Distance:</b> Hold camera 4-6 inches away.<br>
        4. <b>No Filters:</b> Ensure native camera smoothing is turned OFF.
        """
        rules_label = QLabel(rules_text)
        rules_label.setStyleSheet("background: #e1f5fe; color: #333333; padding: 15px; border-radius: 5px;")
        rules_label.setWordWrap(True)
        self.main_layout.addWidget(rules_label) # Also updated here!

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Eye Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image_path = file_name
            
            # THE FIX: Use QImageReader to automatically apply EXIF rotation
            reader = QImageReader(file_name)
            reader.setAutoTransform(True) # This stops mobile photos from going sideways
            img = reader.read()
            
            # Convert to Pixmap and scale it smoothly to fit the 400x300 box
            pixmap = QPixmap.fromImage(img)
            scaled_pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.set_base_image(scaled_pixmap)
            
            self.analyze_btn.setEnabled(True) 
            self.reset_btn.setEnabled(True)

    def reset_image(self):
        self.image_label.reset_crop()

    def run_inference(self):
        if not self.current_image_path:
            return
            
        self.result_label.setText("Analyzing tissue pallor...")
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-top: 20px;")
        QApplication.processEvents() # Force UI to update before heavy math
        
        try:
            # 1. Safely extract the cropped image from the UI
            cropped_pixmap = self.image_label.pixmap()
            
            # 2. Convert PyQt6 Pixmap to a raw byte array in memory
            buffer = QByteArray()
            buffer_device = QBuffer(buffer)
            buffer_device.open(QIODevice.OpenModeFlag.WriteOnly)
            cropped_pixmap.save(buffer_device, "PNG")
            
            # 3. Read the bytes into a NumPy array for OpenCV
            nparr = np.frombuffer(buffer.data(), np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # THE NEW FIX: Guard against a failed memory decode
            if img_cv is None:
                self.result_label.setText("Error: Image decoding failed.")
                self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #d32f2f;")
                return
            
            # 4. Convert BGR (OpenCV default) to RGB (TensorFlow requirement)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # 5. Resize to exactly match the AI's training data (224x224)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # 6. Apply MobileNetV2's exact mathematical scaling [-1 to 1]
            img_array = np.expand_dims(img_resized, axis=0)
            img_preprocessed = preprocess_input(img_array)
            
            # 7. RUN THE PREDICTION
            prediction = self.model.predict(img_preprocessed, verbose=0)
            
            # Extract the float value from the resulting tensor
            predicted_hgb = float(prediction[0][0])
            
            # 8. Update the UI based on the clinical threshold
            self.result_label.setText(f"Predicted Hgb: {predicted_hgb:.2f} g/dL")
            
            # Standard threshold: Under 12.0 is generally considered anemic
            if predicted_hgb < 12.0:
                self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #d32f2f;") # Red alert
            else:
                self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #388e3c;") # Green clear
                
        except Exception as e:
            self.result_label.setText("Analysis Error!")
            print(f"Inference Error: {e}")
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HemoVisionApp()
    window.show()
    sys.exit(app.exec())