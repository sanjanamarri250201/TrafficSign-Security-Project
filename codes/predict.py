import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# --- Parameters ---
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75  # Minimum probability to display label
font = cv2.FONT_HERSHEY_SIMPLEX

# --- Load Model & Labels ---
# Make sure "model_trained.h5" and "labels.csv" are in the same folder
model = load_model("model_trained.h5") 
labels_df = pd.read_csv("labels.csv")

def getClassName(classNo):
    # Ensure classNo is treated as an integer for the lookup
    try:
        name = labels_df.loc[labels_df['ClassId'] == int(classNo), 'Name'].values[0]
        return name
    except:
        return "Unknown"

# --- Preprocessing (Must match Training Preprocessing) ---
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# --- Setup Camera ---
cap = cv2.VideoCapture(0) # 0 for default, 1 for external
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

print("Starting Camera... Press 'q' or close the window to exit.")

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Error: Could not read from camera.")
        break
        
    # --- Process Image for Model ---
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    # Reshape to (1, 32, 32, 1) to match model input
    img = img.reshape(1, 32, 32, 1)
    
    # --- Predict ---
    predictions = model.predict(img, verbose=0) # verbose=0 removes console log spam
    classIndex = np.argmax(predictions, axis=-1)[0]
    probabilityValue = np.amax(predictions)
    
    # --- Draw UI on Original Frame ---
    cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2)
    cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2)
    
    if probabilityValue > threshold:
        name = getClassName(classIndex)
        cv2.putText(imgOriginal, f"{classIndex} {name}", (120, 35), font, 0.75, (0, 255, 0), 2)
        cv2.putText(imgOriginal, f"{round(probabilityValue*100, 2)}%", (200, 75), font, 0.75, (0, 255, 0), 2)
    
    # Show the result
    cv2.imshow("Traffic Sign Recognition", imgOriginal)
    
    # --- Keep Running Logic ---
    key = cv2.waitKey(1)
    # Check if 'q' is pressed OR if the window's "X" button was clicked
    if key & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Traffic Sign Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")

