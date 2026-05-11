# Traffic Sign Classification & Adversarial Attacks

This project demonstrates the complete development of a **Convolutional Neural Network (CNN)** for traffic sign recognition and explores the critical field of **AI Security** through Black-Box and White-Box adversarial attacks.

The implementation is divided into multiple phases, moving from standard model training to security auditing, highlighting how high-accuracy models can be deceptively fragile when faced with physical or mathematical perturbations.

---

# Installations

Main required packages for training, inference, and security testing:

```bash
pip install matplotlib
pip install opencv-contrib-python
pip install tensorflow keras
pip install pandas
pip install scikit-learn
```

---

# Repository Structure

- **codes/** Contains all Python implementation files: `train.py` (model training), `predict.py` (live inference), and `attack_fgsm.py` (white-box attack generation).

- **files/** Contains required input and weights files: `labels.csv` (mapping class IDs to names), `model_trained.h5` (the saved model), and `30img.png` (input for attack generation).

- **images/** Contains screenshots and output visualizations used in this documentation to demonstrate model performance and successful adversarial attacks.

- **README.md** Project documentation file containing installation steps, implementation details, and security analysis.

---

# Development Phases

### 1️⃣ Model Training (CNN Architecture)
**Script:** `codes/train.py` | **Dataset:** `myData/` (Folders 0-42)

The foundation of the project is a deep learning model trained on the GTSRB dataset to recognize 43 different traffic signs.

* **Preprocessing:** Images are grayscaled and Histogram Equalized to standardize lighting conditions before being normalized (scaled between 0 and 1).
* **Architecture:** A multi-layer CNN featuring Convolutional layers for feature extraction, Max Pooling for spatial reduction, and Dropout layers to prevent overfitting.

> **Output Demonstration:**
> <br><img src="images/loss_accuracy_graph.png" alt="Train and Test Loss Graph" width="320">

---

### 2️⃣ Live Inference & Baseline Prediction
**Script:** `codes/predict.py` | **Input:** Live Webcam / Sample Image

This stage establishes the "Eyes" of the system, using the webcam to detect signs in real-time and provide class labels and confidence scores.

* **Real-time Processing:** Captures frames, applies preprocessing, and reshapes data to `(1, 32, 32, 1)` for model compatibility.
* **Baseline Result:** A clear image of a **30 km/h** sign is correctly detected with high confidence (approx. 99%).

> **Output Demonstration:**
> <br><img src="images/prediction_normal_30.png" alt="Normal 30km Prediction" width="320">

---

### 3️⃣ Black-Box Attack (Physical Interference)
**Input:** Manually altered 30 km/h sign images.

This phase simulates real-world environmental "attacks" like vandalism or extreme lighting. It demonstrates that CNNs prioritize local pixel patterns over global context.

* **Case A (Center of '3'):** A small dash in the middle of the '3' dropped accuracy to **85%**.
* **Case B (Red Border):** A black line breaking the red outer circle caused the model to fail, misidentifying it as **20 km/h**.
* **Case C (Top of '3'):** A mark where the number meets the white background also flipped the prediction to **20 km/h**.

> **Attack Results:**

#### Case 0: Normal Image (No Tampering)
* **Input:** <br><img src="images/30_original.png" alt="Original Image" width="320">
* **Prediction:** <br><img src="images/prediction_normal.png" alt="Normal Prediction" width="320">

#### Case 1: The Center of the '3'
* **Input:** <br><img src="images/black_line_middle_original.png" alt="Black Line Middle 3 Original" width="320">
* **Prediction:** <br><img src="images/black_line_middle_prediction.png" alt="Black Line Middle 3 Prediction" width="320">

#### Case 2: The Red Border
* **Input:** <br><img src="images/black_line_red_circle_original.png" alt="Black Line Red Circle Original" width="320">
* **Prediction:** <br><img src="images/black_line_red_circle_prediction.png" alt="Black Line Red Circle Prediction" width="320">

#### Case 3: The Top "Vulnerability" Spot
* **Input:** <br><img src="images/black_line_top_3_original.png" alt="Black Line Top 3 Original" width="320">
* **Prediction:** <br><img src="images/black_line_top_3_prediction.png" alt="Black Line Top 3 Prediction" width="320">

---

### 4️⃣ White-Box Attack (FGSM)
**Script:** `codes/attack_fgsm.py` | **Input:** `model_trained.h5`

The final phase uses the **Fast Gradient Sign Method (FGSM)**. This is a mathematical attack where we use the model's own gradients to create "invisible" noise.

* **Gradient Calculation:** The script looks inside the model to see which pixels, if changed, will most increase the "loss" (confusion).
* **Epsilon ($\epsilon$):** A small multiplier (e.g., 0.04) ensures the noise is faint enough for the human eye to ignore, but strong enough to flip the AI's logic.
* **The Result:** The original **30 km/h** prediction flips to **120 km/h**.

> **FGSM Results:**

#### Step A: The Baseline (Clean Input)
* **Input Image:** <br><img src="images/fgsm_original_image.png" alt="FGSM Original Image" width="320">
* **Prediction:** <br><img src="images/fgsm_original_prediction.png" alt="FGSM Original Prediction" width="320">

#### Step B: The Attack (Adversarial Spoof)
* **Generated Spoof:** <br><img src="images/fgsm_spoofed_image.png" alt="FGSM Spoofed Image" width="320">
* **Prediction:** <br><img src="images/fgsm_spoof_prediction.png" alt="FGSM Spoof Prediction" width="320">

---

# Summary of Pipeline Logic

1.  **Capture/Load:** Capture raw frames or load static images.
2.  **Pre-process:** Grayscale, Equalize, and Normalize.
3.  **Classify:** Run the CNN model to get class ID and probability.
4.  **Audit:** Perform Black-Box/White-Box attacks to test robustness.
5.  **Analyze:** Evaluate the gap between human perception and AI logic.

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Scikit-Learn
- Matplotlib

---

# Applications & Security Insights

-   **Autonomous Vehicle Safety:** Highlights why systems cannot rely solely on vision and require multi-sensor fusion (LiDAR/Radar).
-   **Adversarial Robustness:** Serves as a baseline for implementing "Adversarial Training" to harden models against attacks.
-   **AI Auditing:** Demonstrates how to test for edge cases where lighting and shadows might cause fatal system failures.
