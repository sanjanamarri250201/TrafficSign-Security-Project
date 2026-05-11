import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# 1. Load your model
model = load_model("model_trained.h5")

def generate_adversarial_pattern(input_image, input_label):
    input_image = tf.cast(input_image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.categorical_crossentropy(input_label, prediction)
    
    # Get the gradients of the loss w.r.t the pixels
    gradient = tape.gradient(loss, input_image)
    # Get the 'sign' of the gradients
    signed_grad = tf.sign(gradient)
    return signed_grad

# 2. Prepare a clean 'Speed Limit 30' (Class 1) image
# Select an image from your folder. I'll use a generic path.
img_path = "30img.png" 
img = cv2.imread(img_path)
if img is None:
    print("Error: Could not find image. Check your path!")
    exit()

img_resized = cv2.resize(img, (32, 32))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) / 255.0
img_tensor = img_gray.reshape(1, 32, 32, 1)

# 3. Create the target label (Class 1)
label = tf.one_hot(1, 43)
label = tf.reshape(label, (1, 43))

# 4. Generate Attack
# epsilon = 0.01 is basically invisible to humans. 
# epsilon = 0.05 is grainy but still looks like a 30.
epsilon = 0.04 
pattern = generate_adversarial_pattern(img_tensor, label)
adv_x = img_tensor + epsilon * pattern
adv_x = tf.clip_by_value(adv_x, 0, 1)

# 5. Predictions
orig_preds = model.predict(img_tensor)
adv_preds = model.predict(adv_x)

print(f"Original Prediction: Class {np.argmax(orig_preds)} ({np.max(orig_preds)*100:.2f}%)")
print(f"Adversarial Prediction: Class {np.argmax(adv_preds)} ({np.max(adv_preds)*100:.2f}%)")

# 6. Save the Result to show your webcam
adv_img_to_save = (adv_x.numpy().reshape(32, 32) * 255).astype(np.uint8)
cv2.imwrite("fgsm_attack.png", adv_img_to_save)
print("Saved as fgsm_attack.png")
