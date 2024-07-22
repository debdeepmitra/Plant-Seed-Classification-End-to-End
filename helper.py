import numpy as np
import cv2

def load_and_preprocess_image(image, target_size=(120, 120)):
    # Load the image
    img = image
    print(img)
    # Resize the image
    img = cv2.resize(img, target_size)
    # Normalize the image
    img = img / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, image, class_names):
    image_array = np.array(image)
    img = load_and_preprocess_image(image_array)
    # Make prediction
    predictions = model.predict(img)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    # Get the class name
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, predictions[0]