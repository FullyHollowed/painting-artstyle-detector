import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('model/artstyle-detector-model.h5')

# Labels
class_names = [
    'abstract-expressionism', 'action-painting', 'analytical-cubism',
    'art-nouveau-modern', 'baroque', 'color-field-painting',
    'contemporary-realism', 'cubism', 'early-renaissance', 'expressionism',
    'fauvism', 'high-renaissance', 'impressionism', 'mannerism-late-renaissance',
    'minimalism', 'na-ve-art-primitivism', 'new-realism', 'northern-renaissance',
    'pointillism', 'pop-art', 'post-impressionism', 'realism', 'rococo',
    'romanticism', 'symbolism', 'synthetic-cubism', 'ukiyo-e'
]

# Preprocessing
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_artstyle(img: Image.Image) -> str:
    try:
        arr = preprocess(img)
        preds = model.predict(arr)
        top = np.argmax(preds[0])
        label = class_names[top]
        confidence = float(preds[0][top]) * 100
        return f"{label} ({confidence:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=predict_artstyle,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="🎨 Art Style Detector",
    description="Upload a painting to classify its art style."
)

iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
