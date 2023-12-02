import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2


num_classes = 10
input_shape = (28, 28, 1)
batch_size = 32
image_size = 32  # We'll resize input images to this size
patch_size = 16 # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4
mlp_head_units = [256, 256]  # Size of the dense layers of the final classifier

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Define the Patches class
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Define the PatchEncoder class
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Load the pre-trained VIT model with custom_objects
custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder}
model = tf.keras.models.load_model('model_vit.h5', custom_objects=custom_objects, compile=False)
class_names = ["Nol", "Satu", "Dua", "Tiga", "Empat", "Lima", "Enam", "Tujuh", "Delapan", "Sembilan"]

# Fungsi untuk melakukan prediksi pada gambar yang diunggah
def predict_image(model, img):
    img_array = np.array(img)
    
    # Pastikan gambar berupa grayscale dengan 1 channel
    if len(img_array.shape) > 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Resize gambar ke ukuran yang diharapkan oleh model (28, 28)
    img_array = cv2.resize(img_array, (28, 28))
    
    img_array = img_array / 255.0  # Normalisasi seperti pada data pelatihan
    img_array = np.expand_dims(img_array, axis=-1)  # Tambahkan dimensi channel

    # Lakukan prediksi menggunakan model yang sudah dibuat
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    predicted_label = np.argmax(predictions[0])
    predicted_class = class_names[predicted_label]
    return predicted_class


# Tampilan Streamlit
st.title("MNIST Digit Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
st.write("Link to testing file : https://drive.google.com/drive/folders/1arUn6k9Kt-tbjoRL7Ew1pS1-SyVi8H75?usp=sharing")
if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Lakukan prediksi saat tombol dipencet
    if st.button("Predict"):
        # Lakukan prediksi menggunakan model yang sudah dibuat
        prediction = predict_image(model, image)
        st.success(f"Predicted Label: {prediction}")