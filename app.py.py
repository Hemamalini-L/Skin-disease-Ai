from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

app = Flask(__name__)

# =========================
# 🔴 BUILD MODEL (SAME AS TRAINING)
# =========================
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = Flatten()(base.output)
out = Dense(6, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

# =========================
# 🔴 LOAD TRAINED WEIGHTS
# =========================
model.load_weights('model.weights.h5')

# =========================
# 🔴 CLASS LABELS
# =========================
classes = ['akiec','bcc','bkl','df','mel','nv']

# =========================
# 🔴 DISEASE DETAILS (REAL-LIKE INFO)
# =========================
disease_info = {
    'akiec': {
        'cause': 'Sun exposure and UV radiation damage',
        'symptoms': 'Rough, scaly patches on skin',
        'spread': 'Non-contagious',
        'precaution': 'Use sunscreen, avoid prolonged sun exposure'
    },
    'bcc': {
        'cause': 'DNA damage due to sunlight',
        'symptoms': 'Pearly or waxy bumps',
        'spread': 'Non-contagious',
        'precaution': 'Avoid UV rays, consult dermatologist early'
    },
    'bkl': {
        'cause': 'Benign keratin growth',
        'symptoms': 'Brown or black spots',
        'spread': 'Non-contagious',
        'precaution': 'Regular monitoring'
    },
    'df': {
        'cause': 'Minor skin injury or insect bite',
        'symptoms': 'Firm small lumps',
        'spread': 'Non-contagious',
        'precaution': 'Generally harmless, consult if painful'
    },
    'mel': {
        'cause': 'Melanocyte mutation (skin cancer)',
        'symptoms': 'Irregular dark patches, asymmetry',
        'spread': 'Can spread if untreated',
        'precaution': 'Early detection, avoid sun exposure'
    },
    'nv': {
        'cause': 'Pigmented mole formation',
        'symptoms': 'Small dark spots (moles)',
        'spread': 'Non-contagious',
        'precaution': 'Observe changes in size/color'
    }
}

# =========================
# 🔴 IMAGE PREPROCESSING
# =========================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# 🔴 ROUTE
# =========================
@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    confidence = None
    image = None
    details = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            img = Image.open(file).convert('RGB')

            pred = model.predict(preprocess(img))
            index_pred = np.argmax(pred)

            prediction = classes[index_pred]
            confidence = round(float(np.max(pred)) * 100, 2)

            details = disease_info[prediction]

            # convert image to base64 to display
            file.seek(0)
            image = base64.b64encode(file.read()).decode('utf-8')
            image = f"data:image/jpeg;base64,{image}"

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           details=details,
                           image=image)

# =========================
# 🔴 RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)