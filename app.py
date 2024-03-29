from flask import Flask, render_template, request
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
app = Flask(__name__,static_folder='static')
loaded_label_encoder = joblib.load('lebelencoder.pkl')
@app.route('/')
def home():
    return render_template('page.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        random_sample = request.files['file']
        data, sample_rate = librosa.load(random_sample)
        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=64)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        model = load_model('my_model.h5', compile=False)
        x_predict = model.predict(mfccs_scaled_features)
        predicted_label = np.argmax(x_predict, axis=1)
        prediction_class = loaded_label_encoder.inverse_transform(predicted_label)


        return render_template('page.html', prediction_text='Predicted class: {}'.format(prediction_class))

if __name__ == '__main__':
    app.run(debug=True)
