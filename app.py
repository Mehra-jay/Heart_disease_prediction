from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import joblib
import sklearn
import os
import pickle
import gzip
import pandas as pd
import cv2
import numpy as np
from f import get_prediction

app = Flask(__name__, template_folder=".", static_folder="E:\Testing")
model = joblib.load('Heart_Disease_Prediction.joblib')

modelD = joblib.load('Diabetes_Prediction.joblib')

modelL = joblib.load('LiverDisease.joblib')

modelBTC= tf.keras.models.load_model('effnet.h5')

modelpneumonia = tf.keras.models.load_model('pneumonia.h5')

@app.route('/Liver Disease')
def liver():
    return render_template("html/index.html")

@app.route('/Heart Disease')
def heart():
    return render_template("html/index2.html")

@app.route('/Diabetes')
def diabetes():
    return render_template("html/index3.html")

@app.route('/BTCMRI')
def btc():
    return render_template("html/index4.html")

@app.route('/pneumonia')
def pneumonia():
    return render_template("html/index5.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = request.form["age"]
    sex = request.form["sex"]
    trestbps = request.form["trestbps"]
    chol = request.form["chol"]
    oldpeak = request.form["oldpeak"]
    thalach = request.form["thalach"]
    fbs = request.form["fbs"]
    exang = request.form["exang"]
    slope = request.form["slope"]
    cp = request.form["cp"]
    thal = request.form["thal"]
    ca = request.form["ca"]
    restecg = request.form["restecg"]
    arr = np.array([[age, sex, cp, trestbps,
                     chol, fbs, restecg, thalach,
                     exang, oldpeak, slope, ca,
                     thal]])
    pred = model.predict(arr)
    if pred == 0:
        res_val = "NO HEART PROBLEM"
    else:
        res_val = "HEART PROBLEM"
    return render_template('html/index2.html', prediction_text='PATIENT HAS {}'.format(res_val))

@app.route("/predict1", methods=["POST"])
def predict1():
    print(request.form)
    pregnant = float(request.form["pregnant"])
    glucose = request.form["Glucose"]
    blood_pressure = request.form["Dia. BP"]
    skin_thickness = request.form["Skin Thickness"]
    insulin_level = request.form["Insulin Level"]
    bmi = request.form["BMI"]
    diabetes_pedigree = request.form["DPF"]
    age = request.form["AGE"]
    
    arr = np.array([[pregnant, glucose, blood_pressure, skin_thickness,
                     insulin_level, bmi, diabetes_pedigree, age]])
    print(arr)
    pred = modelD.predict(arr)
    
    if pred == 0:
        res_val = "NO DIABETES"
    else:
        res_val = "DIABETES DETECTED"
    
    return render_template('html/index3.html', prediction_text='PATIENT HAS {}'.format(res_val))

@app.route("/predict2", methods=["POST"])
def predict2():
    print(request.form)
    
    Age = request.form["Age"]
    Gender = int(request.form["Gender"])
    Total_Bilirubin = request.form["Total Bilirubin"]
    Direct_Bilirubin = request.form["Direct Bilirubin"]
    Alkaline_Phosphotase = request.form["Al. Phosphotase"]
    Alamine_Aminotransferase = request.form["A. Aminotransferase"]
    Aspartate_Aminotransferase = request.form["Aspartate Aminotransferase"]
    Total_Protiens = request.form["Total Protein"]
    Albumin = request.form["Albumin"]
    Albumin_and_Globulin_Ratio = request.form["Ratio"]

    arr = np.array([[Age,Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
                        Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, 
                        Albumin, Albumin_and_Globulin_Ratio]],dtype=float)
    print(arr)

    pred = modelL.predict(arr)

    if pred == 2:
        res_val = "NO LIVER DISEASE"
    elif pred == 1:
        res_val = "LIVER DISEASE DETECTED"
    else:
        res_val = "404 Not found"


    return render_template('html/index.html', prediction_text='PATIENT HAS {}'.format(res_val))


def preprocess_image(image):
    
    image = cv2.resize(image, (150, 150))
    
    image = image / 255.0
    
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict3", methods=["POST"])
def predict3():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('html/index4.html', prediction_text='No file uploaded')
        
        # Get the uploaded file
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('html/index4.html', prediction_text='No file selected')
        
        # Read the image file
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = modelBTC.predict(processed_image)
        # Assuming your model output is a probability distribution over classes
        predicted_class = np.argmax(prediction)
        
        # Map predicted class to human-readable label
        labels = ['glioma', 'notumor', 'meningioma', 'pituitary']
        predicted_label = get_prediction(file=file)
        print(predicted_label)
        return render_template('html/index4.html', prediction_text=f'Predicted tumor type: {predicted_label}')
    
def detect_pneumonia(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = modelpneumonia.predict(img_array)
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        return "Prediction: No Pneumonia"
    else:
        return "Prediction: Pneumonia Detected"

@app.route('/predict4', methods=['POST'])
def predict4():
    if 'file' not in request.files:
        return render_template('index5.html', prediction_text='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('index5.html', prediction_text='No file selected')

    if file:
        # Save the uploaded file to a temporary location
        filename = 'temp_image.jpg'
        file.save(filename)

        # Perform pneumonia detection
        prediction_result = detect_pneumonia(filename)

        # Delete the temporary file
        os.remove(filename)

        return render_template('html/index5.html', prediction_text=prediction_result)    

if __name__ == "__main__":
    app.run(debug=True,port=8080)
