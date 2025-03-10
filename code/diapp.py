import string
import pickle
import numpy as np
from flask import Flask, render_template, request,url_for


Diabitic = Flask(__name__)

# Load the trained model
model_filename = "/home/ubuntu/Diabetic Prediction.pickle"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the saved scaler
scaler_filename = "/home/ubuntu/StandardScaler.pkl"
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

def prediction(lst):
    """Transform user input and make prediction."""
    lst = np.array(lst).reshape(1, -1)  # Convert list to numpy array and reshape
    scaled_input = scaler.transform(lst)  # Scale the input
    pred_value = model.predict(scaled_input)  # Make prediction
    return pred_value[0]


#def prediction(lst):
  #  filename = "D:\Data Science\ML\sample pro\classification\diabetes\Diabetic Prediction.pickle"
   # with open(filename, 'rb') as file:
      #  model = pickle.load(file)
     #   pred_value = model.predict([lst])
      #  return pred_value

@Diabitic.route("/",methods=["Post","Get"])

def index():

    pred_value = None

    if request.method == 'POST':

        form_data = request.form

        num_preg = form_data["num_preg"]
        glucose_conc = form_data["glucose_conc"]
        diastolic_bp = form_data["diastolic_bp"]
        thickness = form_data["thickness"]
        insulin = form_data["insulin"]
        bmi = form_data["bmi"]
        diab_pred = form_data["diab_pred"]
        age = form_data["age"]
        
        input_list = []

        input_list.append(int(num_preg))
        input_list.append(int(glucose_conc))
        input_list.append(int(diastolic_bp))
        input_list.append(int(thickness))
        input_list.append(int(insulin))
        input_list.append(float(bmi))
        input_list.append(float(diab_pred))
        input_list.append(int(age ))

        print(input_list)
       

        pred_value = prediction(input_list)

    return render_template("dibitic.html",prediction= pred_value)


if __name__ =='__main__':
    Diabitic.run('0.0.0.0',port=8080)