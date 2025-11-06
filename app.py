from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

#Load model and data
try:
  model = joblib.load("/content/churn-customers.pk1")
  data = pd.read_csv("/content/P585 Churn.csv")
except Exception as e:
  print(f"Error in loading files: {e}")
  raise

data.columns = [c.strip().lower().replace(' ', '.') for c in data.columns]
data.replace(['Nan', 'NA', '', None, pd.NA], np.nan, inplace = True)
states = sorted(data['state'].dropna().unique())
area_codes = sorted(data['area.code'].dropna().unique())

@app.route('/')
def home():
  return render_template('index.html', states = states, area_codes = area_codes)

@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Collect user input
    input_data = {       
        'intl.mins': float(request.form['intl_mins']),
        'intl.calls': float(request.form['intl_calls']),
        'day.mins': float(request.form['day_mins']),     
        'day.charge': float(request.form['day_charge']),       
        'eve.charge': float(request.form['eve_charge']),       
      }

    input_df = pd.DataFrame([input_data])

    #predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    churn_label = 'Yes' if prediction == 1 else 'No'

    return render_template('result.html',churn=churn_label,probability=probability,prediction=f"{prediction:.2%}", inputs = input_data)
  except Exception as e:
    return render_template('result.html', error = f"Error : {str(e)}")
if __name__ == '__main__':
  app.run(host='0.0.0.0',port=5000, debug = True)
