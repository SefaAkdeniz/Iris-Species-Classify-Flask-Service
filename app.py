import pandas as pd
from flask import Flask,jsonify,request
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['POST'])
def index():
    
    body = request.json
    SepalLengthCm = body["SepalLengthCm"]
    SepalWidthCm = body["SepalWidthCm"]
    PetalLengthCm  = body["PetalLengthCm"]
    PetalWidthCm  = body["PetalWidthCm"]
    
    try:     
        
        array=pd.DataFrame([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
            
        array = scaler.transform(array)    
        if model.predict(array)[0]==0:  
            return jsonify(result=1,message="Girilen uzunluklar çiçek türünün Iris Setosa olduğuna işaret.")
        elif model.predict(array)[0]==1:
            return jsonify(result=1,message="Girilen uzunluklar çiçek türünün Iris Versicolor olduğuna işaret.")
        else:
            return jsonify(result=1,message="Girilen uzunluklar çiçek türünün Iris Virginica olduğuna işaret.")
    
    except ValueError:
        return jsonify(result=0,message=str(ValueError))
           
if __name__ == '__main__':
    app.run(port=5000,debug=True)
