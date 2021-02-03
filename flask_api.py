from flask import Flask,request
import pickle
import pandas as pd
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")      # saved model (pickle file)
classifier=pickle.load(pickle_in)

@app.route('/')                              # Starting Page
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])       # Predict function with Get Method
def predict_note_authentication():           # Input for function
    """Let's Authenticate the Banks Note
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    ## Input for Predict function(model)    @ var=reqest.args.get("var")
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Hello The answer is"+str(prediction)+" "+"With Confidence of:" + "88%"

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))

if __name__=='__main__':
    app.run()

