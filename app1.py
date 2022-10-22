from flask import Flask, render_template,url_for,request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('diabetes.pkl','rb'))
@app.route("/")
def home():
    return render_template('ind1.html') 


@app.route("/predict", methods=["POST"])
def predict():
    float_features =[float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = model.predict(features)
    if pred==1:
        strr = "You have Diabetes :("
    else:
        strr = "You are healthy :)"
    
    return render_template("ind1.html", text_prediction ="Your Model Output is {} and {}".format(pred,strr)) 
if __name__=='__main__':
    app.run()