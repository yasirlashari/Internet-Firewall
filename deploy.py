from flask import Flask, render_template, request
import pickle
import numpy as np



app = Flask(__name__)
#load the model
model = pickle.load(open('savedmodel.sav', 'rb'))





@app.route('/')
def home():
    result=' '
    return render_template('home.html',**locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data1=float(request.form['a'])
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    data5=float(request.form['e'])
    data6=float(request.form['f'])
    data7=float(request.form['g'])
    data8=float(request.form['h'])
    data9=float(request.form['i'])
    data0=float(request.form['j'])
    data11=float(request.form['k'])
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data0,data11]])
    arr=arr.reshape(1,-1)
    result= model.predict(arr)
    return render_template('home.html', **locals())

if __name__ =="__main__":
    app.run(debug=True)