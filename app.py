from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model1=pickle.load(open('model1.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("diabetes.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
   # print(int_features)
   # print(final)
    prediction=model1.predict_proba(final)*100
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(50):
        return render_template('diabetes.html',pred='You have a high chance of Diabetes. Please consult doctor.\nProbability of Diabetes is {}%'.format(output))
    else:
        return render_template('diabetes.html',pred='You are safe and healty!. Take care. \n Probability of Diabetes is {}%'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='127.0.0.1',port=12345, debug = True)