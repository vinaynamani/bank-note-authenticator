from flask import Flask,render_template,request
import pickle
import numpy as np

filename = 'rfclassifier.pkl'
model = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/verify',methods=['POST'])
def verify():
    if request.method == 'POST':
        variance = float(request.form['variance'])
        skewness = float(request.form['skewness'])
        curtosis = float(request.form['curtosis'])
        entropy = float(request.form['entropy'])

        data = np.array([[variance,skewness,curtosis,entropy]])
        my_prediction = model.predict(data)

        return render_template('result.html',prediction = my_prediction)

if __name__ =='__main__':
    app.run(debug=True)