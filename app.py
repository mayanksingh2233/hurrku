from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
from numpy.core.numeric import outer

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    '''For rendering the result on HTML gui '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    output = model.predict(final_features)
    return render_template('index.html',prediction_text='Employee purchase{}'.format(output))

   




if __name__=='__main__':
    app.run(debug=False)