#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
          
        int_features = [int(X) for X in request.form.values()]
        str_features = [str(X) for X in request.form.values()]
        
        final_features = [np.array(int_features),(str_features)]
        
        prediction = model.predict(final_features)
        
        output = round(prediction[1], 2)
        return render_template('index.html',prediction_text='The Air Quality Should be {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




