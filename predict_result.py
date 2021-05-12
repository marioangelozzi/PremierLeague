from flask import Flask, jsonify, request 
import numpy as np
import pandas as pd

# instancia o objeto do Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def landing_page():
	return "hello world"

@app.route('/html', methods=['GET'])
def landing_page_html():
	return '<html><h1 style="color:red">hello world<h1></html>'

@app.route('/array', methods=['GET'])
def array():
    response = np.random.randint(0,100, 10)
    return np.array_str(response)

@app.route('/ndarray', methods=['GET'])
def ndarray():
    response = np.random.randint(0,100, 10).reshape(5,-1)
    return np.array_str(response)

@app.route('/table', methods=['GET'])
def table():
    response = np.random.randint(0,100, 10).reshape(5,-1)
    response = pd.DataFrame(response, columns=['col1','col2'])
    return response.to_html()

if __name__ == '__main__':
    app.run(debug=True)
