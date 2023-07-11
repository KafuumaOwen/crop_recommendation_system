from flask import Flask,render_template,request
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/predict', methods=['POST'])  
def home():
    temp = request.form['a']
    hum = request.form['b']
    ph = request.form['c']
    rain = request.form['d']
    pred = model.predict([[temp,hum,ph,rain]])
    return render_template('result.html', data = pred)

@app.route('/about')
def about():
    return render_template ('about.html')



if __name__ == '__main__':
    app.run(debug=True)