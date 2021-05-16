from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model_obj = open('model.pkl', 'rb')
model = pickle.load(model_obj)

def convert_to_str(word):
    word_dict = {0 : 'Iris-setosa',
                1 : 'Iris-versicolor',
                2 : 'Iris-virginica'
    }
    return word_dict[word]

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/', methods=["POST"])
def get_data():
	
	features = [float(x) for x in request.form.values()]
	prediction = model.predict([np.array(features)])
	final_output = convert_to_str(prediction[0])
	image_name = str(final_output) + '.png'
	print(image_name)
	return render_template("home.html", result = 'Flower should be {}'.format(final_output), image = image_name)

if __name__ == "__main__":
	app.run(debug=True)