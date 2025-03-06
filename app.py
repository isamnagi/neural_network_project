from flask import Flask, request, jsonify, render_template
import neural_network

app = Flask(__name__)

nn = neural_network.NeuralNetwork(3, 4, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    output = nn.forward(data)
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)