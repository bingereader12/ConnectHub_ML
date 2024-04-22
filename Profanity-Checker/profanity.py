from flask import Flask, request, jsonify
from profanity_check import predict_prob

app = Flask(__name__)

@app.route('/check-profanity', methods=['POST'])
def check_profanity():
    if 'text' not in request.json:
        return jsonify({'error': 'Text input missing in request'}), 400
    
    text = request.json['text']
    if not text.strip():
        return jsonify({'error': 'Text input is empty'}), 400

    profanity_probability = predict_prob([text])[0]
    print(profanity_probability)

    if profanity_probability > 0.2:
        return jsonify({'error': 'Profanity detected', 'profanity_probability': profanity_probability}), 401
    else:
        return jsonify({'profanity_probability': profanity_probability}), 200

if __name__ == '__main__':
    app.run(debug=True)
