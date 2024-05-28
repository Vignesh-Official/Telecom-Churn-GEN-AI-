from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEndpoint
import os


app = Flask(__name__)

# Load the model from the pickle file
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

 #Set your OpenAI API key
sec_key="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"
os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
repo_id="mistralai/Mistral-7B-Instruct-v0.2"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
def get_telecom_offers():
    comment=llm.invoke("suggest best plans and Offers in telecom domain don't mention company name and end with agent will contact you")
    return comment



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from the request data
        features = [
            data.get('feature1'), data.get('feature2'), data.get('feature3'), 
            data.get('feature4'), data.get('feature5'), data.get('feature6'), 
            data.get('feature7'), data.get('feature8'), data.get('feature9'), 
            data.get('feature10'), data.get('feature11')
        ]
        
        # Ensure all features are present
        if None in features:
            return jsonify({'error': 'Missing feature values'}), 400
        
        # Convert features to a numpy array
        features_array = np.array([features], dtype=float)

        
        # Predict using the loaded model
        prediction = model.predict(features_array)
        
        # Convert prediction to a Python int
        prediction = int(prediction[0])
        
        # If the prediction is 1, suggest telecom offers
        if prediction == 1:
            offers = get_telecom_offers()

            #offers="\n\nWe'd be happy to help you find the best plans and offers in the telecom domain. Here are some suggestions based on your requirements:\n\n1. Postpaid Plans: These plans offer flexible billing cycles and come with various benefits such as unlimited calling, SMS, and data usage. You can choose from a range of plans based on your usage needs. Our agent will contact you with the best options.\n2. Prepaid Plans: If you prefer paying upfront for your mobile services, then prepaid plans are a good option. These plans offer various benefits such as daily, weekly, or monthly packs that cater to different usage needs. Our agent will contact you with the best options.\n3. Broadband Plans: If you're looking for a high-speed internet connection at home, then broadband plans are the way to go. These plans offer various speeds and data limits, along with additional benefits such as free calling and TV channels. Our agent will contact you with the best options."
            return jsonify({'prediction': prediction,'offers': offers})
        else:
            return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
