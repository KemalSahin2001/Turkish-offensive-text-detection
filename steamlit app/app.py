import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import streamlit as st

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

def load_model(model_path, num_labels=2):
    model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load your models
model1 = load_model("../models/BERTModels/model1_v1.pth")
model2 = load_model("../models/BERTModels/model2_v1.pth", num_labels=4)

def classify_text(text, model, tokenizer, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze()
    predictions = (probabilities > threshold).int()
    return predictions, probabilities

# Streamlit UI
st.title('Turkish Text Classification App 🌟')
st.markdown("""
This app classifies Turkish texts into categories such as not offensive, sexist, racist, profanity, or insult. Adjust the slider to set the sensitivity of the classification.
""")

user_input = st.text_area("📝 Enter text to classify:", height=150, placeholder="Type here...")
threshold = st.slider("⚙️ Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

col1, col2 = st.columns(2)
with col1:
    if st.button('🔍 Classify Text', help='Click to classify the text'):
        if user_input:
            offensive_prediction, _ = classify_text(user_input, model1, tokenizer, threshold)
            is_offensive = offensive_prediction.argmax().item() == 0  # '0' is the index for "offensive"
            if is_offensive:
                type_predictions, type_probabilities = classify_text(user_input, model2, tokenizer, threshold)
                labels = ['sexist', 'racist', 'profanity', 'insult']
                results = [labels[i] for i, prediction in enumerate(type_predictions) if prediction == 1]
                results_text = ', '.join(results) if results else 'No specific category'
                st.markdown(f"**Classification Results:** The text is classified as: **{results_text}**")
                st.json({label: f"{prob:.4f}" for label, prob in zip(labels, type_probabilities.numpy())})
            else:
                st.success("The text is not offensive.")
        else:
            st.error("Please enter some text to classify.")
with col2:
    st.info("""
    ### 🤝 Contributions
    To contribute to this project, please email us or submit a pull request on GitHub.

    ### 📧 Contact Information
    - **Email:** kmlshnbusiness@gmail.com 
    - **Email:** burakkurt015@gmail.com.com 
    - **GitHub:** [Visit our repository](https://github.com/KemalSahin2001/Turkish-offensive-text-detection)
    """)

# Enhanced Styling
st.markdown("""
<style>
    body {
        color: #fff; /* Keeps text white */
        background-color: #121212; /* Darker background for better contrast */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stTextInput, .stTextArea {
        background-color: #333333; /* Darker input fields for better visibility */
        color: #fff; /* White text in input fields */
        border: 2px solid #555555; /* Subtle borders for inputs */
        border-radius: 10px; /* Rounded corners for inputs */
    }
    .stButton > button {
        background-color: #ff6347; /* Vibrant button color */
        color: white;
        border-radius: 10px;
        font-size: 16px;
        height: 2.5em;
        width: 100%;
    }
    .stSlider {
        color: #ff6347; /* Slider color to match button */
    }
    .stMarkdown {
        background-color: #1e1e1e; /* Slightly different background for markdown areas */
        border-radius: 10px;
        padding: 10px; /* Padding to prevent text from touching the edges */
    }
</style>
""", unsafe_allow_html=True)

