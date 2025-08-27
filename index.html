
#%% Import necessary libraries
import torch  # For model execution on CPU/GPU
import streamlit as st  # For creating the web app UI
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # For loading BERT and tokenizer
import pandas as pd  # For data manipulation
import numpy as np  # For handling probabilities and arrays
import plotly.express as px  # For visualizing prediction confidence

#%% Set Streamlit page configuration
st.set_page_config(page_title="MindScope", layout="wide")  # Sets the title and layout of the web app

#%% Load fine-tuned BERT model and tokenizer from local path
MODEL_PATH = r"C:/Users/PRATHIKSHA/Downloads/colab_files/final_model_bert_full"  # Path where my fine-tuned model is saved

@st.cache_resource  # Cache the model and tokenizer so they don’t reload every time
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # Load the saved tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)  # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    model.to(device).eval()  # Move model to selected device and set to eval mode
    id2label = {int(k): v for k, v in model.config.id2label.items()}  # Dictionary to map label ids to label names
    return tokenizer, model, device, id2label

tokenizer, model, device, id2label = load_model()  # Call the function and unpack the results

#%% Define a function to run predictions using the BERT model
def predict_with_model(texts, max_length=256):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)  # Tokenize and prepare the input
    with torch.no_grad():  # Disable gradient computation
        logits = model(**enc).logits  # Run input through the model to get logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()  # Convert logits to probabilities
    ids = probs.argmax(axis=1)  # Get the index of the most probable label
    labels = [id2label[i] for i in ids]  # Convert label ids to label names
    return labels, probs  # Return predicted labels and probabilities

#%% Custom CSS styles for UI aesthetics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Playfair+Display&display=swap');

    .stApp {
        background: linear-gradient(155deg, #6B6BBF 23%, #F5DFDF 95%);
        background-attachment: fixed;
        animation: bg 20s ease infinite;
        font-family: 'Playfair Display', serif;
    }

    .stTextArea [data-testid="stTextArea"] textarea {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Darker text color */
        border-radius: 5px; /* Slightly rounded corners */
        padding: 10px; /* Add some padding inside the textarea */
    }

    @keyframes bg {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .chat-bubble-user {
        background-color: #F4EBEB;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        font-family: 'Playfair Display', serif;
    }

    .chat-bubble-bot {
        background-color: #f5dfdf;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        font-family: 'Playfair Display', serif;
    }

    .chat-anxiety { background-color: #FFCCCB; }
    .chat-depression { background-color: #C1C8E4; }
    .chat-bipolar { background-color: #E1F5FE; }
    .chat-bpd { background-color: #FFE0B2; }
    .chat-schizophrenia { background-color: #EDE7F6; }
    .chat-autism { background-color: #E8F5E9; }
    .chat-mentalhealth { background-color: #FFF3E0; }

    .main-title {
        font-family: 'Great Vibes', cursive;
        font-size: 52px;
        font-weight: 500;
        margin-bottom: 10px;
        color: #333;
    }

    div.stButton > button:first-child {
        background-color: black;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)  # Injects the custom style block into the app

# Load logo and main image
LOGO = r"C:/Users/PRATHIKSHA/Downloads/Social Media Posts/logo1.png"
MAIN_IMG = r"C:/Users/PRATHIKSHA/Downloads/Social Media Posts/5.png"

#%% Setup session state to store user and bot messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize message history on first run

#%% Setup three-column layout for content
base = st.container()
left, right = base.columns(2, vertical_alignment="bottom")  # Define layout proportions
bottom = st.columns(1)[0]
base.divider() 

#%% LEFT COLUMN: App title and intro text
with left:
    st.logo(LOGO, size="large")
    st.image(LOGO, width=200)
    st.markdown("<div class='main-title'>MindScope</div>", unsafe_allow_html=True)  # Title
    subline = """Welcome to your mental health assistant.\nLet us know about your thoughts and feelings, and we will help you assess \nyour emotional well-being."""
    st.text(subline)
    st.title("Your _happiness_ is essential.\
    Your _self-care_ is a necessity.")

#%% RIGHT COLUMN: Inspirational quote
with right:
    st.image(MAIN_IMG, width=650)

#%% CENTER COLUMN: Input text and prediction display
with bottom:
    # st.subheader("Talk to us")
    st.markdown("<div class='main-title' style='text-align: center'>Talk to us</div>", unsafe_allow_html=True)  # Title
    st.markdown("<div style='margin-bottom: 20px;text-align: center'>We're here to listen. Tell us how you're feeling today...</div>", unsafe_allow_html=True)
    # st.write("We're here to listen. Tell us how you're feeling today...")  # Instruction text

    user_input = st.text_area("Your thoughts", placeholder="I feel like everything is too much lately...", label_visibility="collapsed")  # Input box

    if st.button("Analyze", use_container_width=True):  # When the button is clicked
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})  # Save user message
            st.markdown(f"<div class='chat-bubble-user'>User: {user_input}</div>", unsafe_allow_html=True)  # Display user message

            labels, probs = predict_with_model([user_input])  # Run prediction
            pred_label = labels[0]  # Get top predicted label
            confidence = probs[0].max()  # Get confidence score

            advice_map = {
                "anxiety": ("It sounds like you're experiencing anxiety. Try calming techniques like deep breathing or grounding exercises, and consider reaching out to a therapist.",  "chat-anxiety"),
                "depression": ("You might be showing signs of depression. Talking to a professional or sharing your feelings with someone you trust could make a difference.", "chat-depression"),
                "bipolar Disorder": ("Your post suggests experiences that may align with bipolar disorder. Keeping track of mood patterns and seeking medical advice can be helpful.", "chat-bipolar"),
                "bpd": ("Some of what you've written reflects challenges linked with borderline personality disorder. Reaching out to a professional or support group could provide guidance.", "chat-bpd"),
                "schizophrenia": ("Certain expressions in your post may relate to schizophrenia. It's important to consult with a healthcare professional for appropriate support and care.", "chat-schizophrenia"),
                "autism": ("Your post highlights themes that may reflect autism experiences. Connecting with supportive communities and seeking professional advice may be beneficial.", "chat-autism"),
                "mentalhealth": ("It seems like you're going through a difficult time. Remember that support is available, whether through friends, family, or professionals.", "chat-mentalhealth")
            }
            advice, css_class = advice_map.get(pred_label, ("Please consult a healthcare professional.", "chat-bubble-bot"))  # Get advice and style

            response_html = f"""
            <div class="chat-bubble-bot {css_class}">
                <b>Prediction:</b> {pred_label} (confidence {confidence:.2f})<br>
                {advice}
            </div>
            """  # Format bot response

            st.markdown(response_html, unsafe_allow_html=True)  # Show bot reply
            st.session_state.messages.append({"role": "assistant", "content": response_html})  # Save bot message
            st.session_state.messages.append({"role": "assistant", "content": response_html})  # Save the bot's formatted prediction/advice message into chat history

            st.progress(float(confidence))  # Show progress bar

            df = pd.DataFrame({"Condition": [id2label[i] for i in range(len(probs[0]))], "Probability": probs[0]})  # Prepare data for plot
            fig = px.bar(df, x="Condition", y="Probability", color="Condition", title="Prediction Confidence by Class")  # Create bar chart
            st.plotly_chart(fig, use_container_width=True)  # Display chart

        else:
            st.warning("Please share a little about how you're feeling.")  # Warn if text is empty

#%% Display chat history
with bottom:
    st.divider()
    if st.session_state.messages:
        st.subheader("Conversation History")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'>User: {msg['content']}</div>", unsafe_allow_html=True)  # Display user message
            else:
                st.markdown(msg["content"], unsafe_allow_html=True)  # Display bot message

