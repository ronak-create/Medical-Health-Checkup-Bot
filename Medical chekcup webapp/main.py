# main.py
from fastapi.middleware.cors import CORSMiddleware
# main.py
from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import io, requests, torch
import os
from werkzeug.utils import secure_filename
import PIL.Image
# import google.generativeai as genai
from datetime import datetime
import re
import markdown
from html import escape
from openai import OpenAI
# Your api key here
# client = OpenAI(api_key="Your_Api_Key")

app = FastAPI()

# Allow React frontend to talk to backend
origins = [
    "http://localhost:3000",   # React dev
    "http://127.0.0.1:5500",   # Live server for HTML
    "http://localhost:5500"    # in case Live Server uses localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["healthbot"]

# ---------------- Models ----------------
# Conversational model (lightweight causal LM)
chatbot = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
chat_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Medical Q&A model (Flan-T5 / BioBART)
medical_chatbot = pipeline("text2text-generation", model="GanjinZero/biobart-base")

# Emotion detection
emotion_detector = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# ---------------- Helpers ----------------
def save_message(user_id, role, message):
    db.conversations.update_one(
        {"userId": user_id},
        {"$push": {"conversation": {
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow()
        }}},
        upsert=True
    )
    
# app content----------------------------------------
# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
# genai.configure(api_key="AIzaSyCO2fVDOLnSyLuMhC6wbeJf-_hXHqHJq0c")  # Replace with your actual API key
# model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_html(text):
    """Clean and format HTML content safely."""
    try:
        # Convert markdown to HTML
        html = markdown.markdown(text)

        # Remove potentially dangerous tags (e.g., <script>, <iframe>)
        html = re.sub(r'<(script|iframe|object|embed).*?>.*?</\1>', '', html, flags=re.DOTALL)

        # Remove potentially dangerous attributes (e.g., onclick, onload)
        html = re.sub(r'on\w+=".*?"', '', html)

        # Escape any remaining potentially dangerous content
        html = escape(html)

        return html
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return text  # Return raw text if cleaning fails
    
def analyze_images(eye_image_path, oral_image_path):
    try:
        prompt = """
        Please analyze these medical images:
        - First image is of the eye
        - Second image is of the oral cavity
        Provide structured medical analysis.
        """

        with open(eye_image_path, "rb") as f1, open(oral_image_path, "rb") as f2:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4o"
                messages=[
                    {"role": "system", "content": "You are a medical analysis assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": f1.read()},
                        {"type": "image", "image": f2.read()}
                    ]}
                ]
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def clean_text(text):
    # Remove markdown symbols
    cleaned = text.replace('#', '').replace('*', '').replace('**', '').strip()
    return cleaned

def parse_analysis(analysis_text):
    sections = {
        'eye_findings': '',
        'eye_conditions': '',
        'eye_recommendations': '',
        'oral_findings': '',
        'oral_conditions': '',
        'oral_recommendations': '',
        'overall_assessment': ''
    }
    print(analysis_text)
    try:
        # Split into major sections based on '##'
        eye_analysis = analysis_text.split("## Eye Analysis")[1]
        eye_findings = clean_text(eye_analysis.split("### Potential Conditions")[0])
        eye_conditions = clean_text(eye_analysis.split("### Potential Conditions")[1].split("### Recommendations")[0])
        eye_recommendations = clean_text(eye_analysis.split("### Recommendations")[1])

        # Store extracted data into sections dictionary
        sections['eye_findings'] = eye_findings
        sections['eye_conditions'] = eye_conditions
        sections['eye_recommendations'] = eye_recommendations

        # Extract Oral Cavity Analysis findings
        oral_analysis = analysis_text.split("## Oral Cavity Analysis")[1]
        oral_findings = clean_text(oral_analysis.split("### Potential Conditions")[0])
        oral_conditions = clean_text(oral_analysis.split("### Potential Conditions")[1].split("### Recommendations")[0])
        oral_recommendations = clean_text(oral_analysis.split("### Recommendations")[1])

        # Store extracted data into sections dictionary
        sections['oral_findings'] = oral_findings
        sections['oral_conditions'] = oral_conditions
        sections['oral_recommendations'] = oral_recommendations
        sections['overall_assessment'] = clean_html(analysis_text)
    
    except Exception as e:
        print(f"Error parsing analysis: {e}")
        sections['overall_assessment'] = clean_html(analysis_text)
    return sections


def extract_subsection(text, header):
    """Extract content for a given subsection header."""
    try:
        pattern = rf"{header}\s*(.*?)\s*(?=(###|$))"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return clean_html(match.group(1).strip())
        return ""
    except Exception as e:
        print(f"Error extracting subsection {header}: {e}")
        return ""
# ---------------------------
def get_context(user_id, limit=5):
    convo = db.conversations.find_one({"userId": user_id})
    if not convo:
        return []
    return convo["conversation"][-limit:]

def extract_intent(user_input):
    triggers = ["checkup", "doctor", "hospital", "appointment", "scan", "test"]
    if any(word in user_input.lower() for word in triggers):
        return "checkup_request"
    return "general"

def generate_chat_response(user_input: str):
    """
    Stateless single-turn chatbot response with cleanup
    """
    prompt = f"The following is a friendly conversation between a user and a helpful medical assistant.\nUser: {user_input}\nBot:"

    inputs = chat_tokenizer(prompt, return_tensors="pt")
    output = chatbot.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=chat_tokenizer.eos_token_id
    )
    reply = chat_tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt part
    if reply.startswith(prompt):
        reply = reply[len(prompt):]

    # Strip off any extra "User:" or "Bot:" that the model may generate
    reply = reply.split("User:")[0].strip()
    reply = reply.split("Bot:")[-1].strip()

    # Safety: avoid echo
    if reply.lower() == user_input.lower():
        reply = "I'm doing well! How are you feeling today?"

    return reply




def generate_medical_response(user_input):
    """
    BioBART for medical Q&A
    """
    response = medical_chatbot(user_input, max_length=128, num_return_sequences=1)
    return response[0]["generated_text"]

# ---------------- Routes ----------------
@app.post("/chat")
async def chat(userId: str = Form(...), message: str = Form(...)):
    save_message(userId, "user", message)
    # history = get_context(userId)

    intent = extract_intent(message)

    if intent == "checkup_request":
        bot_reply = "Sure! I can help you with a basic checkup. Please upload your **eye** and **throat** images."
    elif any(word in message.lower() for word in ["symptom", "cough", "diabetes", "fever"]):
        # Use medical model for questions
        bot_reply = generate_medical_response(message)
    else:
        bot_reply = generate_chat_response(message)

    save_message(userId, "bot", bot_reply)
    return {"reply": bot_reply, "intent": intent}

@app.post("/emotion")
async def emotion(userId: str = Form(...), face: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await face.read()))
    result = emotion_detector(image)
    mood = result[0]["label"]

    db.conversations.update_one(
        {"userId": userId},
        {"$push": {"moodHistory": {
            "mood": mood,
            "confidence": result[0]["score"],
            "timestamp": datetime.utcnow()
        }}},
        upsert=True
    )
    return {"mood": mood}


# ---------------- Route ----------------
@app.post("/gemini-checkup")
async def gemini_checkup(userId: str = Form(...), eye: UploadFile = File(...), throat: UploadFile = File(...)):
    try:
        # Secure filenames
        eye_filename = secure_filename(eye.filename)
        throat_filename = secure_filename(throat.filename)
        print(eye.filename, throat.filename)
        # File paths
        eye_path = os.path.join(UPLOAD_FOLDER, eye_filename)
        throat_path = os.path.join(UPLOAD_FOLDER, throat_filename)

        # Save uploaded files
        with open(eye_path, "wb") as f:
            f.write(await eye.read())
        with open(throat_path, "wb") as f:
            f.write(await throat.read())

        # Analyze with Gemini (using file paths)
        analysis = analyze_images(eye_path, throat_path)
        print(analysis)

        # Parse results
        parsed = parse_analysis(analysis)
        print(parsed)

        # Save into MongoDB
        db.conversations.update_one(
            {"userId": userId},
            {"$push": {"checkupResults": {
                "date": datetime.utcnow(),
                "eye_findings": parsed['eye_findings'],
                "eye_conditions": parsed['eye_conditions'],
                "eye_recommendations": parsed['eye_recommendations'],
                "oral_findings": parsed['oral_findings'],
                "oral_conditions": parsed['oral_conditions'],
                "oral_recommendations": parsed['oral_recommendations'],
                "overall_assessment": parsed['overall_assessment']
            }}},
            upsert=True
        )

        return {"success": True, "analysis": parsed}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)