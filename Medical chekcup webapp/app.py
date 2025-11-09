# app.py
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PIL.Image
import google.generativeai as genai
from datetime import datetime
import re
import markdown
from html import escape

app = Flask(__name__)
app.secret_key = 'abc@123'  # Change this to a secure secret key

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
# genai.configure(api_key="Your_Api_Key")  # Replace with your actual API key
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

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
    """Analyze the uploaded images using Gemini API"""
    try:
        # Open images
        eye_image = PIL.Image.open(eye_image_path)
        oral_image = PIL.Image.open(oral_image_path)
        
        # Define prompt for analysis
        prompt = """
        Please analyze these medical images:
        1. First image is of the eye
        2. Second image is of the oral cavity
        
        Provide a detailed medical analysis covering:
        - Any visible abnormalities or concerns
        - Potential conditions or symptoms indicated
        - Recommendations for follow-up if necessary
        - Overall health status assessment
        
        Format the response with clear sections using markdown:
        ## Eye Analysis
        ### Abnormalities/Concerns
        ### Potential Conditions
        ### Recommendations
        
        ## Oral Cavity Analysis
        ### Abnormalities/Concerns
        ### Potential Conditions
        ### Recommendations
        
        ## Overall Assessment
        """
        
        # Generate analysis
        response = model.generate_content([prompt, eye_image, oral_image])
        return response.text
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


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and generate analysis"""
    # Check if both files are provided
    if 'eye_image' not in request.files or 'oral_image' not in request.files:
        flash('Both eye and oral cavity images are required')
        return redirect(url_for('index'))
    
    eye_image = request.files['eye_image']
    oral_image = request.files['oral_image']
    
    # Check if files are selected
    if eye_image.filename == '' or oral_image.filename == '':
        flash('No files selected')
        return redirect(url_for('index'))
    
    # Validate and process files
    if eye_image and oral_image and allowed_file(eye_image.filename) and allowed_file(oral_image.filename):
        try:
            # Generate unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save eye image
            eye_filename = f'eye_{timestamp}_{secure_filename(eye_image.filename)}'
            eye_path = os.path.join(app.config['UPLOAD_FOLDER'], eye_filename)
            eye_image.save(eye_path)
            
            # Save oral image
            oral_filename = f'oral_{timestamp}_{secure_filename(oral_image.filename)}'
            oral_path = os.path.join(app.config['UPLOAD_FOLDER'], oral_filename)
            oral_image.save(oral_path)
            
            # Generate analysis
            analysis = analyze_images(eye_path, oral_path)
            
            # Parse the analysis
            parsed_analysis = parse_analysis(analysis)
            
            # Render results
            return render_template('result.html',
                                eye_image=eye_filename,
                                oral_image=oral_filename,
                                datetime = datetime,
                                **parsed_analysis)
                                
        except Exception as e:
            flash(f'Error processing images: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload only images (png, jpg, jpeg)')
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    flash('An error occurred processing your request. Please try again.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)