# Install required dependencies
!pip install pyngrok flask flask_cors pdfplumber transformers

# Import necessary modules
from flask import Flask, request, jsonify
from pyngrok import ngrok
from flask_cors import CORS
import pdfplumber
from transformers import pipeline

# Set ngrok authtoken (replace with your actual token)
ngrok.set_auth_token("2tJt24KUrcuYx1L2yWOJbQZSrUS_rCyJQNK3X3mx3q8nhi21")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Hugging Face zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

@app.route("/grade", methods=["POST"])
def grade():
    try:
        student_pdf = request.files["student_pdf"]
        rubric_pdf = request.files["rubric_pdf"]
        grading_scale = request.form["grading_scale"]  # Get grading scale from request

        student_text = extract_text_from_pdf(student_pdf)
        rubric_text = extract_text_from_pdf(rubric_pdf)

        # Use Hugging Face model to classify student response based on rubric
        result = classifier(student_text, candidate_labels=[rubric_text])

        # Get the AI confidence score
        confidence = result["scores"][0]  # First (and only) label's confidence score

        # Convert confidence score to the selected grading scale
        if grading_scale == "R-8":
            if confidence > 0.85:
                grade = "8"
            elif confidence > 0.75:
                grade = "7"
            elif confidence > 0.65:
                grade = "6"
            elif confidence > 0.55:
                grade = "5"
            elif confidence > 0.45:
                grade = "4"
            elif confidence > 0.35:
                grade = "3"
            elif confidence > 0.25:
                grade = "2"
            elif confidence > 0.15:
                grade = "1"
            else:
                grade = "R"

        elif grading_scale == "R-4+":
            if confidence > 0.85:
                grade = "4+"
            elif confidence > 0.75:
                grade = "4"
            elif confidence > 0.65:
                grade = "3"
            elif confidence > 0.45:
                grade = "2"
            elif confidence > 0.25:
                grade = "1"
            else:
                grade = "R"

        elif grading_scale == "F-A+":
            if confidence > 0.85:
                grade = "A+"
            elif confidence > 0.75:
                grade = "A"
            elif confidence > 0.65:
                grade = "B"
            elif confidence > 0.45:
                grade = "C"
            elif confidence > 0.25:
                grade = "D"
            else:
                grade = "F"

        elif grading_scale == "0%-100%":
            grade = f"{int(confidence * 100)}%"

        else:
            grade = "Unknown scale"

        return jsonify({"grade": grade})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

# Set up ngrok to expose Flask to the web
public_url = ngrok.connect(5000)
print(f"Flask app is running at: {public_url}")

# Run the Flask app
app.run(port=5000)
