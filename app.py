from flask import Flask, request, render_template,session, redirect, url_for, jsonify, send_file 
from textblob import TextBlob
from googletrans import Translator, LANGUAGES
import os
import sys
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
import logging
import re
import uuid
from flask_cors import CORS, cross_origin
from text_to_speech.components.textToSpeech import TTSapplication
from text_to_speech.components.get_accent import get_accent_tld, get_accent_message

# Suppress warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.secret_key = "NIKHILSHAH6199" 
CORS(app)

# *********************** Home Page ****************************
@app.route('/')
def home():
    return render_template("home.html")

# ********************* Spell Checker ******************************
@app.route("/spellchecker", methods=["GET", "POST"])
def spellchecker():
    if request.method == "POST":
        input_text = request.form.get("text")
        if input_text:
            corrected_text = str(TextBlob(input_text).correct())
            return render_template("SpellCh.html", original_text=input_text, corrected_text=corrected_text)
    return render_template("SpellCh.html", original_text="", corrected_text="")

# ********************* Lang Identification & Translation *******************
translator = Translator()

def detect_and_translate(text, target_lang):
    detected = translator.detect(text)
    detected_lang = detected.lang  # Detected language code
    translation = translator.translate(text, dest=target_lang).text
    return detected_lang, translation

@app.route('/trans', methods=['POST', 'GET'])
def trans():
    translation = ""
    detected_lang = ""
    if request.method == 'POST':
        text = request.form['text']
        target_lang = request.form['target_lang']
        detected_lang, translation = detect_and_translate(text, target_lang)

    detected_lang_name = LANGUAGES.get(detected_lang, "Unknown")
    return render_template('Langdet.html', 
                           translation=translation, 
                           detected_lang=detected_lang_name, 
                           languages=LANGUAGES)

# ************************************ TTS *******************************

# Directory to store audio files
OUTPUT_DIR = "text_to_speech/artifact/tts_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

@app.route("/tts", methods=['GET'])
@cross_origin()
def tts():
    accent_list = get_accent_message()
    return render_template('TTS.html', accent_list=accent_list)

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        data = request.json['data']
        accent_input = request.json['accent']
        accent = get_accent_tld(accent_input)
        tts_result = TTSapplication().text2Speech(data, accent)

        # Generate a unique file name
        file_name = f"tts_audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # Save the audio file
        with open(file_path, "wb") as audio_file:
            audio_file.write(tts_result)

        return {"data": tts_result.decode("utf-8"), "file_name": file_name}

@app.route("/download/<file_name>", methods=['GET'])
@cross_origin()
def download(file_name):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return {"error": "File not found"}, 404


# ******************** Text Summarization *******************************
# Function to process PDF and extract text

def extract_text_from_pdf(file):
    logging.info("Starting PDF text extraction...")
    text = ""
    
    try:
        with pdfplumber.open(file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    logging.info(f"Extracted text from page {page_number}")
                    # Clean the extracted text
                    page_text = clean_text(page_text)
                    text += page_text.strip() + "\n"
        logging.info("PDF text extraction completed.")
    except Exception as e:
        logging.error(f"Error during PDF text extraction: {str(e)}")
        raise

    # Final cleaning and joining
    return " ".join(text.split())

def clean_text(text):
    """
    Cleans the extracted text to remove gibberish and unwanted characters.
    """
    # Remove known gibberish patterns like (cid:xxx)
    text = re.sub(r'\(cid:\d+\)', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ASCII characters, if necessary
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Additional custom replacements (e.g., fixing common OCR errors)
    text = text.replace('o(cid:332)', 'often')
    return text


# Function to summarize large text in chunks
def summarize_text(
    text, 
    max_length=200, 
    min_length=100, 
    chunk_size=1024, 
    model_name="facebook/bart-large-cnn"
):
    logging.info("Starting text summarization...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

        summaries = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            tokenized = tokenizer(chunk, truncation=True, max_length=chunk_size)
            adjusted_max_length = min(max_length, len(tokenized['input_ids']) // 2)
            adjusted_min_length = min(min_length, adjusted_max_length // 2)

            logging.info(f"Summarizing chunk {i // chunk_size + 1}")
            summary = summarizer(
                chunk,
                max_length=adjusted_max_length,
                min_length=adjusted_min_length,
                do_sample=False
            )
            summaries.append(summary[0]['summary_text'])

        logging.info("Text summarization completed.")
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error during text summarization: {str(e)}")
        return f"Error summarizing text: {str(e)}"

@app.route("/textsum", methods=["GET", "POST"])
def textsum():
    # Initialize default values
    extracted_text = None
    manual_input_text = None
    summary = None
    error_message = None

    if request.method == "POST":
        # Reset session values on form submission
        session["previous_summary"] = None
        session["previous_text"] = None

        # Check if manual text is provided
        manual_input_text = request.form.get("manual_text", "").strip()
        if manual_input_text:
            logging.info("Received manual text for summarization.")
            session["previous_text"] = manual_input_text  # Save the new text in the session
            summary = summarize_text(
                manual_input_text,
                max_length=200,
                min_length=100,
                chunk_size=1024,
                model_name="facebook/bart-large-cnn"
            )
            session["previous_summary"] = summary  # Save the summary in the session
        elif "pdf_file" in request.files:
            file = request.files["pdf_file"]
            if file.filename == "":
                error_message = "No file selected"
                logging.error(error_message)
            else:
                try:
                    logging.info("Received file for processing.")
                    extracted_text = extract_text_from_pdf(file)
                    session["previous_text"] = extracted_text  # Save the text in the session
                    summary = summarize_text(
                        extracted_text,
                        max_length=200,
                        min_length=100,
                        chunk_size=1024,
                        model_name="facebook/bart-large-cnn"
                    )
                    session["previous_summary"] = summary  # Save the summary in the session
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    logging.error(error_message)

        # Redirect to the same route to refresh the page and clear form inputs
        return redirect(url_for("textsum"))

    # Serve the page with default or session values for GET requests
    summary = session.get("previous_summary")
    extracted_text = session.get("previous_text") if session.get("previous_summary") else None

    return render_template(
        "TextSum.html",
        extracted_text=extracted_text,
        manual_input_text="",
        summary=summary,
        error_message=error_message
    )


# ******************** Run ****************************
if __name__ == "__main__":
    app.run(debug=True)
