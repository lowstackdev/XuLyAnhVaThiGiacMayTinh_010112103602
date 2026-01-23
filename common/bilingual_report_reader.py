import logging
from PyPDF2 import PdfReader
from googletrans import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text.strip()

def get_bilingual_report_text(pdf_path, target_lang="vi"):
    en_text = extract_text_from_pdf(pdf_path)

    if not en_text:
        logger.warning(f"No text could be extracted from: {pdf_path}")
        return None

    clean_en = en_text.split("DISCLAIMER")[0].split("Disclaimer")[0].strip()

    try:
        translator = Translator()
        translated = translator.translate(clean_en, src='en', dest=target_lang).text
        return {
            "en": clean_en,
            "translated": translated
        }
    except Exception as e:
        logger.error(f"Translation error in bilingual reader: {e}")
        return {
            "en": clean_en,
            "translated": ""
        }
