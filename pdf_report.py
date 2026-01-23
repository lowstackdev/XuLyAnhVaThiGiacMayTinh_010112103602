from xhtml2pdf import pisa
from jinja2 import Template
import os
from datetime import datetime

def generate_pdf_report(output_path, data=None):
    if data is None:
        data = {}

    # Load HTML template
    template_path = os.path.join(os.path.dirname(__file__), 'pdf_report_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    template = Template(template_content)
    html_content = template.render(data=data)

    # Convert to PDF
    with open(output_path, "wb") as result_file:
        pisa.CreatePDF(html_content, dest=result_file)
