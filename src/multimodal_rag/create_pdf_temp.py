
from reportlab.pdfgen import canvas

def create_dummy_pdf(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "This is a dummy PDF for testing multimodal RAG.")
    c.drawString(100, 730, "It contains some text to be indexed.")
    c.save()

if __name__ == "__main__":
    create_dummy_pdf("/tmp/test_session_upload_dummy.pdf")
