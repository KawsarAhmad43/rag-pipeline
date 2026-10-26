from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def text_to_pdf(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 40

    for line in lines:
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line.strip())
        y -= 15

    c.save()

if __name__ == "__main__":
    text_to_pdf("/home/aiteam/Desktop/rag_pipelines/helper/X.pdf", "/home/aiteam/Desktop/rag_pipelines/helper/X_valid.pdf")
