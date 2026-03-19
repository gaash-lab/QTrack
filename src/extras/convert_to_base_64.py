from pdf2image import convert_from_path
import base64, io

path = "your_file.pdf"

imgs = convert_from_path(path, dpi=72)
img = imgs[0]  

w, h = img.size
max_w = 1400
if w > max_w:
    ratio = max_w / w
    img = img.resize((max_w, int(h * ratio)))

buf = io.BytesIO()
img.save(buf, format="JPEG", quality=82)

b64 = base64.b64encode(buf.getvalue()).decode()

with open("output.b64", "w") as f:
    f.write(b64)