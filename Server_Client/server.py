from flask import Flask, request, jsonify, send_from_directory, render_template
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import base64
import io
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__, template_folder="templates")  # Xác định thư mục chứa HTML
CORS(app)  

# Load model và processor từ thư mục đã lưu
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Đường dẫn đến thư mục con trong folder hiện tại
model_save_path = "./saved_model"

# Lưu mô hình và processor
model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)
# Chuyển model sang thiết bị thích hợp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    try:
        return render_template('client.html')
    except Exception as e:
        return f"Lỗi khi load trang: {str(e)}", 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    try:
        data = request.get_json()
        img_data = data.get('image')

        # Chuyển đổi từ base64 thành ảnh
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")

        # Tạo caption
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_caption = generated_caption.replace(' - ', '-')

        return jsonify({"generated_caption": generated_caption})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)