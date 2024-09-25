from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

# Flask 앱 생성
app = Flask(__name__)

# CNN 모델 로드
model = load_model('/Users/kjh/kjh_Project/AI/MP3/cnn_model.h5')

# 이미지 저장 경로
IMAGE_FOLDER = '/Users/kjh/kjh_Project/AI/MP3/cat_dog'

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 리스트 API
@app.route('/images', methods=['GET'])
def get_images():
    images = []
    for label in ['cat', 'dog']:
        label_folder = os.path.join(IMAGE_FOLDER, label)
        for filename in os.listdir(label_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                images.append({'src': f'/images/{label}/{filename}', 'label': label})
    
    random.shuffle(images)
    return jsonify(images[:16])  # 16개 이미지 반환

# 이미지 파일 제공
@app.route('/images/<label>/<filename>')
def serve_image(label, filename):
    return send_from_directory(os.path.join(IMAGE_FOLDER, label), filename)

# 이미지 예측 API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_path = f'/tmp/{file.filename}'
    file.save(file_path)

    # 이미지 전처리
    img = image.load_img(file_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    # 예측 수행
    prediction = model.predict(img)
    label = 'dog' if prediction[0][0] >= 0.5 else 'cat'

    os.remove(file_path)

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
