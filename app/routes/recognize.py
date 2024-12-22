import base64
from io import BytesIO

from PIL import Image
from flask import request, jsonify

from app import app
from app.models.resp import Resp
from app.models.vgg16 import transform, load_model, extract_feature, retrieve_images, show_retrieval

@app.route('/api/recognize', methods=['POST'])
def recognize():
    file = request.files.get('file')
    if file:
        print('上传文件:', file.filename)
        file_content = file.read()
        file_stream = BytesIO(file_content)
        try:
            image = Image.open(file_stream)
            # print(image.__dict__)
            image.copy().verify()
        except (IOError, SyntaxError) as e:
            print(e)
            resp = Resp(0, '请上传有效的图片文件！', None)
            return jsonify(resp.__dict__)

        sketch_image = image.copy()

        # 图片预处理
        image = image.convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        # 调用模型
        model = load_model()
        feature = extract_feature(model, image)
        retrievals = retrieve_images(feature, 5)

        show_retrieval(sketch_image, retrievals)

        # 转码
        data = []
        for retrieval in retrievals:
            buffer = BytesIO()
            retrieval.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data.append(img_str)

        resp = Resp(1, 'OK', data)
        return jsonify(resp.__dict__)

    else:
        resp = Resp(0, '请上传图片文件！', None)
        return jsonify(resp.__dict__)
