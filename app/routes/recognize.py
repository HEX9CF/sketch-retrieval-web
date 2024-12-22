# from io import BytesIO
#
# from PIL import Image
# from flask import request, jsonify
# import torch
# from matplotlib import pyplot as plt
#
# from app import app
# from app.utils.retrieval import NN
# from app.utils.retrieval import transform
# from app.models.resp import Resp
#
# input_size = 784
# hidden_size = 500
# output_size = 10
# model_path = 'weights/model.pth'
#
# @app.route('/api/recognize', methods=['POST'])
# def recognize():
#     file = request.files.get('file')
#     if file:
#         print('上传文件:', file.filename)
#         file_content = file.read()
#         file_stream = BytesIO(file_content)
#         try:
#             image = Image.open(file_stream)
#             # print(image.__dict__)
#             image.copy().verify()
#         except (IOError, SyntaxError) as e:
#             print(e)
#             resp = Resp(0, '请上传有效的图片文件！', None)
#             return jsonify(resp.__dict__)
#
#         # 展示图片
#         plt.imshow(image.copy())
#         plt.show()
#
#         # 预处理
#         image = transform(image)
#         image = image.view(1, -1)
#         # print(image)
#
#         # 加载模型
#         model = NN(input_size, hidden_size, output_size)
#         model.load_state_dict(torch.load(model_path, weights_only=True))
#
#         # 预测
#         predicted = model.predict(image)
#         data = str(predicted.item())
#         print('预测结果:', data)
#
#         resp = Resp(1, 'OK', data)
#         return jsonify(resp.__dict__)
#
#     else:
#         resp = Resp(0, '请上传图片文件！', None)
#         return jsonify(resp.__dict__)
#
