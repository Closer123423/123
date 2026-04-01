from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os

# ====================== 你的个人信息（已填好）======================
STUDENT_ID = "202335020631"  # 你的学号
STUDENT_NAME = "麦澄明"       # 你的姓名
# ==========================================================================

# 初始化Flask应用
app = Flask(__name__)

# 配置上传文件存储路径（严格对应项目结构）
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# 自动创建上传目录，避免路径错误
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载YOLOv8轻量预训练模型（首次运行自动下载，无需手动操作）
model = YOLO('yolov8n.pt')

# 首页路由：展示学号、姓名 + 图片上传界面
@app.route('/')
def index():
    return render_template(
        'index.html',
        student_id=STUDENT_ID,
        student_name=STUDENT_NAME
    )

# AI目标检测接口：接收上传图片、执行推理、返回检测结果
@app.route('/predict', methods=['POST'])
def predict():
    # 校验上传文件
    if 'file' not in request.files:
        return jsonify({"status": "fail", "msg": "未检测到上传文件"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "fail", "msg": "请选择有效图片文件"})

    if file:
        # 保存原始图片
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(original_path)

        # 执行AI目标检测推理
        results = model(original_path, save=False)

        # 手动保存检测结果到static/uploads，确保前端可访问
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        results[0].save(result_path)

        # 拼接前端可访问的静态资源路径
        result_access_path = f"/static/uploads/{result_filename}"

        return jsonify({
            "status": "success",
            "result_path": result_access_path
        })

    # 兜底异常处理
    return jsonify({"status": "fail", "msg": "未知错误，请重试"})

# 启动Flask服务（严格缩进，零错误）
if __name__ == '__main__':
    # 关闭Flask自带debug，避免VSCode调试冲突；端口5001规避系统占用
    app.run(debug=False, host='0.0.0.0', port=5001)
