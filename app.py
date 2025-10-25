import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd

# استيراد الكلاس الخاص بك من ملف deidentifier.py
from deidentifier import AdvancedHybridDeidentifierV10_3

# إعدادات أساسية
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تأكد من وجود مجلد 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =================================================================
#  تحميل الموديل مرة واحدة فقط عند بدء تشغيل الخادم
# =================================================================
print("🚀 [الخادم]: جاري تحميل الموديل... قد يستغرق هذا بعض الوقت.")
deidentifier = AdvancedHybridDeidentifierV10_3()
print("✅ [الخادم]: تم تحميل الموديل بنجاح! الخادم جاهز.")
# =================================================================

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    # هذه الدالة ستقوم بعرض ملف index.html الخاص بك
    return render_template('index.html')

@app.route('/process-text', methods=['POST'])
def process_text_route():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text_to_process = data['text']
    
    # استخدام الكلاس الخاص بك لمعالجة النص
    result = deidentifier.process_single_text(text_to_process)
    
    # تنسيق النتيجة النهائية لإرسالها للموقع
    return jsonify({
        'processed_text': result.get('final_merged_result', ''),
        'entities': result.get('final_merged_entities', [])
    })

@app.route('/process-file', methods=['POST'])
def process_file_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"deidentified_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        file.save(input_path)
        
        # استخدام الكلاس الخاص بك لمعالجة الملف
        deidentifier.process_excel_file(input_path, output_path)
        
        # إرسال الملف الجديد للمستخدم ليقوم بتحميله
        return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)
    
    return jsonify({'error': 'File not allowed'}), 400

if __name__ == '__main__':
    # تشغيل الخادم
    # عند النشر على Render، سيتم استخدام أمر مختلف (gunicorn)، لكن هذا جيد للتجربة المحلية
    app.run(host='0.0.0.0', port=5000, debug=True)