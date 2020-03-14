import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "ronsong0809"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file1 = request.files['file1']
        if file1.filename == '':
            flash('No selected file')
            return render_template('index.html')

        if file1 and allowed_file(file1.filename):
            filename1 = "input.png"
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        
        if 'file2' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file2 = request.files['file2']
        if file2.filename == '':
            flash('No selected file')
            return render_template('index.html')

        if file2 and allowed_file(file2.filename):
            filename2 = "reference.pdf"
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        return redirect("/result")
    return render_template('index.html')

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=8000, debug=True)