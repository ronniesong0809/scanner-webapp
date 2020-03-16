import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import cv2
import sys
import numpy as np
import random
from pdf2image import convert_from_path
import img2pdf

port = int(os.environ.get("PORT", 5000))
UPLOAD_FOLDER = 'static'
ALLOWED_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_PDF = {'pdf'}

app = Flask(__name__)
app.secret_key = "ronsong0809"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE

def allowed_pdf(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_PDF

@app.route('/', methods=['GET', 'POST'])
def index():
    if os.path.exists("static/input.png") and os.path.exists("static/reference.pdf"):
        os.remove("static/input.png")
        os.remove("static/reference.png")
        os.remove("static/reference.pdf")
        os.remove("static/output_orb.png")
        os.remove("static/output.pdf")
        os.remove("static/matches_orb.png")
        return render_template('index.html')
    else:
        if request.method == 'POST':
            if 'file1' not in request.files or 'file2' not in request.files:
                flash('No file part')
                return '''
                    <div style="text-align: center;">
                        <div id="content">
                            <h3>No file part, please <a href="">try again</a>!</h3>
                        </div>
                    </div>
                    '''

            file1 = request.files['file1']
            file2 = request.files['file2']

            if file1.filename == '' or file2.filename == '':
                flash('No selected file')
                return '''
                    <div style="text-align: center;">
                        <div id="content">
                            <h3>No selected file, please <a href="">try again</a>!</h3>
                        </div>
                    </div>
                    '''

            if not allowed_image(file1.filename) or not allowed_pdf(file2.filename):
                flash('Incorrect file type')
                return '''
                    <div style="text-align: center;">
                        <div id="content">
                            <h3>Incorrect file type, please <a href="">try again</a>!</h3>
                        </div>
                    </div>
                    '''

            if file1 and allowed_image(file1.filename) and file2 and allowed_pdf(file2.filename):
                filename1 = "input.png"
                file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                filename2 = "reference.pdf"
                file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            return redirect("/result")
        return render_template('index.html')

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def align_orb_homo(img_1, img_2):
    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors.
    orb = cv2.ORB_create(2000)
    kp_1, des_1 = orb.detectAndCompute(gray_1, None)
    kp_2, des_2 = orb.detectAndCompute(gray_2, None)

    # drawKeypoints(img_1, img_2, gray_1, gray_2, kp_1, kp_2)

    # find matches
    match_list = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = match_list.match(des_1, des_2, None)

    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)

    # keep good matches
    good = int(len(matches) * 0.15)
    matches = matches[:good]

    # draw matches
    image = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches, None)
    cv2.imwrite("static/matches_orb.png", image)

    # extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp_1[match.queryIdx].pt
        points2[i, :] = kp_2[match.trainIdx].pt

    # find homography
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("homo: \n",  homography)

    # backward mapping
    h, w, _ = img_2.shape
    result = cv2.warpPerspective(img_1, homography, (w, h))

    return result

def scan():
    # read images
    img_1 = cv2.imread("static/input.png", cv2.IMREAD_COLOR)

    pages = convert_from_path('static/reference.pdf', size=(805,876))
    for page in pages:
        page.save('static/reference.png', 'PNG')

    # align image basic on the reference
    img_2 = cv2.imread("static/reference.png", cv2.IMREAD_COLOR)

    # orb features
    result1 = align_orb_homo(img_1, img_2)
    # save image
    cv2.imwrite("static/output_orb.png", result1)

    with open("static/output.pdf","wb") as f:
	    f.write(img2pdf.convert('static/output_orb.png'))

@app.route('/result')
def result():
    scan()
    return render_template('result.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=port, debug=True)