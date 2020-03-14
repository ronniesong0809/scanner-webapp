import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import cv2
import sys
import numpy as np
import random
from pdf2image import convert_from_path
import img2pdf

UPLOAD_FOLDER = 'static'
ALLOWED_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_PDF = {'pdf'}

app = Flask(__name__)
app.secret_key = "ronsong0809"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE

def allowed_pdf(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_PDF

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

        if file1 and allowed_image(file1.filename):
            filename1 = "input.png"
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        
        if 'file2' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file2 = request.files['file2']
        if file2.filename == '':
            flash('No selected file')
            return render_template('index.html')

        if file2 and allowed_pdf(file2.filename):
            filename2 = "reference.pdf"
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        return redirect("/result")
    return render_template('index.html')

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def align_sift_homo(img_1, img_2):
    # image to gray
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # detect features and compute descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(gray_1, None)
    kp_2, des_2 = sift.detectAndCompute(gray_2, None)

    # find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_1, des_2, k=2)

    # keep good matches
    good = []
    good1 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good1.append(m)

    # draw matches
    h, w, _ = img_2.shape
    image = np.zeros((h, w))
    image = cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good, None, flags=2)
    cv2.imwrite("static/matches_sift.jpg", image)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # for i, match in enumerate(good1):
    #     points1[i, :] = kp_1[match.queryIdx].pt
    #     points2[i, :] = kp_2[match.trainIdx].pt

    points1 = np.float32([kp_1[m.queryIdx].pt for m in good1]).reshape(-1, 1, 2)
    points2 = np.float32([kp_2[m.trainIdx].pt for m in good1]).reshape(-1, 1, 2)

    # find homography
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("homo: \n",  homography)

    # backward mapping
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

    # sift features
    result1 = align_sift_homo(img_1, img_2)
    # save image
    cv2.imwrite("static/output_sift.png", result1)

    with open("static/output.pdf","wb") as f:
	    f.write(img2pdf.convert('static/output_sift.png'))

@app.route('/result')
def result():
    scan()
    return render_template('result.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=8000, debug=True)