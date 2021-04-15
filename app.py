import os

from flask import Flask, request, abort, jsonify, send_from_directory, render_template, redirect,flash,url_for
from imageai.Detection import ObjectDetection

UPLOAD_DIRECTORY = "./static/api_uploaded_files"
UPLOAD_FILENAME="default.jpg"
MODEL_DIRECTORY = "./model"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


api = Flask(__name__)


@api.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    return os.listdir(UPLOAD_DIRECTORY)
       
@api.route('/')
def upload_form():
    return render_template('index.html')
 
    
@api.route("/", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        print('No image found')
        return redirect(request.url)
    image = request.files["file"]
    print(image.filename)
    image.save(os.path.join(UPLOAD_DIRECTORY, image.filename))    
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(MODEL_DIRECTORY , "yolo.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(UPLOAD_DIRECTORY , image.filename), output_image_path=os.path.join(UPLOAD_DIRECTORY , image.filename))

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )       
    return render_template('index.html', filename=image.filename)
   


@api.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


@api.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + UPLOAD_DIRECTORY + '/' + filename)
    return redirect(url_for('static', filename='api_uploaded_files/' + filename),code=301)


if __name__ == "__main__":
    api.run(debug=True, port=8000)