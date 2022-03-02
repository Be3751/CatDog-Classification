from crypt import methods
from urllib import request
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('index.html')
    
    # if request.method == 'POST':
    #     image = request.files['file']
    #     # filepath = './static/' + 


if __name__ == '__main__':
    app.run()