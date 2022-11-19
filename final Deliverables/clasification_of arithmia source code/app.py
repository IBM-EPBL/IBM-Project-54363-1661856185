
from flask import Flask,render_template,request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model=load_model('model.h5')

@app.route('/',methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/info',methods=['GET'])
def info():
	return render_template('info.html')

@app.route('/predict',methods=['GET'])
def hello_world():
    return render_template('predict.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method=='POST':
		imagefile=request.files['imagefile']
		image_path="./static/images/" +imagefile.filename
		imagefile.save(image_path)

		img=image.load_img(image_path,target_size=(64,64))
		x=image.img_to_array(img)
		x=np.expand_dims(x,axis=0)

		pred = model.predict(x)
		print("prediction",pred)

		class_name=["Left Bundle Branch Block","Normal","Premature Atrial Contraction","Premature Ventricular Contractions","Right Bundle Branch Block","Ventricular Fibrillation"]
		pred_id = pred.argmax(axis=1)[0]
		result = str(class_name[pred_id])
		return render_template('predict.html',predition=result,path=image_path)
	return render_template('predict.html')

    
if __name__ == '__main__':
 	app.run(debug=True)
