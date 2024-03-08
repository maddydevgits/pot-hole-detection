from flask import Flask, render_template, request, redirect, url_for, jsonify,session
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import os
from bson import ObjectId
from math import radians, sin, cos, sqrt, atan2

from pymongo import MongoClient

client=MongoClient('mongodb://127.0.0.1:27017')
db=client['PotHoleDetection']
c=db['register']
c1=db['potholes']
 
app = Flask(__name__)
app.secret_key='1234'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 
# Load the pre-trained Keras model for pothole detection
model = load_model('keras_Model.h5', compile=False)
 
# Load the labels
with open('labels.txt', 'r') as labels_file:
    class_names = labels_file.readlines()
 
def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Radius of the Earth in kilometers
    R = 6371.0

    # Difference in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance

def detect_pothole(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array
 
    # Make predictions
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])
 
    # Draw text on the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{class_name} - Confidence: {confidence_score:.2f}"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)
 
    # Save the modified image
    modified_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_image.jpg')
    img.save(modified_filename)
 
    return modified_filename, class_name, confidence_score
 
 
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/authority_login')
def authority_login():
    return render_template('alogin.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html') 

@app.route('/aloginform',methods=['post'])
def aloginform():
    username=request.form['username']
    password=request.form['password']
    print(username,password)

    if username=="admin" and password=="admin123":
        return redirect('/adashboard')
    else:
        return render_template('alogin.html',error='invalid credentials')

@app.route('/updatepothole')
def updatepothole():
    data=[]
    for i in c1.find():
        if(i['status']!='3'):
            dummy=[]
            dummy.append(i['_id'])
            data.append(dummy)
    return render_template('updatepothole.html',l=len(data),dashboard_data=data)

@app.route('/updatepotholeform',methods=['post'])
def updatepotholeform():
    potholeid=request.form['potholeid']
    potholestatus=request.form['potholestatus']
    print(potholeid,potholestatus)

    c1.update_one({'_id':ObjectId(potholeid)},{'$set':{'status':potholestatus}})
    data=[]
    for i in c1.find():
        if(i['status']!='3'):
            dummy=[]
            dummy.append(i['_id'])
            data.append(dummy)

    return render_template('updatepothole.html',result="status updated",l=len(data),dashboard_data=data)

@app.route('/adashboard')
def adashboard():
    data=[]
    for i in c1.find():
        dummy=[]
        dummy.append(i['latitude'])
        dummy.append(i['longitude'])
        if(i['status']==str('0')):
            dummy.append("Complaint Initiated")
        elif(i['status']==str('1')):
            dummy.append("Complaint Accepted")
        elif(i['status']==str('2')):
            dummy.append("Complaint In-Processing")
        elif(i['status']==str('3')):
            dummy.append('Complaint Closed')
        # dummy.append(i['status'])
        k=i['filepath']
        dummy.append(k)
        dummy.append(i['class_type'])
        data.append(dummy)
    print(data)
    return render_template('adashboard.html',l=len(data),dashboard_data=data)

@app.route('/signupform',methods=['post'])
def signupform():
    username=request.form['username']
    email=request.form['email']
    password=request.form['password']
    print(username,email,password)

    data={}
    data['username']=username
    data['password']=password
    data['email']=email

    for i in c.find():
        if(i['email']==data['email']):
            return render_template('signup.html',err='Account Already Exist')
        if(i['username']==data['username']):
            return render_template('signup.html',err='username already exist')
    
    c.insert_one(data)
    return render_template('signup.html',res='Registration Successful')

@app.route('/loginform',methods=['post'])
def loginform():
    username=request.form['username']
    password=request.form['password']
    print(username,password)
    
    data={}
    data['username']=username
    data['password']=password

    for i in c.find():
        if(i['password']==data['password'] and i['username']==data['username']):
            session['username']=username
            return redirect('/dashboard')
        
    return render_template('login.html',err='invalid credentials')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
 
    image_file = request.files['image']
 
    if image_file.filename == '':
        return redirect(request.url)
 
    if image_file:
        # Generate a unique filename
        print(os.listdir(app.config['UPLOAD_FOLDER']))
        if session['username'] not in os.listdir(app.config['UPLOAD_FOLDER']):
            print('creating directory')
            os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'],session['username']))
        
        filename = os.path.join(app.config['UPLOAD_FOLDER']+'/'+session['username'], image_file.filename)
        image_file.save(filename)
 
        # Perform pothole detection using the loaded Keras model
        modified_image, class_name, confidence_score = detect_pothole(filename)

        pothole_data={}
        pothole_data['identified_by']=session['username']
        pothole_data['latitude']=session['latitude']
        pothole_data['longitude']=session['longitude']
        pothole_data['filepath']=filename[7:]
        pothole_data['class_type']=class_name
        pothole_data['status']='0'
        
        flag=0
        for i in c1.find():
            if calculate_distance(pothole_data['latitude'],pothole_data['longitude'],i['latitude'],i['longitude'])<0.2:
                flag=1
        
        if(flag==0):
            c1.insert_one(pothole_data)
        else:
            return render_template('index.html',error='it was already informed')
  
        # Render the result template with the detection results
        return render_template('index.html',res=class_name +', updated to authority')
 
    return redirect(request.url)


@app.route('/receive_location', methods=['POST'])
def receive_location():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']

    # Do something with latitude here, like store it in a database or process it
    print("Received latitude:", latitude)
    print("Received Longitude:",longitude)

    session['latitude']=latitude
    session['longitude']=longitude

    return render_template('index.html',res='location updated')

@app.route('/mypotholes')
def mypotholes():
    data=[]
    for i in c1.find({'identified_by':session['username']}):
        dummy=[]
        dummy.append(i['latitude'])
        dummy.append(i['longitude'])
        if(i['status']==str('0')):
            dummy.append("Complaint Initiated")
        elif(i['status']==str('1')):
            dummy.append("Complaint Accepted")
        elif(i['status']==str('2')):
            dummy.append("Complaint In-Processing")
        elif(i['status']==str('3')):
            dummy.append('Complaint Closed')
        dummy.append(i['filepath'])
        dummy.append(i['class_type'])
        data.append(dummy)
    print(data)
    return render_template('mypotholes.html',l=len(data),dashboard_data=data)

@app.route('/logout')
def logout():
    session['username']=None
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)