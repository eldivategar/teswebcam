from flask import Flask, render_template, Response, redirect, url_for
import cv2
import bleedfacedetector as fd
import numpy as np
import time


app=Flask(__name__)


model = 'model/emotion-ferplus-8-final.onnx'

def init_emotion():
    
    # Set global variables
    global net,emotions
    
    # Define the emotions
    emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
    
    # Initialize the DNN module
    net = cv2.dnn.readNetFromONNX(model)


def emotion(image, returndata=False, confidence=0.3):
    
    # Make copy of  image
    img_copy = image.copy()
    
    # Detect face in image
    faces = fd.ssd_detect(img_copy,conf=confidence)
    
    # Define padding for face ROI
    padding = 3 
    
    # Iterate process for all detected faces
    for x,y,w,h in faces:
        
        # Get the Face from image
        face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
        
        # Convert the  detected face from BGR to Gray scale
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        # Resize the gray scale image into 64x64
        resized_face = cv2.resize(gray, (64, 64))
        
        # Reshape the final image in required format of model
        processed_face = resized_face.reshape(1,1,64,64)
        
        # Input the processed image
        net.setInput(processed_face)
        
        # Forwards pass
        Output = net.forward()
 
        #Compute softmax values for each sets of scores  
        expanded = np.exp(Output - np.max(Output))
        probablities =  expanded / expanded.sum()
        
        # Get the final probablities 
        prob = np.squeeze(probablities)
        
        # Get the predicted emotion
        global predicted_emotion
        predicted_emotion = emotions[prob.argmax()]
       
        # Write predicted emotion on image
        cv2.putText(img_copy,'{}'.format(predicted_emotion),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 
                        2, cv2.LINE_AA)
        # Draw rectangular box on detected face
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
    
    if  returndata:
        # Return the the final image if return data is is True
        return img_copy

camera = cv2.VideoCapture(0)

def gen_frames():  
    init_emotion()    
    fps=0

    while (True):
        start_time = time.time()
        success, frame= camera.read()   
        
        if not success:
            break                                      
        
        image = emotion(frame, returndata=True, confidence = 0.8)
        fps= (1.0 / (time.time() - start_time))

        cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)        

        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                  

    

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/video')
def video():
    Video = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Video

@app.route('/done')
def done():
    camera.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)