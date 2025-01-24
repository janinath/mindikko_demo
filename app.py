from flask import Flask, request, jsonify,Response
import openai
import os
from dotenv import load_dotenv
from flask_cors import CORS
from PIL import Image
import pytesseract
import cv2
import numpy as np
#importing the required libraries
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json, load_model
import face_recognition
import traceback
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("API_KEY")
# client = OpenAI(
#     api_key=os.environ.get("API_KEY"),  # This is the default and can be omitted
# )
app = Flask(__name__)
CORS(app)

# # MongoDB Setup
# client = MongoClient("mongodb://127.0.0.1:27017/")  # Adjust as needed
# db = client['chatbot']
# messages_collection = db['messages']

def enhance_image(image_cv):
    """Process the image using OpenCV to extract information."""
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    inverted_image = cv2.bitwise_not(edges)
    return inverted_image



@app.route('/predict', methods=['POST'])
def predict():
    image_text = ""
    user_message = ""
    
    # Check if 'file' is present in the request
    if 'file' in request.files:
        image_file = request.files['file']
        try:
            # Extract text from image
            image = Image.open(image_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            extracted_text = pytesseract.image_to_string(image)
            if not extracted_text.strip():
                processed_image = enhance_image(image_cv)
                processed_pil_image = Image.fromarray(processed_image)
                extracted_text = pytesseract.image_to_string(processed_pil_image)

            if extracted_text.strip():
                image_text = extracted_text.strip()
            else:
                return jsonify({"status": "No text found in the image"}), 400
        except Exception as e:
            return jsonify({"status": f"Error processing image: {str(e)}"}), 500

    # Check if 'statement' (message) is present in the request
    if request.is_json:
        data = request.get_json()
        user_message = data.get("statement", "").strip()
    combined_input = user_message
    #Combine image text and user message if both are provided
    if image_text:
        if combined_input:
            combined_input = f"{user_message}\n{image_text}"
        else:
            combined_input = image_text

    if not combined_input:
        return jsonify({"status": "No input provided"}), 400

    try:
        # Generate bot response
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": combined_input}]
        )
        bot_message = response['choices'][0]['message']['content']
        # # Save the conversation to the database
        # messages_collection.insert_one({
        #     "date": datetime.now(),
        #     "user_message": combined_input,
        #     "bot_response": bot_message
        # })
        return jsonify({"status": bot_message, "extracted_text": image_text, "user_message": user_message}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route('/detect_emotion', methods=['GET'])
def detect_emotion():
 try:
    #capture the video from default camera 
  webcam_video_stream = cv2.VideoCapture(0)

    #load the model and load the weights
    #face_exp_model = model_from_json(open("datasets/facial_expression_model_structure.json","r",encoding="utf-8").read())
    #face_exp_model.load_weights('datasets/facial_expression_model_weights.h5')
  face_exp_model = load_model('datasets/facial_expression_model_combined.h5')

    #declare the emotions label
  emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


    #initialize the array variable to hold all face locations in the frame
  all_face_locations = []
  def generate_emotions():
    print('secnd fucn called')
    #loop through every frame in the video
    while True:
        print('inside while loop')
        #get the current frame from the video stream as an image
        ret,current_frame = webcam_video_stream.read()
        if not ret:
                    yield "data: {\"error\": \"Failed to access the webcam.\"}\n\n"
                    break
        #resize the current frame to 1/4 size to proces faster
        current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
        #detect all faces in the image
        #arguments are image,no_of_times_to_upsample, model
        all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
        
        #looping through the face locations
        for index,current_face_location in enumerate(all_face_locations):
            #splitting the tuple to get the four position values of current face
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
            #change the position maginitude to fit the actual size video frame
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4
            #printing the location of current face
            print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        
            #Extract the face from the frame, blur it, paste it back to the frame
            #slicing the current face from main image
            current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
            
            #draw rectangle around the face detected
            cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
            
            #preprocess input, convert it to an image like as the data in dataset
            #convert to grayscale
            current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
            #resize to 48x48 px size
            current_face_image = cv2.resize(current_face_image, (48, 48))
            #convert the PIL image into a 3d numpy array
            img_pixels = image.img_to_array(current_face_image)
            #expand the shape of an array into single row multiple columns
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            #pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
            #img_pixels /= 255 
            
            #do prodiction using model, get the prediction values for all 7 expressions
            exp_predictions = face_exp_model.predict(img_pixels) 
            #find max indexed prediction value (0 till 7)
            max_index = np.argmax(exp_predictions)
            print(max_index)
            #get corresponding lable from emotions_label
            emotion_label = emotions_label[max_index]
            # Send emotion result to the client
            yield f"data: {{\"emotion\": \"{emotion_label}\"}}\n\n"
            # #display the name as text in the image
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
            # Display the frame with OpenCV for debugging
            cv2.imshow("Webcam Video", current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    yield "data: {\"message\": \"User exited detection.\"}\n\n"
                    break
            # Clean up
            # webcam_video_stream.release()
            # cv2.destroyAllWindows()
        
        #showing the current face with rectangle drawn
        cv2.imshow("Webcam Video",current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #release the stream and cam
    #close all opencv windows open
    webcam_video_stream.release()
    cv2.destroyAllWindows()
    return jsonify({"emotion": "No face detected"}), 200
  return Response(generate_emotions(), mimetype='text/event-stream')

 except Exception as e:
        print('exp') 
        traceback.print_exc()
        return jsonify({"error": "An error occurred while detecting emotion."}), 500
    


if __name__ == '__main__':
    app.run(debug=True)


