from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np
import os

def main():
    from dotenv import load_dotenv

    try:
        # Get Configuration Settings
        load_dotenv()
        prediction_endpoint = os.getenv('PredictionEndpoint')
        prediction_key = os.getenv('PredictionKey')
        project_id = os.getenv('ProjectID')
        model_name = os.getenv('ModelName')

        # Authenticate a client for the training API
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # ========================== Update for each model: 
        n = 1
        probability_threshold = 8


        # Load image and get height, width and channels
        for i in range(n):
            # image_file = f'testing_images/part_{i}.jpg'
            # image_file = f'testing_images/{i+1}.jpg'
            
            st.title("Metal Condition Detection")

            image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if image_file is not None:

                file_name = image_file.name
                st.write('Detecting objects in ', file_name)
                image = Image.open(image_file)
                h, w, ch = np.array(image).shape

                st.write("Image size = ({h}, {w})")
            

                # st.image(image, caption="Uploaded Image", use_column_width=True)

                image_bytes = image_file.getvalue()
                
                # with open(file_bytes, mode="rb") as image_data:
                results = prediction_client.detect_image(project_id, model_name, image_bytes)

                # Create a figure for the results
                fig = plt.figure(figsize=(w * 1.1, h * 1.1))
                plt.axis('off')

                # Display the image with boxes around each detected object
                draw = ImageDraw.Draw(image)
                lineWidth = int(w/500)
                color = 'magenta'
                for prediction in results.predictions:
                    # Only show objects with a > 50% probability
                    if (prediction.probability*100) > probability_threshold:
                        # Box coordinates and dimensions are proportional - convert to absolutes
                        left = prediction.bounding_box.left * w 
                        top = prediction.bounding_box.top * h 
                        height = prediction.bounding_box.height * h
                        width =  prediction.bounding_box.width * w
                        # Draw the box
                        points = ((left,top), (left+width,top), (left+width,top+height), (left,top+height),(left,top))
                        draw.line(points, fill=color, width=lineWidth)
                        # Add the tag name and probability
                        plt.annotate(prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100),(left,top), 
                            backgroundcolor=color, fontsize=20)
                st.image(image, caption="Detected Output", use_column_width=True)
            
            # # outputfile = f'res_part_{i}.jpg'
            # output_path = "output"
            # os.makedirs(output_path, exist_ok=True)

            # fig.savefig(outputfile)
            # fig.savefig(os.path.join(output_path, f'res_part_{i}.jpg'))
            # fig.savefig(os.path.join(output_path, f'res_part_{i+1}.jpg'))

            # print('Results saved in ', output_path)
    except Exception as ex:
        st.write(ex)

if __name__ == "__main__":
    main()
