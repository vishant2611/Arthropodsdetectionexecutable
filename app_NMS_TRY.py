import streamlit as st
import time 
import cv2
import os
from PIL import Image
from collections import Counter
import pandas as pd
from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image
from numpy import asarray
from sahi.utils.yolov8 import download_yolov8n_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict, visualize_object_predictions
from IPython.display import Image as IPythonImage

st.set_page_config(layout="wide", page_title="Arthropods Detection APP")

st.write("## Detect Arthropods from your image")
st.write("Try uploading an image to detect the arthropods. Full quality images can be downloaded from the sidebar:grin:")
st.sidebar.write("## Upload :gear:")


#NMS function -  selecting the most confident bounding box detections and eliminating overlapping boxes that are less confident, 
    #based on the provided IoU threshold, thus reducing the number of redundant boxes for the same object.

#sort scores in descending, select box with highest score, calculate IOU(measure of overlap betwen 2 box), 
    # removes box with high IOU i.e if selected box is > than certain threshold, it has high overlap & box is removed
        # from consideration & repeats until there are no more box to consider.

# function looks at bunch of boxes, picks best one & removes any other boxes that overlap too much with it.
    # result is list of best box, name of object they contain and how confident function is that object is really.
def nms(boxes, scores, iou_threshold, object_names):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_indices = []

    while sorted_indices:
        current_index = sorted_indices.pop(0)
        selected_indices.append(current_index)
        rest_boxes = [boxes[i] for i in sorted_indices]
        iou_scores = [iou(boxes[current_index], rest_box) for rest_box in rest_boxes]

        sorted_indices = [
            sorted_indices[i] for i in range(len(iou_scores)) if iou_scores[i] < iou_threshold
        ]

    selected_boxes = [boxes[i] for i in selected_indices]
    selected_object_names = [object_names[i] for i in selected_indices]
    selected_confidence_scores = [scores[i] for i in selected_indices]

    return selected_boxes, selected_object_names, selected_confidence_scores

#calculates the Intersection over Union (IoU), which is a measure of the overlap between two bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def process_image(image_file, confidence_threshold):
       
    # Define the directory to save the uploaded image
    save_dir = '/content/drive/MyDrive/Capstone_new/Data_New_images/'
    
    
    # Save the uploaded image to the specified directory
    saved_image_path = os.path.join(save_dir, image_file.name)
    with open(saved_image_path, 'wb') as f:
        f.write(image_file.getbuffer())
    
    # Now, you can use saved_image_path for further processing
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='/content/drive/MyDrive/Capstone_new/runs/detect/YOLOv8nwithP2/weights/best.pt',
        confidence_threshold=confidence_threshold,
        device='cpu'
    )

    result = get_sliced_prediction(
        saved_image_path,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3
    )

    # reads an image, applies Non-Maximum Suppression to filter the object detections,
        # and then draws red bounding boxes with a thickness of 2 pixels around the detected objects on the image.
    img = cv2.imread(saved_image_path, cv2.IMREAD_UNCHANGED)
    img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    numpydata = asarray(img_converted)
    boxes = [[pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy] for pred in result.object_prediction_list]
    scores = [pred.score.value for pred in result.object_prediction_list]
    names  = [pred.category.name for pred in result.object_prediction_list]
    #iou_threshold = 0.5
    selected_boxes, selected_names, selected_scores = nms(boxes, scores, iou_threshold,names)
    image = cv2.imread(saved_image_path)
    for box, object_name, confidence in zip(selected_boxes, selected_names, selected_scores):
      x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
      cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Green rectangle with thickness 2

      # Increase font size by changing the fontScale parameter
      font_scale = 2.0  # Change this value to adjust font size
      text = f"{object_name}: {confidence:.2f}"
      cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
    
    # for box, object_name, confidence in zip(selected_boxes, selected_names, selected_scores):
    #   x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
    #   cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Green rectangle with thickness 2
      # Display object name and confidence score
      # text = f"{object_name}: {confidence:.2f}"
      # cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite('/content/drive/MyDrive/Capstone_new/result_with_nms.jpg', image)

    # visualize_object_predictions(
    #     numpydata,
    #     object_prediction_list=result.object_prediction_list,
    #     output_dir='/content/drive/MyDrive/Capstone_new',
    #     file_name='result_701',
    #     export_format='jpeg'
    # )
    col1.write("Original Image :camera:")
    col1.image(saved_image_path)
    result_image_path = '/content/drive/MyDrive/Capstone_new/result_with_nms.jpg'
    result_image = PILImage.open(result_image_path)
    # col2.write("Detected Image :beetle:")
    # col2.image(result_image)
    #st.image(result_image, caption='Result')
    object_prediction_list = result.object_prediction_list
    class_map = {
    'MA': 'Melon Aphid',
    'NT': 'Nesidiocorius Tenius',
    'OI': 'Orius Insidiosus',
    'WFT': 'Western Flower Thrips',
    'WF': 'White Fly',
    'TS': 'Two Spotted Spidermite'}
    class_counts = {class_name: 0 for class_name in class_map.keys()}
    for prediction in object_prediction_list:
      class_name = prediction.category.name
      if class_name in class_counts:
        class_counts[class_name] += 1
      
    data = {
    'Abbreviation': [],
    'Full Name': [],
    'Count': []}

    for class_name, count in class_counts.items():
      data['Abbreviation'].append(class_name)
      data['Full Name'].append(class_map.get(class_name, class_name))
      data['Count'].append(count)

    df = pd.DataFrame(data)
    
    # Display class names and counts as table
    #table_data = {"Class Name": list(class_counts.keys()), "Count": list(class_counts.values())}
    table_data = {"Abbreviation": df['Abbreviation'].tolist(), "Full Name": df['Full Name'].tolist(), "Count": df['Count'].tolist()}
    col2.write("## Class Counts")
    # col2.table(table_data)
    col2.write(df.set_index('Abbreviation', drop=True))
    col3.write("Detected Image :beetle:")
    col3.image(result_image)

MAX_FILE_SIZE = 5 * 1024 * 1024 
col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.05, 0.05,key="confidence_slider")
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.1, 0.1, key="iou_slider")

if st.button('Detect Arthropods'):
    
    
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
             st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            with st.spinner(text='In progress'):
              process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')

                bar = st.progress(50)
                time.sleep(3)
                bar.progress(100)
                st.success('Success message')
    else:
        st.error('Please upload an image')
ColorMinMax = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
        background: rgb(1 1 1 / 0%);
    } 
    </style>''', unsafe_allow_html=True)

Slider_Cursor = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #1E90FF; 
        box-shadow: rgba(14, 38, 74, 0.2) 0px 0px 0px 0.2rem;
    } 
    </style>''', unsafe_allow_html=True)

Slider_Number = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        color: #000000; 
    } 
    </style>''', unsafe_allow_html=True)

col = f'''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div {{
        background: linear-gradient(to right, #1E90FF 0%, #1E90FF {confidence_threshold }%, rgba(151, 166, 195, 0.25) {confidence_threshold }%, rgba(151, 166, 195, 0.25) 100%);
    }} 
    </style>'''

ColorSlider = st.markdown(col, unsafe_allow_html=True)

if (confidence_threshold > 0.05 and confidence_threshold < 1.00) and (iou_threshold>0 and iou_threshold<1.00):

    if my_upload is not None:
          if my_upload.size > MAX_FILE_SIZE:
              st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
          else:
              with st.spinner(text='In progress'):
                process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
              with st.spinner(text='In progress'):
                  time.sleep(3)
                  st.success('Done')

                  bar = st.progress(50)
                  time.sleep(3)
                  bar.progress(100)
                  st.success('Success message')
    # else:
    #   st.write("In else")
    #   st.error('Please upload an image')



