import os
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import numpy as np
import argparse

# suppresses all tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Recieve arguments from command line
parser = argparse.ArgumentParser(description='Takes in Arguments for Image prediction')
parser.add_argument('image_path')
parser.add_argument('--model','-m', type=str, help='full path to model', default='./model_1657087576.h5')
parser.add_argument('--class_file','-c', type=str, help='full path to json file of class name', default='./label_map.json')
parser.add_argument('--top_k','-t', type=int, help='Number of top Predictions', default= 5)
args = parser.parse_args()

# collect the supplied command line arguments
input_image = args.image_path
input_model = args.model
input_classes = args.class_file
input_top_k = args.top_k

# load the class name
with open(input_classes, 'r') as f:
    class_names = json.load(f)


def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224,224])
    img /= 255
    img = img.numpy()
    assert img.max() <= 1.0
    assert img.shape == (224,224,3)
    return img


def make_index(predictions):
    pred_map = [(str(index+1), value) for index, value in enumerate(predictions)]
    pred_map.sort(key=lambda x: x[1])
    pred_map.reverse()
    return pred_map


def predict(image_path, model, top_k=5):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    prediction = np.array(prediction).squeeze()
    prediction = make_index(prediction)
    prediction = prediction[:top_k]
    prediction = {class_names[predict[0]]:predict[1] for predict in prediction}
  
    return prediction


def run():
    "Runs the program as a standalone application"
    
    # define the model
    model = tf.keras.models.load_model(input_model, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

    # make predictions
    outcome = predict(input_image, model,input_top_k)
    
    # Draw a Probability table
    template = '\t{:>20}  {}  {:5.5}'
    template2 = '\t{:>20}  {}  {}'
    print('\tBelow is the Prediction\n')
    print(template2.format('Name','|','Predicted Probability'))
    print('\t{}'.format('-' * 50))
    for item, value in outcome.items():
        print(template.format(item.capitalize(),'|',value))
 
    return outcome  # return the probability dictionary

if __name__ == '__main__':
    run()
