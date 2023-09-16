import tensorflow as tf


model_folder_path = r'D:\DL projects March\CNN_project\model_archive'


loaded_model = tf.keras.models.load_model(model_folder_path)
import matplotlib.pyplot as plt
from skimage.io import imread
img=imread(r'D:\DL projects March\CNN_project\Radish\1003.jpg')
# plt.imshow(img)
from skimage.transform import resize
img=resize(img,(150,150,1))
img=img.reshape(1,150,150,1)
img.shape
y=loaded_model.predict(img)
ind=y.argmax()
categories=['Broccoli','Capsicum','Bottle_Gourd','Radish','Tomato','Brinjal','Pumpkin','Carrot','Papaya','Cabbage','Bitter_Gourd',
 'Cauliflower','Bean','Cucumber','Potato']
veg=(categories[ind])
print(veg)
import gradio as gr
import openai
import cv2
import numpy as np


openai.api_key = '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'  


model = loaded_model 


def preprocess_image(input_image):
    
    image = np.array(input_image)

    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    
    image = cv2.resize(image, (150, 150)) / 255.0

    
    image = np.expand_dims(image, axis=-1)

    return image

def predict_vegetable_and_chat(input_image):
    
    preprocessed_image = preprocess_image(input_image)

    
    predicted_vegetable = model.predict(np.array([preprocessed_image]))
    ind=predicted_vegetable.argmax()
    categories=['Broccoli','Capsicum','Bottle_Gourd','Radish','Tomato','Brinjal','Pumpkin','Carrot','Papaya','Cabbage','Bitter_Gourd',
 'Cauliflower','Bean','Cucumber','Potato']
    predicted_vegetable=(categories[ind])
      

    messages = [
        {"role": "system", "content": "You are a vegetable expert."},
        {"role": "user", "content": f"Tell me about the nutrients (also in grams) in {predicted_vegetable}."},
        {"role": "assistant", "content": ""},  
        {"role": "user", "content": f"What are some recipes using {predicted_vegetable}?"}, 
        {"role": "assistant", "content": ""},  
    ]

    
    for turn in range(2, len(messages), 2):
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=messages[:turn + 1],  
        )
        chat_response = response.choices[0].message["content"].strip()
        
        messages[turn]["content"] = chat_response

    
    nutrient_info = messages[2]["content"]
    recipe_suggestions = messages[4]["content"]

    return f"Predicted Vegetable: {predicted_vegetable}\nNutrient Information: {nutrient_info}\nRecipe Suggestions: {recipe_suggestions}"

iface = gr.Interface(
    fn=predict_vegetable_and_chat,
    inputs=gr.inputs.Image(type="pil"),  
    outputs="text",
    title="Vegetable Recognition & Info",
    description="Upload an image of a vegetable for recognition and ask ChatGPT for nutrient info and recipes.",
)

iface.launch()
