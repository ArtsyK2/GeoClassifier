import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Artsyk2/Geo_Classifier",
    filename="GeoClaas_v01.keras"
)
model = load_model(model_path)

def classify_image(inp):
    inp = tf.image.resize(inp, (150, 150))  # resize
    inp = inp / 255.0                        # normalize
    inp = np.expand_dims(inp, axis=0)        # batch dimension
    pred = model.predict(inp)
    class_index = pred.argmax()
    class_labels = {'Building' : 0,'Forest' : 1, 'Glacier' : 2, 'Mountain' : 3, 'Sea' : 4, 'Street' : 5}
    class_labels = {v: k for k, v in class_labels.items()}
    return 'Given Image is : ' + class_labels[class_index]

desc = 'Classify the given image into one of the six categories: Buildings, Forest, Glacier, Mountain, Sea, Street'

demo = gr.Interface(fn=classify_image, 
                    inputs=gr.Image(
                        label = 'Upload Image',
                        type = 'numpy',
                    ),
                    outputs = 'text',
                    title = 'Image Classification',
                    description = desc,
)

demo.launch()
