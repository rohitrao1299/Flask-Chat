from flask import Flask, render_template, request, send_file
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import io
import os

app = Flask(__name__)

# Define the prompt here
PROMPT = "You are a helpful assistant. Please provide blur,sketch,black and white TAT Images."

# Function to generate AI-based images using Stable Diffusion
def generate_images_using_stable_diffusion(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    image = pipe(prompt).images[0]
    return image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # No need to get the prompt from the request form
        # prompt = request.form['text']
        image_output = generate_images_using_stable_diffusion(PROMPT)
        img_io = io.BytesIO()
        image_output.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)