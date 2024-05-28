from flask import Flask, render_template, request, send_file, redirect, url_for
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import os
import io
import base64
from io import BytesIO
import google.generativeai as genai
import threading
import subprocess

app = Flask(__name__)

genai.configure(api_key='AIzaSyB4hqhFg7WPiotLHiA0KOZMgD6gjjarG2U')
model = genai.GenerativeModel('gemini-pro-vision')

UPLOAD_FOLDER = 'static/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def process_inpaint(img, mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

    # Resize the mask to match the image size
    mask_resized = cv2.resize(mask_gray, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # Create a binary mask
    _, mask_binary = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
    
    # Apply inpainting
    inpainted_img = cv2.inpaint(img_rgb, mask_binary, 3, cv2.INPAINT_TELEA)
    
    return inpainted_img


def calculate_rule_of_thirds(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    third_width = width // 3
    third_height = height // 3
    intersection_points = [(third_width, third_height), (2 * third_width, third_height),
                           (third_width, 2 * third_height), (2 * third_width, 2 * third_height)]
    elements_near_intersections = 0

    for point in intersection_points:
        x, y = point
        region = img_cv[y - 10:y + 10, x - 10:x + 10]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if cv2.countNonZero(edges) > 0:
            elements_near_intersections += 1

    composition_score = elements_near_intersections / len(intersection_points)
    return composition_score

def calculate_color_balance(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_cv)
    avg_blue = np.mean(b)
    avg_green = np.mean(g)
    avg_red = np.mean(r)
    weight_blue = 0.3
    weight_green = 0.3
    weight_red = 0.4
    balance_score = (weight_blue * avg_blue + weight_green * avg_green + weight_red * avg_red) / 255.0
    return balance_score

def calculate_clarity_score(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = 100.0
    clarity_score = min(max(laplacian_var / threshold, 0), 1)
    return clarity_score

def calculate_contrast(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hist_l = cv2.calcHist([l], [0], None, [256], [0, 256])
    contrast = np.var(hist_l)
    max_contrast = 15000
    contrast_score = min(contrast / max_contrast, 1.0)
    return contrast_score

def calculate_saturation(image):
    img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    sat = img_hsv[:, :, 1]
    avg_sat = np.mean(sat)
    max_saturation = 255
    saturation_score = avg_sat / max_saturation
    return saturation_score

def calculate_brightness(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])
    max_brightness = 255
    brightness_score = brightness / max_brightness
    return brightness_score

def calculate_noise(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    noise = np.std(img_cv)
    max_noise = 255.0
    noise_score = 1 - (noise / max_noise)
    return noise_score

def calculate_aesthetic_score(scores):
    weights = {
        'Composition': 0.18,
        'Color Balance': 0.18,
        'Clarity': 0.14,
        'Contrast': 0.14,
        'Saturation': 0.14,
        'Brightness': 0.14,
        'Noise': 0.08
    }
    return (scores['Composition']*weights['Composition'] + scores['Color Balance']*weights['Color Balance'] +scores['Clarity']*weights['Clarity'] + scores['Contrast']*weights['Contrast'] + scores['Saturation']*weights['Saturation'] + scores['Brightness']*weights['Brightness'] - scores['Noise']*weights['Noise'])


def enhance_image(image, scores):
    thresholds = {
        'brightness': 0.8,
        'contrast': 0.65,
        'saturation': 0.6,
        'clarity': 0.7,
        'color_balance': 0.65,
        'noise': 0.65
    }

    enhanced_image = image

    if scores['Brightness'] < thresholds['brightness']:
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(1.5)

    if scores['Contrast'] < thresholds['contrast']:
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.5)

    if scores['Saturation'] < thresholds['saturation']:
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.5)

    if scores['Clarity'] < thresholds['clarity']:
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(2.0)

    if scores['Color Balance'] < thresholds['color_balance']:
        enhanced_image = ImageOps.autocontrast(enhanced_image, 3)

    if scores['Noise'] < thresholds['noise']:
        enhanced_image = Image.fromarray(cv2.fastNlMeansDenoisingColored(np.array(enhanced_image), None, 10, 10, 7, 21))

    return enhanced_image

def aesthetic_scores(scores):
    weights = {
        'Composition': 0.18,
        'Color Balance': 0.18,
        'Clarity': 0.14,
        'Contrast': 0.14,
        'Saturation': 0.14,
        'Brightness': 0.14,
        'Noise': 0.08
    }
    return (scores['Composition']*weights['Composition'] + scores['Color Balance']*weights['Color Balance'] + scores['Clarity']*weights['Clarity'] + scores['Contrast']*weights['Contrast'] + scores['Saturation']*weights['Saturation'] + scores['Brightness']*weights['Brightness'] - scores['Noise']*weights['Noise'])

# Run the Pygame game
def run_editor_function():
    subprocess.run(['python3', 'src/main.py'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        apply_inpainting = 'apply_inpainting' in request.form
        if file:
            original_image = Image.open(file).convert("RGBA")

            # Convert to RGB before saving as JPEG
            original_image_rgb = original_image.convert("RGB")
            original_image_path = os.path.join(UPLOAD_FOLDER, 'original.jpg')
            original_image_rgb.save(original_image_path, "JPEG")

            if apply_inpainting:
                canvas_data = request.form['canvas_data']
                
                canvas_data = canvas_data.split(',')[1]
                canvas_data = base64.b64decode(canvas_data)
                mask_image = Image.open(BytesIO(canvas_data)).convert("RGBA")
                
                # Ensure mask and image have the same dimensions
                mask_image = mask_image.resize(original_image.size)

                img_np = np.array(original_image)
                mask_np = np.array(mask_image)

                inpainted_image_np = process_inpaint(img_np, mask_np)
                inpainted_image = Image.fromarray(inpainted_image_np)

                # Convert to RGB before saving as JPEG
                enhanced_image = inpainted_image.convert("RGB")
            else:
                enhanced_image = original_image.convert("RGB")

            enhanced_image_path = os.path.join(UPLOAD_FOLDER, 'enhanced.jpg')
            enhanced_image.save(enhanced_image_path, "JPEG")

            scores = {
                'Composition': calculate_rule_of_thirds(enhanced_image),
                'Color Balance': calculate_color_balance(enhanced_image),
                'Clarity': calculate_clarity_score(enhanced_image),
                'Contrast': calculate_contrast(enhanced_image),
                'Saturation': calculate_saturation(enhanced_image),
                'Brightness': calculate_brightness(enhanced_image),
                'Noise': calculate_noise(enhanced_image)
            }
            
            scores = {
                'Aesthetic Score': aesthetic_scores(scores)*10,
                'Composition': scores['Composition']*10,
                'Color Balance': scores['Color Balance']*10,
                'Clarity': scores['Clarity']*10,
                'Contrast': scores['Contrast']*10,
                'Saturation': scores['Saturation']*10,
                'Brightness': scores['Brightness']*10,
                'Noise': scores['Noise']*10
            }

            enhanced_image = enhance_image(enhanced_image, scores)
            enhanced_image.save(enhanced_image_path, "JPEG")

            enhanced_scores = {
                'Composition': calculate_rule_of_thirds(enhanced_image),
                'Color Balance': calculate_color_balance(enhanced_image),
                'Clarity': calculate_clarity_score(enhanced_image),
                'Contrast': calculate_contrast(enhanced_image),
                'Saturation': calculate_saturation(enhanced_image),
                'Brightness': calculate_brightness(enhanced_image),
                'Noise': calculate_noise(enhanced_image)
            }
            
            enhanced_scores = {
                'Aesthetic Score': aesthetic_scores(enhanced_scores)*10,
                'Composition': enhanced_scores['Composition']*10,
                'Color Balance': enhanced_scores['Color Balance']*10,
                'Clarity': enhanced_scores['Clarity']*10,
                'Contrast': enhanced_scores['Contrast']*10,
                'Saturation': enhanced_scores['Saturation']*10,
                'Brightness': enhanced_scores['Brightness']*10,
                'Noise': enhanced_scores['Noise']*10
            }

            captions = ' '
            captions_image = Image.open(enhanced_image_path)
            try:
                response = model.generate_content(
                    ["Can you please generate 6 captions for me based on this image, Give me 2 Fun and Witty captions, 2 Formal and sophisticated and 2 that you think will fit this Image the best.", captions_image], stream=True
                )
                for chunk in response:
                    captions = captions + ' ' + chunk.text

            except Exception as e:
                print(e)

            return render_template('result.html', 
                                   original_image=original_image_path, 
                                   enhanced_image=enhanced_image_path, 
                                   original_scores=scores, 
                                   enhanced_scores=enhanced_scores,
                                   captions=captions)

    return render_template('index.html')

@app.route('/download')
def download():
    enhanced_image_path = request.args.get('image')
    return send_file(enhanced_image_path, as_attachment=True)

# Route to run the game
@app.route('/run_editor')
def run_editor():
    threading.Thread(target=run_editor_function).start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
