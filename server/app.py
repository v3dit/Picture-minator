from flask import Flask, render_template, request, redirect, send_file
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
from io import BytesIO
import base64


app = Flask(__name__)

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

    max_contrast = 15000  # Adjust this threshold as needed
    contrast_score = min(contrast / max_contrast, 1.0)
    return contrast_score

def calculate_saturation(image):
    img_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    sat = img_hsv[:, :, 1]
    avg_sat = np.mean(sat)

    max_saturation = 255  # Maximum saturation value
    saturation_score = avg_sat / max_saturation
    return saturation_score

def calculate_brightness(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])

    max_brightness = 255  # Maximum brightness value
    brightness_score = brightness / max_brightness
    return brightness_score

def calculate_noise(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Calculate noise using standard deviation of pixel intensities
    noise = np.std(img_cv)
    max_noise = 255.0  # Maximum noise level
    noise_score = 1 - (noise / max_noise)  # Invert the score, lower noise = higher score
    return noise_score

def enhance_image(image):
    enhanced_image = ImageOps.autocontrast(image, 3)
    return enhanced_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            original_image = Image.open(file)
            
            # Calculate scores for the original image
            comp_score = calculate_rule_of_thirds(original_image)
            bal_score = calculate_color_balance(original_image)
            clar_score = calculate_clarity_score(original_image)
            cont_score = calculate_contrast(original_image)
            sat_score = calculate_saturation(original_image)
            brightness_score = calculate_brightness(original_image)
            noise_score = calculate_noise(original_image)

            # Enhance the image
            enhanced_image = enhance_image(original_image)

            # Calculate scores for the enhanced image
            enhanced_comp_score = calculate_rule_of_thirds(enhanced_image)
            enhanced_bal_score = calculate_color_balance(enhanced_image)
            enhanced_clar_score = calculate_clarity_score(enhanced_image)
            enhanced_cont_score = calculate_contrast(enhanced_image)
            enhanced_sat_score = calculate_saturation(enhanced_image)
            enhanced_brightness_score = calculate_brightness(enhanced_image)
            enhanced_noise_score = calculate_noise(enhanced_image)

            # Combine scores for comparison
            original_scores = {
                'Composition': comp_score,
                'Color Balance': bal_score,
                'Clarity': clar_score,
                'Contrast': cont_score,
                'Saturation': sat_score,
                'Brightness': brightness_score,
                'Noise': noise_score
            }

            enhanced_scores = {
                'Composition': enhanced_comp_score,
                'Color Balance': enhanced_bal_score,
                'Clarity': enhanced_clar_score,
                'Contrast': enhanced_cont_score,
                'Saturation': enhanced_sat_score,
                'Brightness': enhanced_brightness_score,
                'Noise': enhanced_noise_score
            }

            return render_template('result.html', 
                                   original_image=file.read(), 
                                   enhanced_image=enhanced_image, 
                                   original_scores=original_scores, 
                                   enhanced_scores=enhanced_scores)

    return render_template('index.html')

@app.route('/download')
def download():
    enhanced_image = request.args.get('image')
    # Remove padding characters from the base64 string
    enhanced_image = enhanced_image.replace(' ', '+')
    enhanced_image_bytes = base64.b64decode(enhanced_image)
    return send_file(BytesIO(enhanced_image_bytes), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
