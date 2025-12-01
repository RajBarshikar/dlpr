print("Installing Tesseract-OCR...")
!sudo apt-get install tesseract-ocr


print("\nInstalling Python packages...")
!pip install pytesseract opencv-python


import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from pytesseract import Output

print("\nPlease upload the image you want to test:")
uploaded = files.upload()


img_name = list(uploaded.keys())[0]
print(f"Image '{img_name}' uploaded successfully.")



img = cv2.imread(img_name)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



_, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

print("\nImage preprocessing complete (grayscale + thresholding).")


extracted_text = pytesseract.image_to_string(img_thresh)
print("\n--- Extracted Text Only ---")
if not extracted_text:
    print("No text found.")
else:
    print(extracted_text)
print("---------------------------\n")


word_dict = pytesseract.image_to_data(img_thresh, output_type=Output.DICT)
n_boxes = len(word_dict['text'])
print(f"Found {n_boxes} potential word boxes.")

img_with_boxes = img_rgb.copy()

for i in range(n_boxes):

    if int(word_dict['conf'][i]) > 60:
        (x, y, w, h) = (word_dict['left'][i], word_dict['top'][i],
                        word_dict['width'][i], word_dict['height'][i])


        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)


        text = word_dict['text'][i]
        cv2.putText(img_with_boxes, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


print("Displaying image with bounding boxes...")
plt.figure(figsize=(10, 10))
plt.imshow(img_with_boxes)
plt.axis("off")
plt.title("OCR Text Detection with Bounding Boxes")
plt.show()