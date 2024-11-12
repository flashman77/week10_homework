import cv2 as cv
import numpy as np

def Lines(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # yellow color range
    lower_yellow = np.array([15, 180, 180])
    upper_yellow = np.array([150, 255, 255])
    
    # white color range 
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    # create masks and result
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv.inRange(hsv, lower_white, upper_white)

    result_yellow = cv.bitwise_and(image, image, mask=mask_yellow)
    result_white = cv.bitwise_and(image, image, mask=mask_white)

    # for white line
    contours, _ = cv.findContours(mask_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    all_points = np.vstack([contours[i] for i in range(len(contours))])
    hull = cv.convexHull(all_points)

    mask_outline = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv.drawContours(mask_outline, [hull], -1, 255, thickness=cv.FILLED)

    inside_mask = cv.bitwise_not(mask_outline)
    result_inside = cv.bitwise_and(result_white, result_white, mask=mask_outline)

    # final
    final_result = cv.add(result_yellow, result_inside)

    return final_result

image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
images = [cv.imread(image_file) for image_file in image_files]

Line_images = [Lines(img) for img in images]

for i, Line_image in enumerate(Line_images):
    cv.imshow(f'Image{i+1}', Line_image)

cv.waitKey(0)
cv.destroyAllWindows()