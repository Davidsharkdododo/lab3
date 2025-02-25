import cv2
import numpy as np
hsv_ranges = {
            "blue": (np.array([100, 80, 130]), np.array([140, 255, 255])),
            "red":  (np.array([0, 70, 150]),   np.array([10, 255, 255])),
            "green": (np.array([50, 40, 100]),   np.array([80, 255, 255])),
            #"yellow": (np.array([20, 70, 100]), np.array([30, 255, 255]))
        }
camera_matrix = np.array([[324.29, 0, 308.7],
                                       [0, 322.679, 215.88],
                                       [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.array([-0.31, 0.071, -0.002, 0.002, 0], dtype=np.float32)
def undistort_image(image):
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        return undistorted

def preprocess_image(image):
    resized = cv2.resize(image, (640, 480))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred
def detect_lane_color(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_color = None
        output = image.copy()
        for color, (lower, upper) in hsv_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Choose the largest contour assuming it corresponds to the lane marking.
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:  # Threshold to filter noise
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(output, color, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    detected_color = color
        return output, detected_color

if __name__ == '__main__':
    image = cv2.imread('/home/roboticslab229r/Downloads/Mobile-Robotics-3/packages/my_package/src/green.png')
    image1 = undistort_image(image)
    image2 = preprocess_image(image1)
    output, color = detect_lane_color(image2)
    print(color)
    cv2.imshow("Detected Lane Color", output)
    cv2.waitKey(0)  # Wait until any key is pressed
    cv2.destroyAllWindows()
