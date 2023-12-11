import cv2
import numpy as np

# Định nghĩa kích thước bộ lọc Gaussian và các tham số khác
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


# Hàm tiền xử lý ảnh
def preprocess(imgOriginal):
    # Trích xuất kênh giá trị từ ảnh gốc
    imgGrayscale = extractValue(imgOriginal)

    # Tăng cường độ tương phản của ảnh giá trị
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape

    # Làm mờ ảnh với bộ lọc Gaussian
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    # Chuyển đổi ảnh thành ảnh nhị phân sử dụng ngưỡng tương thích thích ứng
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh


# Hàm trích xuất giá trị từ ảnh gốc
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    # Tách các kênh màu từ không gian màu HSV
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue


# Hàm làm cho độ tương phản lớn nhất
def maximizeContrast(imgGrayscale):
    # Lấy kích thước của ảnh
    height, width = imgGrayscale.shape

    # Tạo các ảnh top hat và black hat để nổi bật chi tiết sáng và tối
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tạo bộ lọc kernel

    # Thực hiện phép tophat và blackhat để nổi bật chi tiết sáng và tối
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)

    # Kết hợp ảnh gốc với top hat và sau đó trừ đi black hat
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
