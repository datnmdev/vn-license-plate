# Import các thư viện cần thiết
import math
import cv2
import numpy as np

# Import module tùy chỉnh Preprocess
import Preprocess

# Định nghĩa một class cho việc nhận diện biển số xe
class License_plate_recognition(object):
    def __init__(self):
        # Các tham số cho việc ngưỡng c adapt
        self.ADAPTIVE_THRESH_BLOCK_SIZE = 19
        self.ADAPTIVE_THRESH_WEIGHT = 9

        # Biến đếm
        self.n = 1

        # Các tham số cho việc lọc kí tự dựa trên kích thước
        self.Min_char = 0.01
        self.Max_char = 0.09

        # Các tham số cho việc resize ảnh
        self.RESIZED_IMAGE_WIDTH = 20
        self.RESIZED_IMAGE_HEIGHT = 30

    # Phương thức cho việc nhận diện biển số
    def recognize(self, img_path):
        # Khởi tạo các danh sách rỗng và load ảnh đầu vào
        plate = []
        character_zone = []
        plate_string = []
        img = cv2.imread(img_path)

        # Load mô hình KNearest đã được huấn luyện từ các tệp văn bản
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        # Tiền xử lý ảnh
        imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation

        # Tìm contours trong ảnh đã dilate
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Khởi tạo danh sách để lưu contours của biển số xe tiềm ẩn
        screenCnt = []

        # Duyệt qua contours để tìm biển số xe tiềm ẩn
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            if (len(approx) == 4):
                screenCnt.append(approx)
                cv2.putText(img, "", (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        # Kiểm tra xem có biển số xe nào được tìm thấy hay không
        if screenCnt is None:
            detected = 0
            print("Không tìm thấy biển số xe")
        else:
            detected = 1

        # Xử lý biển số xe được tìm thấy
        if detected == 1:
            for screenCnt in screenCnt:
                # Vẽ contours xung quanh biển số xe
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                # Trích xuất bốn điểm góc của biển số xe
                (x1, y1) = screenCnt[0, 0]
                (x2, y2) = screenCnt[1, 0]
                (x3, y3) = screenCnt[2, 0]
                (x4, y4) = screenCnt[3, 0]
                array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                (x1, y1) = array[0]
                (x2, y2) = array[1]
                doi = abs(y1 - y2)
                ke = abs(x1 - x2)
                angle = math.atan(doi / ke) * (180.0 / math.pi)

                # Tạo một binary mask cho biển số xe
                mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

                # Lấy tọa độ của các pixel khác không trong mask
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                # Cắt ảnh vùng chứa biển số xe
                roi = img[topx:bottomx, topy:bottomy]
                imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
                ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                # Xác định ma trận xoay dựa trên hướng của biển số xe
                if x1 < x2:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                else:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                # Áp dụng xoay cho biển số xe và ảnh nhị phân
                roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
                roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                # Áp dụng các phép toán hình thái học để tăng cường việc nhận diện kí tự
                kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Vẽ contours xung quanh các kí tự
                cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)

                # Khởi tạo danh sách để lưu thông tin về kí tự
                char_x_ind = {}
                char_x = []
                height, width, _ = roi.shape
                roiarea = height * width

                # Duyệt qua contours để lọc kí tự dựa trên kích thước và tỉ lệ khía cạnh
                for ind, cnt in enumerate(cont):
                    (x, y, w, h) = cv2.boundingRect(cont[ind])
                    ratiochar = w / h
                    char_area = w * h

                    if (self.Min_char * roiarea < char_area < self.Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                        if x in char_x:
                            x = x + 1
                        char_x.append(x)
                        char_x_ind[x] = ind

                # Sắp xếp các kí tự dựa trên tọa độ x
                char_x = sorted(char_x)
                first_line = ""
                second_line = ""

                # Duyệt qua các kí tự đã sắp xếp để nhận diện và lưu chúng
                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Cắt ảnh vùng chứa kí tự
                    imgROI = thre_mor[y:y + h, x:x + w]

                    # Resize ảnh kí tự
                    imgROIResized = cv2.resize(imgROI, (self.RESIZED_IMAGE_WIDTH, self.RESIZED_IMAGE_HEIGHT))
                    npaROIResized = imgROIResized.reshape((1, self.RESIZED_IMAGE_WIDTH * self.RESIZED_IMAGE_HEIGHT))

                    # Chuyển đổi sang float32 và nhận diện kí tự bằng KNearest
                    npaROIResized = np.float32(npaROIResized)
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
                    strCurrentChar = str(chr(int(npaResults[0][0])))
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                    # Xác định xem kí tự ở dòng thứ nhất hay thứ hai của biển số xe
                    if (y < height / 3):
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                # Kiểm tra xem tổng số kí tự có đúng không cho một biển số xe
                if (len(first_line + second_line) == 8 or len(first_line + second_line) == 9):
                    # Thêm biển số xe và vùng chứa kí tự vào các danh sách tương ứng
                    plate.append(cv2.cvtColor(cv2.resize(img[topx:bottomx, topy:bottomy], None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
                    roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
                    character_zone.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    plate_string.append(first_line + second_line)

                # Tăng biến đếm
                self.n = self.n + 1

        # Trả kết quả dưới dạng mảng numpy
        return np.array(plate), np.array(character_zone), plate_string
