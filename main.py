# Import các thư viện và module cần thiết
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from License_plate_recognition import License_plate_recognition
from PyQt5.QtCore import Qt

# Import file UI được tạo ra tự động
from mainGUI import Ui_LicensePlateRecognition

# Định nghĩa lớp ứng dụng nhận diện biển số xe
class LicensePlateRecognitionApp(QMainWindow, Ui_LicensePlateRecognition):
    def __init__(self):
        super().__init__()

        # Thiết lập giao diện người dùng
        self.setupUi(self)

        # Kết nối sự kiện nhấn nút với một hàm xử lý
        self.pushButton.clicked.connect(self.loadImage)

    # Chuyển mảng hình ảnh sang QPixmap
    def array_to_pixmap(self, array):
        height, width, channel = array.shape
        bytes_per_line = 3 * width
        q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)

        return pixmap

    # Hàm xử lý sự kiện khi nhấn nút chọn ảnh
    def loadImage(self):
        # Mở hộp thoại chọn tệp với bộ lọc chỉ cho phép chọn ảnh
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        selected_file, _ = file_dialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg *.bmp)",
                                                       options=options)

        if selected_file:
            # Tạo QPixmap từ đường dẫn tệp được chọn
            pixmap = QPixmap(selected_file)

            # Hiển thị ảnh trên QLabel
            self.lblOriginImage.setPixmap(pixmap.scaled(self.lblOriginImage.size(), Qt.KeepAspectRatio))

            # Tạo một đối tượng nhận diện biển số xe
            LPR = License_plate_recognition()
            plate, character_zone, plate_string = LPR.recognize(selected_file)

            # Hiển thị kết quả lên giao diện
            if (len(plate) >= 1):
                self.lblLicensePlate.setPixmap(self.array_to_pixmap(plate[0]).scaled(self.lblLicensePlate.size(), Qt.KeepAspectRatio))
                self.lblCharacterZone.setPixmap(self.array_to_pixmap(character_zone[0]).scaled(self.lblCharacterZone.size(), Qt.KeepAspectRatio))
                self.txtLicensePlateCode.setText(plate_string[0])

# Khởi tạo ứng dụng và hiển thị cửa sổ chính
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = LicensePlateRecognitionApp()
    mainWindow.show()
    sys.exit(app.exec_())
