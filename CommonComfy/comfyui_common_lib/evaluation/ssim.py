import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# Hàm tính SSIM giữa hai hình ảnh
def calculate_ssim(imageA, imageB):
    # Chuyển đổi hai hình ảnh sang dạng xám
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Resize images to the same dimensions
    grayA = cv2.resize(grayA, (grayB.shape[1], grayB.shape[0]))

    # Tính toán SSIM
    score, _ = ssim(grayA, grayB, full=True)
    return score


# SSIM class module
class SSIM():

    def __call__(self, image_path_1, image_path_2):
        # Danh sách lưu trữ SSIM scores
        ssim_scores = []

        # Duyệt qua tất cả các tệp trong thư mục hình ảnh thực
        for filename in tqdm(os.listdir(image_path_1)):
            # Đọc hình ảnh thực và hình ảnh sinh ra tương ứng
            image_real = cv2.imread(os.path.join(image_path_1, filename))
            image_generated = cv2.imread(os.path.join(image_path_2, filename )) 

            # Kiểm tra xem hình ảnh sinh ra có tồn tại không
            if image_real is not None and image_generated is not None:
                # Tính toán SSIM
                ssim_score = calculate_ssim(image_real, image_generated)
                ssim_scores.append(ssim_score)
            else:
                print(f"Không tìm thấy ảnh sinh ra cho {filename}")

        # Tính trung bình SSIM
        average_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
        return average_ssim

if __name__ == "__main__":
    ssim = SSIM()
    print(ssim("test1", "test2"))
