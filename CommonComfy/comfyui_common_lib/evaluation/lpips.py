import lpips
import torch

class LPIPS:
    def __init__(self):
        # Instantiate LPIPS loss with VGG network
        self.lpips_loss = lpips.LPIPS(net='vgg')

    def calculate_lpips(self, img1, img2):
        return self.lpips_loss(img1, img2)

if __name__ == "__main__":
    lpips_loss = LPIPS()
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    print(lpips_loss.calculate_lpips(img1, img2))
