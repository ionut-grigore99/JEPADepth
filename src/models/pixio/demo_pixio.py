from PIL import Image
from torchvision import transforms

from src.models.pixio.pixio import pixio_vith16

model = pixio_vith16(pretrained="your/checkpoint/path")

# you can try larger resolution, but ensure both sides are divisible by 16
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=3), # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = Image.open("assets/pixio_demo_image.jpg").convert("RGB")
img = transform(img)

# block-wise features containing class tokens and patch tokens
features = model(img.unsqueeze(0))