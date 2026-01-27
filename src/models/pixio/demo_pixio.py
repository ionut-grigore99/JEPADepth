from PIL import Image
from torchvision import transforms
import lovely_tensors as lt

from src.models.pixio.pixio import pixio_vith16

if __name__ == "__main__":
    lt.monkey_patch()

    model = pixio_vith16(pretrained="weights/pixio/pixio_vith16.pth")

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
    print("Input image shape:", img.shape)
    print(f"Number of blocks extracted: {len(features)}")
    print(f"Patch tokens shape: {features[0]['patch_tokens'].shape}") # (1, num_patches, embed_dim)
    print(f"Class tokens shape: {features[0]['cls_tokens'].shape}") # (1, n_cls_tokens, embed_dim)