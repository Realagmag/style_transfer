import torch
from PIL import Image
from torchvision import transforms

def load_image(path):
    image = Image.open(path).convert("RGB")
    in_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return in_transform(image).unsqueeze(0)


def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)


def get_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features