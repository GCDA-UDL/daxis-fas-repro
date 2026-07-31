import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    vit_b_16,
    vit_b_32, 
    resnet50, 
    resnet152, 
    resnext50_32x4d,
    resnext101_64x4d
)

from torchvision.models.video import r2plus1d_18


class BaseModelWrapper(nn.Module):
    def __init__(self, model_fn, output_attr, num_classes, pretrained=False):
        super().__init__()
        # self.model = model_fn(weights=None)
        self.model = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
        self._adjust_output_layer(output_attr, num_classes)

    def _adjust_output_layer(self, output_attr, num_classes):
        module = self.model
        for part in output_attr[:-1]:
            module = getattr(module, part)
        last_name = output_attr[-1]
        last_layer = getattr(module, last_name)
        new_layer = nn.Linear(last_layer.in_features, num_classes)
        setattr(module, last_name, new_layer)

    def forward(self, x):
        return self.model(x)


class R2Plus1D18(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ViTB16(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(vit_b_16, ["heads", "head"], num_classes, pretrained)
        
class ViTB32(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(vit_b_32, ["heads", "head"], num_classes, pretrained)


class ResNet50(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(resnet50, ["fc"], num_classes, pretrained)


class ResNet152(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(resnet152, ["fc"], num_classes, pretrained)

class ResNeXt50(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(resnext50_32x4d, ["fc"], num_classes, pretrained)
        
class ResNeXt101(BaseModelWrapper):
    def __init__(self, num_classes, pretrained):
        super().__init__(resnext101_64x4d, ["fc"], num_classes, pretrained)


class ModelFactory:
    MODEL_MAP = {
        "vit_b_32": ViTB32,
        "vit_b_16": ViTB16,
        "resnet50": ResNet50,
        "resnet152": ResNet152,
        "resnext50_32x4d": ResNeXt50,
        "resnext101_64x4d": ResNeXt101,
        "r2plus1d_18": R2Plus1D18
    }

    @staticmethod
    def list_model_names():
        return list(ModelFactory.MODEL_MAP.keys())

    @staticmethod
    def create(model_name: str, num_classes: int, device="cpu", pretrained=False, multiGPU=False):
        print(f"ModelFactory->create: Pretrained: {pretrained}")
        if model_name not in ModelFactory.MODEL_MAP:
            raise ValueError(f"Modelo «{model_name}» no está soportado.")
        model_cls = ModelFactory.MODEL_MAP[model_name]
        # model = model_cls(num_classes)
        model = model_cls(num_classes, pretrained=pretrained)

        if multiGPU:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            model = torch.nn.DataParallel(model)
            
        return model.to(device)

def is_3d_model(model_name):
    models_3d = ["r2plus1d_18", "mc3_18", "r3d_18"
                 , "r2plus1d_18_2c", "mc3_18_2c", "r3d_18_2c"
                 , "r2plus1d_18_3c", "mc3_18_3c", "r3d_18_3c"
                 , "r2plus1d_18_6c", "mc3_18_6c", "r3d_18_6c"
                 , "r2plus1d_18_15c", "mc3_18_15c", "r3d_18_15c"
                 ]
    return model_name in models_3d

# class ModelFactory:
#     """Factory class to create models with adjusted final layers for given num_classes."""
    
#     @staticmethod
#     def resnet50(num_classes: int, device: torch.device):
#         model = models.resnet50(pretrained=True)  # Carga modelo ResNet-50 preentrenado
#         # Reemplaza la capa final fully-connected por una nueva con num_classes salidas
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model.to(device)
    
#     @staticmethod
#     def resnet152(num_classes: int, device: torch.device):
#         model = models.resnet152(pretrained=True)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model.to(device)
    
#     @staticmethod
#     def resnext50_32x4d(num_classes: int, device: torch.device):
#         model = models.resnext50_32x4d(pretrained=True)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model.to(device)
    
#     @staticmethod
#     def resnext101_64x4d(num_classes: int, device: torch.device):
#         model = models.resnext101_64x4d(pretrained=True)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#         return model.to(device)
    
#     @staticmethod
#     def vit_b_16(num_classes: int, device: torch.device):
#         model = models.vit_b_16(pretrained=True)
#         # Reemplaza la cabeza de clasificación del ViT
#         model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
#         return model.to(device)
    
#     @staticmethod
#     def vit_b_32(num_classes: int, device: torch.device):
#         model = models.vit_b_32(pretrained=True)
#         model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
#         return model.to(device)