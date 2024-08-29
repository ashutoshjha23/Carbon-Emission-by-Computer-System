import timm

def load_vit_model(pretrained=True, num_classes=1000):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model
