from transformers import ViTFeatureExtractor, ViTModel
import torch

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.feature_extractor = None
        self.init_feature_extractor()
        
    def init_feature_extractor(self):
        model_name = 'google/vit-base-patch16-224-in21k'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def extract_features(self, image):
        inputs = self.feature_extractor(
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return features