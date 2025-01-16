"""
Script used to do inferences using trained classification model
"""
import torch
from PIL import Image

from utils.utils import default_transforms


class InferenceClassification:
    def __init__(self, model_path, is_float16=True):
        """
        Load model and configurations
        Args:
            model_path (str): Path to the model
            is_float16 (bool, optional): Whether to use float16 inference mode. Defaults to True
        """
        self.is_float16 = is_float16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(model_path)
        self.id_to_class = checkpoint['id_to_label']
        self.model = checkpoint['model']
        if self.is_float16:
            self.model.to(self.device).half()
        else:
            self.model.to(self.device).float()
        self.model.eval()

        self.transforms = default_transforms(
            mean=checkpoint['mean'],
            std=checkpoint['std'],
            image_size=checkpoint['image_size'],
        )

    def inference(self, img) -> (str, list):
        """
        Classify an image using a trained model
        Args:
            img (uint8): Image to be classified (BGR)
        Returns:
            (str, float): Class name and confidence score for the class
        """
        img = self.transforms(Image.fromarray(img))
        with torch.inference_mode():
            if self.is_float16:
                outputs = self.model(img.unsqueeze(0).to(self.device).half())
            else:
                outputs = self.model(img.unsqueeze(0).to(self.device))

        confidences = list(torch.nn.functional.softmax(outputs[0], dim=0).cpu().detach().numpy())
        confidence = max(confidences)
        cls = self.id_to_class[confidences.index(max(confidences))]

        return cls, confidence


"""
# ### Usage example
import cv2

classification = InferenceClassification(model_path='checkpoints/20240806-191524/best_97.pth', is_float16=True)

img = cv2.imread('dataset/100_sports_image_classification/test/air hockey/1.jpg')
result = classification.inference(img)
print(result)
"""
