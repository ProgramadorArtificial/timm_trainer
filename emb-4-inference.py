"""
Script used to do inferences using trained embedding model
"""
import torch
from PIL import Image

from utils.utils import default_transforms


class InferenceEmbedding:
    def __init__(self, model_path, is_float16=True, threshold=None, is_hook=False):
        """
        Load model and configurations
        Args:
            model_path (str): Path to the model
            is_float16 (bool, optional): Whether to use float16 inference mode. Defaults to True
            threshold (float, optional): Threshold to use for inference. If None use threshold saved in model.
             Defaults to None
            is_hook (bool, optional): Whether to use hook inference mode (must configure the layer). Defaults to False
        """
        self.is_float16 = is_float16
        self.is_hook = is_hook
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(model_path)
        self.threshold = threshold
        if threshold is None:
            self.threshold = checkpoint['threshold']
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

        if self.is_hook:
            self.outputs = {}
            def hook(module, input, output):
                self.outputs['embedding'] = output

            # Layer to use (hook)
            self.model.head.pre_logits.register_forward_hook(hook)

    def inference(self, img1, img2) -> (str, list):
        """
        Compare images using embedding model
        Args:
            img1 (uint8): First image (BGR)
            img2 (uint8): Second image (BGR)
        Returns:
            (bool, float): If is the same class (if threshold is None, always return False) and distance
        """
        embeddings = []
        for img in [img1, img2]:
            img = self.transforms(Image.fromarray(img))
            with torch.inference_mode():
                if self.is_float16:
                    embedding = self.model(img.unsqueeze(0).to(self.device).half())
                else:
                    embedding = self.model(img.unsqueeze(0).to(self.device))

            if self.is_hook:
                embeddings.append(self.outputs['embedding'].cpu().detach())
            else:
                embeddings.append(embedding.cpu().detach())

        # Euclidian distance
        dist = round(float(torch.nn.functional.pairwise_distance(embeddings[0], embeddings[1])), 4)

        if self.threshold is not None:
            return dist <= self.threshold, dist
        else:
            return False, dist


"""
# ### Usage example
import cv2

comparator = InferenceEmbedding(model_path='checkpoints/20250116-120128/best_45.pth', is_float16=True, threshold=None)

img1 = cv2.imread('dataset/face-recognition-dataset/test/Anushka Sharma/Anushka Sharma_0.jpg')
img2 = cv2.imread('dataset/face-recognition-dataset/test/Anushka Sharma/Anushka Sharma_1.jpg')
#img2 = cv2.imread('dataset/face-recognition-dataset/test/Marmik/Marmik_1.jpg')

result = comparator.inference(img1, img2)
print(result)
"""