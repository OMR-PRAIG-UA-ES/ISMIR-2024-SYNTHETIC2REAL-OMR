import os

import cv2
import fire
import torch
import numpy as np

from networks.base.model import CTCTrainedCRNN
from networks.amd.da_model import DATrainedCRNN
from my_utils.data_preprocessing import preprocess_image_from_file

GRAD_CAM = "analysis/grad_cam_outputs"
os.makedirs(GRAD_CAM, exist_ok=True)


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Implementation adapted from:  https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter09_part03_interpreting-what-convnets-learn.ipynb
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        super(GradCAM, self).__init__()
        self.model = model
        self.device = device

    def make_gradcam(self, img_path: str) -> torch.Tensor:
        # Preprocess input image
        x = preprocess_image_from_file(img_path)
        x = x.unsqueeze(0).to(self.device)

        # 1. Retrieving the gradients of the top predicted class:
        # Encoder (CNN)
        ypred_encoder = self.model.encoder(x)
        ypred_encoder = ypred_encoder.requires_grad_(True)  # Detach the tensor before calculating gradients
        # Prepare for RNN
        _, _, _, w = ypred_encoder.size() # 1, 128, 64 / 16 = 4, w
        ypred = ypred_encoder.permute(0, 3, 1, 2).contiguous()
        ypred = ypred.reshape(1, w, self.model.decoder_input_size)
        # Decoder (RNN) -> ypred.shape = (batch_size, seq_len, num_classes)
        ypred = self.model.decoder(ypred)
        # Extract the top class channels
        top_class_channels = torch.topk(ypred, k=1, dim=-1, sorted=False).values.squeeze()
        # Compute gradients
        torch.autograd.backward(top_class_channels, torch.ones_like(top_class_channels))
        # Extract gradients with respect to encoder output
        grads = ypred_encoder.grad

        # 2. Gradient pooling and channel-importance weighting:
        pooled_grads = torch.mean(grads, dim=(0, 2, 3)).detach() # pooled_grads.shape = (128,)
        ypred_encoder = ypred_encoder.detach()
        # Multiply pooled gradients with encoder output
        for i in range(ypred_encoder.shape[1]):
            ypred_encoder[:, i, :, :] *= pooled_grads[i]
        heatmap = torch.mean(ypred_encoder, dim=1).squeeze()

        # 3. Heatmap postprocessing:
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)
        return {"heatmap": heatmap, "img": x[0].permute(1, 2, 0)}

    def get_and_save_gradcam_heatmap(self, img_path: str, grad_img_output_path: str):
        # Apply Grad-CAM
        gc_output = self.make_gradcam(img_path=img_path)
        heatmap = gc_output["heatmap"].cpu().numpy()
        img = gc_output["img"].repeat(1, 1, 3).cpu().numpy()
        
        # Combine heatmap with original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img_cam = heatmap + img
        img_cam = img_cam / np.max(img_cam)
        # img_cam = cv2.cvtColor(img_cam,cv2.COLOR_RGB2BGR)

        # Save Grad-CAM image
        cv2.imwrite(grad_img_output_path, np.uint8(255 * img_cam))


def run_grad_cam(checkpoint_path: str, img_path: str):
    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    try:
        model = CTCTrainedCRNN.load_from_checkpoint(checkpoint_path).model
    except Exception:
        try:
            model = DATrainedCRNN.load_from_checkpoint(checkpoint_path).model
        except Exception:
            raise ValueError(
                f"Could not load model from checkpoint path {checkpoint_path}"
            )
    for param in model.parameters():
        param.requires_grad = False

    # Grad-CAM
    grad_cam = GradCAM(model=model.to(device), device=device)
    output_path = os.path.join(GRAD_CAM, "-".join(img_path.split("/")))
    grad_cam.get_and_save_gradcam_heatmap(
        img_path=img_path, grad_img_output_path=output_path
    )


if __name__ == "__main__":
    fire.Fire(run_grad_cam)
