import os

import torch
import fire

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
        # Constants
        self.model = model
        self.device = device

    def make_gradcam(self, img_path: str):
        # Preprocess input image
        x = preprocess_image_from_file(img_path)
        x = x.unsqueeze(0).to(self.device)

        # 1. Retrieving the gradients of the top predicted class:
        # Encoder (CNN)
        ypred_encoder = self.model.encoder(x)
        ypred_encoder.requires_grad_(True)  # Watch for gradient calculation
        # Prepare for RNN
        _, _, _, w = ypred_encoder.size()
        ypred_encoder = ypred_encoder.permute(0, 3, 1, 2).contiguous()
        ypred_encoder = ypred_encoder.reshape(1, w, self.decoder_input_size)
        # Decoder (RNN) -> ypred.shape = (seq_len, num_classes)
        ypred = self.model.decoder(ypred_encoder)[0]
        # Extract the top class channels
        top_class_channels = ypred[:, torch.argmax(ypred, dim=1)]
        # Compute gradients
        top_class_channels.backward()
        # Extract gradients with respect to encoder output
        grads = ypred_encoder.grad

        # 2. Gradient pooling and channel-importance weighting:
        print(grads.shape)
        pooled_grads = torch.mean(grads, dim=(0, 1, 2)).cpu().numpy()
        ypred_encoder = ypred_encoder.cpu().numpy()[0]
        print(ypred_encoder.shape)
        # for i in range(pooled_grads.shape[-1]):
        #     last_conv_layer_output[:, :, i] *= pooled_grads[i]
        # heatmap = np.mean(last_conv_layer_output, axis=-1)

        # # 3. Heatmap postprocessing:
        # heatmap = np.maximum(heatmap, 0)
        # heatmap /= np.max(heatmap)
        # return heatmap

    def get_and_save_gradcam_heatmap(self, img_path: str, grad_img_output_path: str):
        pass
        #     # Make the heatmap
        # heatmap = self.make_gradcam(img_path=img_path)
        # Rescale heatmap
        # heatmap = np.uint8(255 * heatmap)

        # # Use the jet colormap to recolorize the heatmap
        # jet = cm.get_cmap("jet")
        # jet_colors = jet(np.arange(256))[:, :3]
        # jet_heatmap = jet_colors[heatmap]

        # #
        # jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        # jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        # superimposed_img = jet_heatmap * 0.4 + img
        # superimposed_img = keras.utils.array_to_img(superimposed_img)
        # save_path = "elephant_cam.jpg"
        # superimposed_img.save(save_path)


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
