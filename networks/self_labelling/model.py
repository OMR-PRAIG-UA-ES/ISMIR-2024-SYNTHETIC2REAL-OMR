from copy import deepcopy

import torch
from torch.nn import CTCLoss

from networks.base.model import CTCTrainedCRNN
from my_utils.augmentations import AugmentStage
from my_utils.data_preprocessing import preprocess_image_from_file


class SLTrainedCRNN(CTCTrainedCRNN):
    def __init__(
        self,
        train_dataloader,
        src_checkpoint_path,
        ytest_i2w,
        confidence_threshold=0.9,
        use_augmentations=True,
    ):
        super(SLTrainedCRNN, self).__init__(w2i={}, i2w={})
        # Save hyperparameters
        self.save_hyperparameters()
        ##################### Initialization constants:
        # Dataloader
        self.train_dataloader = train_dataloader
        # Source model checkpoint path
        self.src_checkpoint_path = src_checkpoint_path
        # Target dictionary
        self.ytest_i2w = ytest_i2w
        # Initialize source model
        self.initialize_src_model()
        # Initialize self-labelled dataset
        self.XFiles_Left = deepcopy(self.train_dataloader.dataset.X)
        self.YFiles_Left = deepcopy(self.train_dataloader.dataset.Y)
        self.perform_self_labelling()
        ##################### CTC-training constants:
        # Augmentations
        self.augment = AugmentStage() if use_augmentations else lambda x: x
        # Loss
        self.compute_ctc_loss = CTCLoss(
            blank=len(self.w2i), zero_infinity=True
        )  # The target index cannot be blank!
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
        ##################### Prediction constants:
        self.Y = []
        self.YHat = []
        ##################### Summary:
        self.summary()

    def initialize_src_model(self):
        # 1) Load source model
        print(f"Loading source model from {self.src_checkpoint_path}")
        src_model = CTCTrainedCRNN.load_from_checkpoint(
            self.src_checkpoint_path, ytest_i2w=self.ytest_i2w
        )
        # 2) Deep copy the source model
        self.model = deepcopy(src_model.model)
        # 3) Deep copy the source model's dictionaries
        self.w2i = deepcopy(src_model.w2i)
        self.i2w = deepcopy(src_model.i2w)
        # 4) Delete the source model
        for src_param, tgt_param in zip(
            src_model.model.parameters(), self.model.parameters()
        ):
            assert torch.all(
                torch.eq(src_param, tgt_param)
            ), "Source model and target model parameters are not equal"
        del src_model
        print("Source model ready!")

    def perform_self_labelling(self):
        keep = []
        leave = []
        for xpath, ypath in zip(self.XFiles_Left, self.YFiles_Left):
            # Preprocess image
            x = preprocess_image_from_file(xpath)
            x = x.unsqueeze(0)  # Add batch dimension
            # Model pass
            yhat = self.model(x)[0]  # yhat.shape = [frames, vocab_size]
            # Get most probable transcript and its probability values
            yhat = yhat.softmax(dim=-1)
            yhat_prob, yhat_indices = yhat.topk(k=1, dim=-1, sorted=False)
            # Get overall confidence (avg. of all probabilities)
            yhat_confidence = yhat_prob.mean()
            # If the confidence is high enough, keep the sample
            if yhat_confidence >= self.confidence_threshold:
                # Decode the most probable transcript:
                # - Merge repeated elements
                y_pred_decoded = torch.unique_consecutive(
                    yhat_indices.flatten(), dim=0
                ).tolist()
                # - Convert to string; len(i2w) -> CTC-blank
                y_pred_decoded = [
                    self.i2w[i] for i in y_pred_decoded if i != len(self.i2w)
                ]
                # Append to keep list
                keep.append((xpath, ypath, y_pred_decoded))
            else:
                # Append to leave list
                leave.append((xpath, ypath))
        # Update dataset
        self.train_dataloader.dataset.X.clear()
        self.train_dataloader.dataset.Y.clear()
        for xpath, ypath, y_pred_decoded in keep:
            self.train_dataloader.dataset.X.append(xpath)
            ypath = ypath.replace(".txt", "_self_labelled.txt")
            self.train_dataloader.dataset.Y.append(ypath)
            with open(ypath, "w") as f:
                f.write(" ".join(y_pred_decoded))
        # Update left samples
        self.XFiles_Left = [x[0] for x in leave]
        self.YFiles_Left = [x[1] for x in leave]

        print(
            f"Using {len(keep)} self-labelled samples out of {len(keep)+len(leave)} total possible for training"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def on_train_epoch_end(self):
        # Perform self-labelling if there are any samples left
        if self.XFiles_Left:
            self.perform_self_labelling()
