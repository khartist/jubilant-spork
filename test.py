import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from simple_clip import CLIP
from simple_clip.utils import accuracy, get_dataset, get_image_encoder, get_text_encoder
from simple_clip.custom_datasets.clip_datasets import get_image_tranforms


def validate_pretrained(model_path, args, device):
    """
    Validates a pre-trained CLIP model.

    Args:
        model_path (str): Path to the pre-trained model weights.
        args: Arguments containing dataset and model configuration.
        device (torch.device): Device to run validation on.
    """

    # Load the model
    image_encoder = get_image_encoder(args.image_encoder_name)
    text_encoder = get_text_encoder(args.text_encoder_name)
    model = CLIP(image_encoder, text_encoder)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Prepare the validation dataset and dataloader
    transforms_inference = get_image_tranforms(
        (args.image_size, args.image_size))
    ds_val = get_dataset(args.text_encoder_name,
                         args.dataset_name,
                         args.dataset_path,
                         transforms=transforms_inference,
                         split="test")  # Use "test" split for validation
    val_loader = DataLoader(ds_val,
                            batch_size=min(args.batch_size, 256),
                            num_workers=4)

    top1_acc_images = []
    top5_acc_images = []
    top1_acc_texts = []
    top5_acc_texts = []

    with torch.no_grad():  # Disable gradient calculation during validation
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ['input_ids', 'attention_mask', 'image']
            }

            logits = model(**batch)  # Get the model's output (logits)
            print(len(logits))

            labels = torch.arange(logits.size(0)).to(
                device)  # Create target labels
            top1, top5 = accuracy(logits, labels, topk=(1, 5))  # Calculate accuracy

            top1_acc_images.append(top1[0].item())
            top5_acc_images.append(top5[0].item())

            top1, top5 = accuracy(logits.t(), labels, topk=(1, 5))
            top1_acc_texts.append(top1[0].item())
            top5_acc_texts.append(top5[0].item())

    print("#" * 100)
    print('Pretrained eval acc/top1 images', np.mean(top1_acc_images))
    print('Pretrained eval acc/top5 images', np.mean(top5_acc_images))
    print('Pretrained eval acc/top1 texts', np.mean(top1_acc_texts))
    print('Pretrained eval acc/top5 texts', np.mean(top5_acc_texts))


if __name__ == "__main__":
    # Set the device to run the validation on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the arguments
    class Args:
        image_encoder_name = 'mobile_net_v3_small'
        text_encoder_name = 'phobert-base'
        dataset_name = 'coco'
        dataset_path = './data'
        batch_size = 256
        image_size = 224

    args = Args()

    # Path to the pre-trained model
    model_path = './models/clip_model.pth'

    # Validate the pre-trained model
    validate_pretrained(model_path, args, device)