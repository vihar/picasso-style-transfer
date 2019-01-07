import argparse
import os
import sys
import re
import torch
from torchvision import transforms
import torch.onnx
import utils
from transformer_net import TransformerNet
from vgg import Vgg16


# Parser
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image_local(args.content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def main():
    # Argument Parser
    main_arg_parser = argparse.ArgumentParser(
        description="parser for fast-neural-style"
    )
    subparsers = main_arg_parser.add_subparsers(
        title="subcommands", dest="subcommand"
    )

    train_arg_parser = subparsers.add_parser(
        "train", help="parser for training arguments"
    )

    train_arg_parser.add_argument(
        "--style-image",
        type=str,
        default="images/style-images/mosaic.jpg",
        help="path to style-image"
    )

    train_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU"
    )

    eval_arg_parser = subparsers.add_parser(
        "eval", help="parser for evaluation/stylizing arguments")

    eval_arg_parser.add_argument(
        "--content-image",
        type=str,
        required=True,
        help="path to content image you want to stylize"
    )

    eval_arg_parser.add_argument(
        "--content-scale",
        type=float,
        default=None,
        help="factor for scaling down the content image"
    )

    eval_arg_parser.add_argument(
        "--output-image",
        type=str,
        required=True,
        help="path for saving the output image"
    )

    eval_arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="saved model to be used for stylizing the image."
    )

    eval_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU"
    )

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
