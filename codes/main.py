
## 라이브러리 추가하기
import argparse

from train import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default= 1e-4, type=float, dest="lr")
#default: 1e-4 for scnas, 2e-4 for unet
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

# parser.add_argument("--data_dir", default="./../datasets/img_align_celeba", type=str, dest="data_dir")
# parser.add_argument("--data_dir", default="./../../datasets/facades", type=str, dest="data_dir")
# parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
# parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
# parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--data_dir", default="J:/MAYO_DEMO/dataset", type=str, dest="data_dir")
# "C:/Users/bmol/Desktop/JYpark/000. 3D RI2BF/SERVER_pix2pix/0517_HTX_pix2pix_test"
parser.add_argument("--ckpt_dir", default="J:/MAYO_DEMO/model/ckpt", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="J:/MAYO_DEMO/model/log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="J:/MAYO_DEMO/results", type=str, dest="result_dir")


#parser.add_argument("--task", default="pix2pix", choices=['DCGAN', 'pix2pix'], type=str, dest="task")
#parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')

parser.add_argument("--ny", default=1024, type=int, dest="ny")
parser.add_argument("--nx", default=1024, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt", default=1e2, type=float, dest="wgt")
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--network", default="pix2pix", choices=['DCGAN', 'pix2pix', 'scnas'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

parser.add_argument("--gpu_ids", default=[0], type=list, dest="gpu_ids")


# parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument("--gpus", default='3,4,5,6', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)