import torch
import cv2
import numpy as np
import pandas as pd
import argparse
import sys
import os
import sys
import shutil
import onnx
from onnx2pytorch import ConvertModel
from extra_utils import vis_tensor_and_save, prep_image

from keypoint_net import KeypointNet

def main(model,img,img_size,output,flip,rotate):

    output_path = output

    model_path = model

    model_filepath = model_path

    image_path = img

    image_filepath = image_path

    img_name = '_'.join(image_filepath.split('/')[-1].split('.')[0].split('_')[-5:])

    image_size = (img_size, img_size)

    image = cv2.imread(image_filepath)
    h, w, _ = image.shape

    image = prep_image(image=image,target_image_size=image_size)
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')

    model = KeypointNet()
    if model_path[-1] == 't':
        model.load_state_dict(torch.load(model_filepath).get('model'))
    elif model_path[-1] == 'x':
        model.load_state_dict(ConvertModel(onnx.load(model_filepath)).state_dict()) #.get('model'))
    model.eval()
    output = model(image)
    out = np.empty(shape=(0, output[0][0].shape[2]))        # Out is the image with the points marked
    for o in output[0][0]:
        chan = np.array(o.cpu().data)
        cmin = chan.min()
        cmax = chan.max()
        chan -= cmin
        chan /= cmax - cmin
        out = np.concatenate((out, chan), axis=0)
    keypoints = output[1][0].cpu().data     # keypoints for the given image

    # print(cv2.imwrite(os.path.join(output_path, str(img_name + "_hm.jpg")), out * 255))
    # print(f'please check the output image here: {output_path + img_name + "_hm.jpg", out * 255}')
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    print('\nPlease check the output image here: ',(output_path + '\\' + str(img_name.split("\\")[-1]) + "_inference.jpg"))

    image = cv2.imread(image_filepath)
    h, w, _ = image.shape

    out_img, points_gop = vis_tensor_and_save(image=image, h=h, w=w, tensor_output=output[1][0].cpu().data, image_name=img_name.split("\\")[-1], output_uri=output_path)

    path_hehe = 'D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\october-2021-experiments\\geo_loss_train\\model_predictions.xlsx' #output path
    OUTPUT_XL = pd.read_excel(path_hehe)
    OUTPUT_XL.loc[len(OUTPUT_XL.index)] = points_gop
    OUTPUT_XL.drop_duplicates(subset=['Unnamed: 0'], keep='last',  inplace=True, ignore_index=True)
    OUTPUT_XL.to_excel(path_hehe, index=False)
    print(OUTPUT_XL)
    print("-------------------------------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoints Visualization')
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--img', help='path to single image', type=str, default="gs://mit-dut-driverless-external/ConeColourLabels/vid_3_frame_22063_0.jpg")
    parser.add_argument('--img_size', help='image size', default=80, type=int)
    parser.add_argument('--output', help='path to upload the detection', default="D:\\Python Projects\\Rektnet\\MITRepo\\RektNet\\outputs\\visualization")

    add_bool_arg('flip', default=False, help='flip image')
    add_bool_arg('rotate', default=False, help='rotate image')

    args = parser.parse_args(sys.argv[1:])

    main(model=args.model,img=args.img,img_size=args.img_size,output=args.output,flip=args.flip,rotate=args.rotate)
