import sys
sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *
import cv2
from net import *
from data import *
import argparse 
import matplotlib.pyplot as plt

def BCHW2colormap(tensor,earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result,axis=2)
    return result

def initialize(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    trans = transforms.Compose([transforms.ToTensor()])
    orig = cv2.imread(args.image_path)
    orig = cv2.resize(orig,(512,512))
    image = trans(orig.astype(np.float32)/255.)
    image = image.unsqueeze(0).to(device)
    # model
    model = DFPmodel()
    model.load_state_dict(torch.load(args.loadmodel))
    model.to(device)
    return device,orig,image,model

def post_process(rm_ind,bd_ind):
    hard_c = (bd_ind>0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind>0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask>=1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask//255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask,rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask*new_rm_ind

    return new_rm_ind


def main(args):
    device, orig,image,model = initialize(args)
    # run
    with torch.no_grad():
        model.eval()
        logits_r,logits_cw = model(image)
        predroom = BCHW2colormap(logits_r)
        predboundary = BCHW2colormap(logits_cw)
    if args.postprocess:
        # postprocess
        predroom = post_process(predroom,predboundary)
    rgb = ind2rgb(predroom,color_map=floorplan_fuse_map)
    # plot
    plt.subplot(1,3,1); plt.imshow(orig[:,:,::-1])
    plt.subplot(1,3,2); plt.imshow(rgb)
    plt.subplot(1,3,3); plt.imshow(predboundary)
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--loadmodel',type=str,default="log/store2/checkpoint.pt")
    p.add_argument('--postprocess',type=bool,default=False)
    p.add_argument('--image_path',type=str,default="/media/yui/Disk/data/deepfloorplan/dataset/newyork/test/47545145.jpg")
    args = p.parse_args()

    main(args)




