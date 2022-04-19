import argparse
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/store2')
from importmod import *
import tqdm
import torch.optim as optim
import random
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from net import *
from data import *

def balanced_entropy(preds,targets):
    eps = 1e-6
    m = nn.Softmax(dim=1)
    z = m(preds)
    cliped_z = torch.clamp(z,eps,1-eps)
    log_z = torch.log(cliped_z)
    num_classes = targets.size(1)
    ind = torch.argmax(targets,1).type(torch.int)

    total = torch.sum(targets)
    
    m_c,n_c = [],[]
    for c in range(num_classes):
        m_c.append((ind==c).type(torch.int))
        n_c.append(torch.sum(m_c[-1]).type(torch.float))

    c = []
    for i in range(num_classes):
        c.append(total-n_c[i])
    tc = sum(c)

    loss = 0
    for i in range(num_classes):
        w = c[i]/tc
        m_c_one_hot = F.one_hot((i*m_c[i]).permute(1,2,0).type(torch.long),
                num_classes)
        m_c_one_hot = m_c_one_hot.permute(2,3,0,1)
        y_c = m_c_one_hot*targets
        loss += w*torch.sum(-torch.sum(y_c*log_z,axis=2))
    return loss/num_classes

def cross_two_tasks_weight(rooms,boundaries):
    p1 = torch.sum(rooms).type(torch.float)
    p2 = torch.sum(boundaries).type(torch.float)
    w1 = torch.div(p2,p1+p2)
    w2 = torch.div(p1,p1+p2)
    return w1,w2

def BCHW2colormap(tensor,earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1,2,0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result,axis=2)
    return result

def compare(images,rooms,boundaries,r,cw):
    import matplotlib.pyplot as plt
    image = (BCHW2colormap(images,earlyexit=True)*255).astype(np.uint8)
    room = BCHW2colormap(rooms).astype(np.uint8)
    boundary = BCHW2colormap(boundaries).astype(np.uint8)
    r = BCHW2colormap(r).astype(np.uint8)
    cw = BCHW2colormap(cw).astype(np.uint8)
    f = plt.figure()
    plt.subplot(2,3,1);plt.imshow(image)
    plt.subplot(2,3,2);plt.imshow(room)
    plt.subplot(2,3,3);plt.imshow(boundary)
    plt.subplot(2,3,5);plt.imshow(r)
    plt.subplot(2,3,6);plt.imshow(cw)
    return f

def setup(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dnn model
    model = DFPmodel(pretrained=args.pretrained,freeze=args.freeze)
    if args.loadmodel:
        model.load_state_dict(torch.load(args.loadmodel))
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    return device,model,optimizer

def getloader(args):
    # data
    trans = transforms.Compose([transforms.ToTensor()])
    r3d = r3dDataset(transform=trans)
    total_image = len(r3d)
    indices = list(range(total_image))
    total_batch = total_image//args.batch_size
    split = int(np.floor(args.valsplit*total_image))
    train_indices,val_indices = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(r3d,batch_size=args.batch_size,
            sampler=train_sampler,num_workers=args.numworkers)
    valid_loader = DataLoader(r3d,batch_size=args.batch_size,
            sampler=valid_sampler,num_workers=args.numworkers)
    return train_loader,valid_loader,total_image,total_batch

def main(args):
    device,model,optimizer = setup(args)
    train_loader,valid_loader,total_image,total_batch = getloader(args)
    if args.earlystop:
        early_stopping = EarlyStopping(patience=args.patience,verbose=True)
    for epoch in range(args.maxiters):
        running_loss = 0.0
        
        for idx,(im,cw,r,_) in enumerate(train_loader):
            im,cw,r = im.to(device),cw.to(device),r.to(device)
            # zero gradients
            optimizer.zero_grad()
            model.train()
            # forward
            logits_r,logits_cw = model(im)
            # loss
            loss1 = balanced_entropy(logits_r,r)
            loss2 = balanced_entropy(logits_cw,cw)
            w1,w2 = cross_two_tasks_weight(r,cw)
            loss = w1*loss1 + w2*loss2
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            
            if idx%20 == 0:
                print('[INFO] Batch {}/{:d} Train Loss: {:.1f}, R Loss: {:.1f}, B Loss: {:.1f}'.format(idx,int(total_batch*(1-args.valsplit)),loss.item(),loss1.item(),loss2.item()))

            # statistics
            running_loss += loss.item()

        if args.tensorboard:
            if epoch == 0 or epoch%20 ==0:
                f1 = compare(im,r,cw,logits_r,logits_cw)
                writer.add_figure('train',f1,epoch)
        print("Epoch {} Total Train Loss: {:.1f}".format(epoch,running_loss))
        
        running_loss_val = 0.0
        for idx,(im,cw,r,_) in enumerate(valid_loader):
            im,cw,r = im.to(device),cw.to(device),r.to(device)
            with torch.no_grad():
                model.eval()
                optimizer.zero_grad()
                # forward
                logits_r,logits_cw = model(im)
                # loss
                loss1 = balanced_entropy(logits_r,r)
                loss2 = balanced_entropy(logits_cw,cw)
                w1,w2 = cross_two_tasks_weight(r,cw)
                loss = w1*loss1 + w2*loss2
            # statistics
            running_loss_val += loss.item()

        print("Epoch {} Total Val Loss: {:.1f}".format(epoch,running_loss_val))
        if args.tensorboard:
            if epoch == 0 or epoch%20 ==0:
                f2 = compare(im,r,cw,logits_r,logits_cw)
                writer.add_figure('val',f2,epoch)
            writer.add_scalars('loss',{'train_loss':running_loss,
            'valid_loss':running_loss_val},epoch)
        if args.earlystop:
            early_stopping(running_loss_val,model)
        else:
            torch.save(model.state_dict(),'log/store2/checkpoint.pt')
        
        if args.earlystop:
            if early_stopping.early_stop:
                print("Early stopping")
                break
    model.load_state_dict(torch.load('log/store2/checkpoint.pt'))
    
    return model

if __name__ == "__main__":    
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--maxiters',type=int,default=2000)
    p.add_argument('--numworkers',type=int,default=1)
    p.add_argument('--pretrained',type=bool,default=True)
    p.add_argument('--freeze',type=bool,default=True)
    p.add_argument('--loadmodel',type=str,default=None)
    p.add_argument('--valsplit',type=float,default=.1)
    p.add_argument('--tensorboard',type=bool,default=True)
    p.add_argument('--earlystop',type=bool,default=False)
    p.add_argument('--patience',type=int,default=20)
    args = p.parse_args()
    main(args)
    breakpoint()

