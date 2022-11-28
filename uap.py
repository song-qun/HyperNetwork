import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients






def deepfool(image, net, num_classes, overshoot, max_iter,device):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = net(image).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)

        r_tot = np.float32(r_tot + r_i)
        
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    
    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image


def project_lp(v, xi, p):

    if p==2:
        # pass
        normVal = torch.norm(v,p)
        mask = normVal<=xi
        scaling = xi/normVal
        scaling[mask] = 1
        v = v*scaling

    elif p == np.inf:
        v=torch.sign(v)*torch.min(abs(v),torch.ones_like(v)*xi)
    else:
        raise ValueError("Values of a different from 2 and Inf are currently not surpported...")

    return v

def generate(args, train_loader, test_loader, net, delta=0.2, max_iter_uni=10, xi=0.1, p=np.inf, overshoot=0.2, max_iter_df=50):

    device = args.device
    num_classes = args.num_classes

    num_img_trn = 1000
    num_img_tst = 100
   
    iter = 0
    v=torch.zeros([1,args.in_channels,args.input_size,args.input_size]).to(device)
    best_fooling_rate = 0.0

    # start an epoch
    while best_fooling_rate < 1-delta and iter < max_iter_uni:
        print("Starting pass number ", iter)
        fooling_rate = 0.0
        for i, (data, labels) in enumerate(train_loader):
            if i >= num_img_trn:
                break
            else:
                cur_img = data.to(device)
                cur_img1 = cur_img.clone().detach()
                # cur_img1 = cur_img1[np.newaxis, :]
                r2 = int(net(cur_img1).max(1)[1])
                torch.cuda.empty_cache()

                per_img = cur_img.clone().detach()
                per_img += v

                per_img1 = per_img.clone().detach()
                # per_img1 = per_img1[np.newaxis, :]
                r1 = int(net(per_img1).max(1)[1])
                torch.cuda.empty_cache()

                if r1 == r2:
                    dr, iter_k, label, k_i, pert_image = deepfool(per_img1, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df,device=device)
                    dr = torch.tensor(dr).to(device)
                    dr = torch.reshape(dr,v.shape)
                    if iter_k < max_iter_df-1:
                        v += dr
                        v = project_lp(v, xi, p)

        iter = iter + 1

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_idx >= num_img_tst:
                    break
                else:
                    inputs = inputs.to(device)
                    inputs += v
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    if predicted.data.item() != targets.data.item():
                        fooling_rate += 1

            torch.cuda.empty_cache()

            fooling_rate = fooling_rate/num_img_tst
            print("FOOLING RATE: ", fooling_rate)
            if fooling_rate > best_fooling_rate:
                np.save('{}-uap-{}-{}.npy'.format(args.dataset,num_img_trn,xi), v.cpu().data)
                # print(v.shape)
                best_fooling_rate = fooling_rate

    return v