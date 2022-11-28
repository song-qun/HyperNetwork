import argparse
import torch
import dataloader
from models import main_models
import numpy as np
import sys
import torch
import time
parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches_1',type=int,default=10)
parser.add_argument('--n_epoches_2',type=int,default=100)
parser.add_argument('--n_epoches_3',type=int,default=50)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=64)

opt=vars(parser.parse_args())

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)


#--------------pretrain g and h for step 1---------------------------------
train_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=True)
test_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=False)
# for data,labels in train_dataloader:
#     print(f'mnist dataset: mean: {data.mean()} max: {data.max()} min: {data.min()} std: {data.std()}')

#     sys.exit()

classifier=main_models.Classifier()
print("classifier: ")
print(classifier)
encoder=main_models.Encoder()
print("Encoder: ")
print(encoder)
discriminator=main_models.DCD(input_features=128)
print("Discriminator")
print(discriminator)

classifier.to(device)
encoder.to(device)
discriminator.to(device)
loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()))


# for epoch in range(opt['n_epoches_1']):
#     for data,labels in train_dataloader:
#         # print(f'mnist dataset: mean: {data.mean()} max: {data.max()} min: {data.min()} std: {data.std()}')
#         # 
#         data=data.to(device)
#         labels=labels.to(device)

#         optimizer.zero_grad()

#         y_pred=classifier(encoder(data))

#         loss=loss_fn(y_pred,labels)
#         loss.backward()

#         optimizer.step()

#     acc=0
#     for data,labels in test_dataloader:
#         data=data.to(device)
#         labels=labels.to(device)
#         y_test_pred=classifier(encoder(data))
#         acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

#     accuracy=round(acc / float(len(test_dataloader)), 3)

#     print("step1----Epoch %d/%d  accuracy: %.3f "%(epoch+1,opt['n_epoches_1'],accuracy))
#-------------------------------------------------------------------
# generate 20 models
s_t = time.time()
for xx in range(20):
    # save model
    classifier_PATH = "saved_models/classifier_mnist.pt"
    encoder_PATH = "saved_models/encoder_mnist.pt"
    # torch.save(classifier.state_dict(),classifier_PATH)
    # torch.save(encoder.state_dict(),encoder_PATH)

    encoder=main_models.Encoder()
    classifier=main_models.Classifier()

    encoder.load(encoder_PATH)
    classifier.load(classifier_PATH)

    # method 2
    # classifier_PATH = "saved_models/classifier.pt.tar"
    # encoder_PATH = "saved_models/encoder_original.pt.tar"

    # torch.save({'state_dict':classifier.state_dict()},classifier_PATH)
    # torch.save({'state_dict':encoder.state_dict()},encoder_PATH)

    # encoder=main_models.Encoder()
    # classifier=main_models.Classifier()

    # encoder_checkpoint = torch.load(encoder_PATH)
    # classifier_checkpoint = torch.load(classifier_PATH)

    # encoder.load_state_dict(encoder_checkpoint['state_dict'])
    # classifier.load_state_dict(classifier_checkpoint['state_dict'])

    encoder.to(device)
    classifier.to(device)

    encoder.eval()
    classifier.eval()

    # double check accuracy
    acc=0
    for data,labels in test_dataloader:
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=classifier(encoder(data))
        acc+=(torch.max(y_test_pred,1)[1]==labels).float().mean().item()

    accuracy=round(acc / float(len(test_dataloader)), 3)
    print("final accuracy: ",accuracy)

    #-------------------------------------------------------------------
    X_s,Y_s=dataloader.sample_data()
    X_t,Y_t=dataloader.create_target_samples_mnist(opt['n_target_samples'])

    print (f'X_s length {len(X_s)}, X_t length {len(X_t)}')
    print (f'X_s shape {X_s.shape}, X_t shape {X_t.shape}')
    #-----------------train DCD for step 2--------------------------------

    optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.001)


    for epoch in range(opt['n_epoches_2']):
        # data
        groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

        n_iters = 4 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size=40 #use mini_batch train can be more stable


        loss_mean=[]

        X1=[];X2=[];ground_truths=[]
        for index in range(n_iters):

            ground_truth=index_list[index]//len(groups[1])

            x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            #select data for a mini-batch to train
            if (index+1)%mini_batch_size==0:
                X1=torch.stack(X1)
                X2=torch.stack(X2)
                ground_truths=torch.LongTensor(ground_truths)
                X1=X1.to(device)
                X2=X2.to(device)
                ground_truths=ground_truths.to(device)

                optimizer_D.zero_grad()
                X_cat=torch.cat([encoder(X1),encoder(X2)],1)
                print("X_cat shape: ",X_cat.shape)
                y_pred=discriminator(X_cat.detach())
                loss=loss_fn(y_pred,ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        print("step2----Epoch %d/%d loss:%.3f"%(epoch+1,opt['n_epoches_2'],np.mean(loss_mean)))

    #----------------------------------------------------------------------

    #-------------------training for step 3-------------------
    optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.001)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.001)


    # test_dataloader=dataloader.svhn_dataloader(train=False,batch_size=opt['batch_size'])
    test_dataloader=dataloader.mnist_dataloader(batch_size=opt['batch_size'],train=False)


    for epoch in range(opt['n_epoches_3']):
        #---training g and h , DCD is frozen

        groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=opt['n_epoches_2']+epoch)
        G1, G2, G3, G4 = groups
        Y1, Y2, Y3, Y4 = groups_y
        groups_2 = [G2, G4]
        groups_y_2 = [Y2, Y4]

        n_iters = 2 * len(G2)
        index_list = torch.randperm(n_iters)

        n_iters_dcd = 4 * len(G2)
        index_list_dcd = torch.randperm(n_iters_dcd)

        mini_batch_size_g_h = 20 #data only contains G2 and G4 ,so decrease mini_batch
        mini_batch_size_dcd= 40 #data contains G1,G2,G3,G4 so use 40 as mini_batch
        X1 = []
        X2 = []
        ground_truths_y1 = []
        ground_truths_y2 = []
        dcd_labels=[]
        for index in range(n_iters):


            ground_truth=index_list[index]//len(G2)
            x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
            # y1=torch.LongTensor([y1.item()])
            # y2=torch.LongTensor([y2.item()])
            dcd_label=0 if ground_truth==0 else 2
            X1.append(x1)
            X2.append(x2)
            ground_truths_y1.append(y1)
            ground_truths_y2.append(y2)
            dcd_labels.append(dcd_label)

            if (index+1)%mini_batch_size_g_h==0:

                X1=torch.stack(X1)
                X2=torch.stack(X2)
                ground_truths_y1=torch.LongTensor(ground_truths_y1)
                ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                dcd_labels=torch.LongTensor(dcd_labels)
                X1=X1.to(device)
                X2=X2.to(device)
                ground_truths_y1=ground_truths_y1.to(device)
                ground_truths_y2 = ground_truths_y2.to(device)
                dcd_labels=dcd_labels.to(device)

                optimizer_g_h.zero_grad()

                encoder_X1=encoder(X1)
                encoder_X2=encoder(X2)

                X_cat=torch.cat([encoder_X1,encoder_X2],1)
                y_pred_X1=classifier(encoder_X1)
                y_pred_X2=classifier(encoder_X2)
                y_pred_dcd=discriminator(X_cat)

                loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
                loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
                loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

                loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

                loss_sum.backward()
                optimizer_g_h.step()

                X1 = []
                X2 = []
                ground_truths_y1 = []
                ground_truths_y2 = []
                dcd_labels = []


        #----training dcd ,g and h frozen
        X1 = []
        X2 = []
        ground_truths = []
        for index in range(n_iters_dcd):

            ground_truth=index_list_dcd[index]//len(groups[1])

            x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            if (index + 1) % mini_batch_size_dcd == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_d.zero_grad()
                X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_d.step()
                # loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []

        #testing
        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

        accuracy = round(acc / float(len(test_dataloader)), 3)

        print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_3'], accuracy))
            
    classifier_PATH = "saved_models/mnist/classifier_mnist_" + str(xx) + ".pt"
    encoder_PATH = "saved_models/mnist/encoder_mnist_" + str(xx) + ".pt"
    torch.save(classifier.state_dict(),classifier_PATH)
    torch.save(encoder.state_dict(),encoder_PATH)
e_t = time.time()
print("total time to generate 20 models for mnist: ",e_t-s_t)




















