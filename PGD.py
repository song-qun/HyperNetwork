import torch
import torch.nn as nn
import numpy as np

class Attack:
    '''
    parent class for the attacks
    generate one adversarial example at a time
    '''
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def attack(self, data,target, model):
        pass

class FGSM(Attack):
    '''Fast Gradient Sign Method'''
    def __init__(self, epsilon):
        super(FGSM,self).__init__(epsilon)
        
    def attack(self, data, target, model):

        data = data.requires_grad_(True)
        loss_fn = nn.NLLLoss()
        modelOut = model.forward(data)
        loss = loss_fn(modelOut, target)
        loss.backward()
        updates = torch.sign(data.grad)*self.epsilon

        adv = data+updates
        adv = torch.clamp(adv,-0.5,0.5)
        
        return adv

class PGD(Attack):
    '''Projected Gradient Descent'''
    def __init__(self, epsilon, p, stepsize, numIters):
        super(PGD,self).__init__(epsilon)
        self.p = p
        self.stepsize = stepsize
        self.numIters = numIters

    def attack(self, data,target, model):

        batchsize = data.shape[0]

        # random start
        delta = torch.rand_like(data)*2*self.epsilon-self.epsilon

        # projected into feasible set if needed
        if self.p!=np.inf: 
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
            mask = normVal<=self.epsilon
            scaling = self.epsilon/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1)
        adv = data+delta

        # PGD to get adversarial example
        loss_fn = nn.NLLLoss()
        for i in range(self.numIters):
            adv = adv.clone().detach().requires_grad_(True) # clone the imgAdv as the next iteration input
            modelOut = model.forward(adv)
            loss = loss_fn(modelOut, target)
            loss.backward()
            updates = adv.grad
            if self.p==np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates/normVal.view(batchsize, 1)
            updates = updates*self.stepsize
            adv = adv+updates
            # project the disturbed image to feasible set if needed
            delta = adv-data
            if self.p==np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(batchsize, 1)
            adv = data+delta

        adv = torch.clamp(adv,-0.5,0.5)
        return adv