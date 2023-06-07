import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

class NetBlock(torch.nn.Module):
    
    def __init__(self, arch, activation=torch.tanh):
        """
        :param arch: architecture of block [list]
        :param activation: activation function [function]
        """
        super(NetBlock, self).__init__()
        
        # parameters
        self.nb_layers = len(arch)
        self.activation = activation
        
        # build layers
        for i in range(self.nb_layers-1):
            self.add_module(f'Layer_{i}', torch.nn.Linear(arch[i], arch[i+1], dtype=torch.double))
    
    def forward(self, x):
        for i in range(self.nb_layers-2):
            x = self.activation(self._modules[f'Layer_{i}'](x))
        i+=1
        x = self._modules[f'Layer_{i}'](x)
        return x

class Net(torch.nn.Module):
    # parameter step gap makes no sense anymore, update code

    def __init__(self, arch, dt, step_gap, activation=torch.tanh, residNet=True):
        """
        param arch: architecture of net [list]
        param dt: time step of dataset
        param step_gap: amount of steps dt taken in forward pass
        param activation: activation function
        """
        super(Net, self).__init__()

        # checks
        assert isinstance(arch, list)
        assert arch[0] == arch[-1]

        # param
        self.n_dim = arch[0]
        self.dt = dt
        self.step_gap = step_gap
        self.activation = activation
        self.nb_layers = len(arch)
        self.residNet = residNet
        self.arch = arch

        # build net
        self.add_module('NNBlock', NetBlock(arch, activation=activation))

    def forward(self, x):
        """
        param x_init: array of shape n x n_dim
        return: next step of shape n x n_dim
        """
        nn_out = self._modules['NNBlock'](x)

        if self.residNet:   # nn_out acts as increment
            return x + nn_out
        else:               # nn_out acts as next state
            return nn_out

    def train(self, data, max_epoch, lr=1e-3, model_path=None, crit=torch.nn.MSELoss(reduction='none'), reg=False):
        """
        param dataset: n_train x 2 (in/out) x ndof 
        param max_epoch: maximum number of iterations
        param lr: learning rate
        param model_path: path to save model
        return: None
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
        epoch = 0
        self.loss = np.zeros(max_epoch)
        no_improvement_counter = 0
        best_loss = 1e10
        l1_lambda = 0.001
        while epoch < max_epoch:
            epoch += 1
            # to do: shuffle data
            train_x = data[:,0,:]
            train_y = data[:,1,:]
            # print(train_x.shape, train_y.shape)
            train_loss = self.calculate_loss(train_x, train_y, crit)

            # l1_regularization = torch.tensor(0.)
            # for param in self.parameters():
            #     l1_regularization += torch.norm(param, p=1)
            # train_loss += l1_lambda * l1_regularization

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loss = train_loss.item()
            self.loss[epoch-1] = loss

            if loss < best_loss:
                best_loss = loss
                no_improvement_counter = 0
            
            else:  
                no_improvement_counter += 1
                if no_improvement_counter > 15:
                    scheduler.step()
                    no_improvement_counter = 0

                if loss*1.075 > self.loss[epoch-5000] and loss<1.01*best_loss and epoch>10000:
                    print(f"epoch {epoch}")
                    print(f"no improvement observed in last {no_improvement_counter} iterations")
                    print(f"final loss : {loss:.2e}")
                    self.loss = self.loss[:epoch]
                    break
            
            if epoch > max_epoch-1000 and loss<1.5*best_loss:
                break

            if epoch % 10000 == 0:
                print(f'epoch {epoch}, training loss {train_loss.item():.2e}')
            
        if model_path is not None:
            print("model saved")
            f = open(model_path, "wb")
            pickle.dump(self, f)
        
    def calculate_loss(self, x, y_seq, criterion):
        """
        param x: initial data, shape n x n_dim
        param y_seq: sequence following to x, shape n x nb_steps x n_dim
        return: loss (mse alltogether)
        """
        # nb_steps in terms of dt
        n, n_dim = y_seq.shape
        assert n_dim == self.n_dim

        # forward (recurrence)
        y_preds = self.forward(x)

        # compute loss
        loss = criterion(y_preds, y_seq).mean()

        return loss


    def my_loss(self, output, target):
        loss = ((output - target)/target)**2
        return loss


    def forecast(self, init, t_end):
        """
        init: n x ndim
        out: n x timesteps x ndim
        """
        n, ndim = np.shape(init)
        modeldt = self.dt*self.step_gap
        nsteps = int(t_end/modeldt)
        print(modeldt, nsteps)
        result = torch.zeros((n, nsteps, ndim), dtype=self.precision)
        result[:,0,:] = init
        for i in range(1,nsteps):
            result[:,i,:] = self.forward(result[:,i-1])
        return result.detach().cpu().numpy()
