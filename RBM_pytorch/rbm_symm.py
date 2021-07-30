# IMPORTANT: To see progress bars during training, comment or uncomment appropriately in lines 5,6, 420,421
import torch
import h5py
import numpy as np
#import tqdm.notebook as tq # Uncomment if using notebook
from tqdm import tqdm # Uncomment if using terminal

class RBM:
    # NEEDED VAR:
    # * num_visible
    # * num_hidden
    def __init__(self, num_visible: int, # number of visible nodes
                 num_hidden: int, # number of hidden nodes
                 device, # CPU or GPU ?
                 gibbs_steps=10, # number of MCMC steps for computing the neg term
                 anneal_steps=0,# number of MCMC steps for computing the neg term when doing anneal (= for no annealing)
                 var_init=1e-4, # variance of the init weights
                 dtype=torch.float,
                 num_pcd = 100, # number of permanent chains
                 learning_rate = 0.01, # learning rate
                 epochs_max = 100, # number of epochs
                 mini_batch_size = 100, # size of the minibatch
                 sampler_name = "SIG", # type of activation function : SIG|RELU_MAX|...
                 ann_threshold = 4, # threshold (of the max value of the spectrum of w) above which, the annealing is used 
                 regL2 = 0, # value of the L2 regularizer
                 DynBetaGradient = False, # Activate the temperature for the gradient descent
                 UpdFieldsVis = True, # Update visible fields ?
                 UpdFieldsHid = True, # Update hidden fields ?
                 Symmetry_training = False,
                 Symmetry_cells = 0,
                 Saving_interval = 10
                 ): 
        self.Nv = num_visible
        self.Nh = num_hidden
        self.gibbs_steps = gibbs_steps
        self.anneal_steps = anneal_steps
        self.device = device
        self.dtype = dtype
        self.Symmetry_training = Symmetry_training
        self.Saving_interval = Saving_interval
        if not Symmetry_training:
            # weight of the RBM
            self.W = torch.randn(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)*var_init
            # visible and hidden biases
            self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
            self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
            # Set samplers
            self.SamplerHiddens = self.SampleHiddens01
            self.SamplerVisibles = self.SampleVisibles01
        # permanent chain
        self.X_persistent_chain = torch.bernoulli(torch.rand((self.Nv,num_pcd), device=self.device, dtype=self.dtype))
        self.learning_rate = learning_rate
        self.epochs_max = epochs_max
        self.mini_batch_size = mini_batch_size
        self.num_pcd = num_pcd
        self.sampler_name = sampler_name
        
        self.ann_threshold = ann_threshold
        # effective temperature
        self.β = 1
        self.total_epochs = 0
        self.up_tot = 0
        self.list_save_time = []
        self.regL2 = regL2
        self.TempUpd = 1
        self.DynBetaGradient = DynBetaGradient
        self.UpdFieldsVis = UpdFieldsVis
        self.UpdFieldsHid = UpdFieldsHid
        # A matrix array of the weights, to export
        # This would export the flattened matrix at each epoch
        # self.W_matrix = np.zeros((epochs_max,self.Nh*self.Nv))
        # List to store energy averages while training
        # at each epoch we will store here the average energy of the training df
        self.energies_train = []
        self.energies_test = []
        self.free_energies_ = [] 
        self.log_likelihoods_train = []
        self.log_likelihoods_test = []
        self.pseudo_likelihoods_train = []
        self.pseudo_likelihoods_test = []
        self.list_of_W_matrices = []
        self.list_of_vbias = []
        self.list_of_hbias = []
        if Symmetry_training:
            # Total number of channels of the RBM
            self.Symmetry_cells = Symmetry_cells
            if not self.Symmetry_cells > 1:
                raise ValueError('Symmetry cells should be an integer greater than 1')
            # Visible neurons per channel
            self.v_units_per_cell = int(self.Nv/self.Symmetry_cells)
            # Hidden neurons per channel
            self.h_units_per_cell = int(self.Nh/self.Symmetry_cells)
            # weight of the RBM
            self.W = torch.randn(size=(self.Symmetry_cells, self.h_units_per_cell, self.v_units_per_cell), device=self.device, dtype=self.dtype)*var_init
            # Temporary matrices to save some time during visible and hidden sampling steps
            self._W_v_temp = torch.zeros(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)
            self._W_h_temp = torch.zeros(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)
            # visible and hidden biases
            self.vbias = torch.zeros(self.v_units_per_cell, device=self.device, dtype=self.dtype)
            self.hbias = torch.zeros(self.h_units_per_cell, device=self.device, dtype=self.dtype)
            # Set samplers
            self.SamplerHiddens = self.Sampler_Hiddens_Symmetry
            self.SamplerVisibles = self.Sampler_Visibles_Symmetry
            # List to store the Singular values at each epoch
            self.list_of_singular_values_by_cell = []
            # Adjust learning rate to the number of cells
            self.learning_rate /= self.Symmetry_cells
            
        


    def ImConcat(self,X,ncol=10,nrow=5):
        tile_X = []
        for c in range(nrow):
            L = torch.cat((tuple(X[i,:].reshape(28,28) for i in np.arange(c*ncol,(c+1)*ncol))))
            tile_X.append(L)
        return torch.cat(tile_X,1)

    def ExpCos(self,t,a,τ1,τ2):
        return 1+(a-1)*torch.exp(-torch.tensor(t)/τ1)*torch.cos(torch.tensor(t)/τ2)

    def ComputeAATS(self,X,fake_X,s_X):
        CONCAT = torch.cat((X[:,:s_X],fake_X[:,:s_X]),1)
        dAB = torch.cdist(CONCAT.t(),CONCAT.t())    
        torch.diagonal(dAB).fill_(float('inf'))
        dAB = dAB.numpy()
        ## torch.diagonal(dAB).fill_(float('inf'))
        # dAB_n = dAB.numpy()
        # the next line is use to tranform the matrix into
        #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
        #  d_FT d_FF        d_FT  d_FF
        dAB[:int(dAB.shape[0]/2),:] = dAB[:int(dAB.shape[0]/2),::-1] 
        closest = dAB.argmin(axis=1) 
        n = int(closest.shape[0]/2)
        ninv = 1/n
        correctly_classified = closest>=n   #np.concatenate([(closest[:n] < n), (closest[n:] >= n)])
        AAtruth = (closest[:n] >= n).sum()*ninv  # for a true sample, proba that the closest is in the set of true samples
        AAsyn = (closest[n:] >= n).sum()*ninv  # for a fake sample, proba that the closest is in the set of fake samples
        # AATS = (AAtruth + AAsyn)/2 
        return AAtruth, AAsyn

    # init the visible bias using the empirical frequency of the training dataset
    def SetVisBias(self,X):
        NS = X.shape[1]
        prob1 = torch.sum(X,1)/NS
        prob1 = torch.clamp(prob1,min=1e-5)
        prob1 = torch.clamp(prob1,max=1-1e-5)
        self.vbias = -torch.log(1.0/prob1 - 1.0)

    # SetSample: change de sampler used for the hidden nodes
    # SIG: sigmoid activation function (h=0,1)
    # RELU_MAX: relu activation function (h>0)
    def SetSampler(self,sampler_name):
        if sampler_name == "SIG":
            self.SamplerHiddens = self.SampleHiddens01
        elif sampler_name == "RELU_MAX":
            self.SamplerHiddens = self.SampleHiddensRELU_MAX

    # define an initial value fo the permanent chain
    def InitXpc(self,V):
        self.X_persistent_chain = V

    # Sampling and getting the mean value using RELU
    def SampleHiddensRELU_MAX(self,V,β=1):        
        act_h = β*(self.W.mm(V).t() + self.hbias).t()
        mh = torch.clamp(act_h,min=0)
        h = torch.clamp(act_h + torch.randn(size=mh.size(),device=self.device, dtype=self.dtype),min=0)
        return h,mh

    # Sampling and getting the mean value using Sigmoid
    def SampleHiddens01(self,V,β=1):             
        mh = torch.sigmoid(β*(self.W.mm(V).t() + self.hbias).t())
        h = torch.bernoulli(mh)
        return h,mh

    def _get_W_v_matrix(self, W_by_cells):
        # Create a full matrix out of the rolled matrices
        rolled_W_temp = torch.zeros(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)
        for cell in range(self.Symmetry_cells):
            rolled_W_temp[cell*self.h_units_per_cell:(cell+1)*self.h_units_per_cell,:] = torch.transpose(torch.roll(W_by_cells, cell,0),0,1).reshape(self.h_units_per_cell, self.Nv)
        return rolled_W_temp
    
    def Sampler_Hiddens_Symmetry(self, visible, β=1):
        # Matrix multiply Wx+h Bmm is batch matrix multiplication
        # Symmetry of weights by translation means the weights must be rolled by cell axis
        # once every neuron on each cell have performed their operation, this is done in the get matrices functions
        '''
        mh = torch.sigmoid( β*(torch.transpose(self.W,1,2).reshape(visible.shape[0],self.h_units_per_cell).t().mm(visible) + self.hbias.reshape(-1,1)) )
        reshaped_visible = visible.reshape(self.Symmetry_cells,self.v_units_per_cell,visible.shape[1])
        mh = torch.sigmoid( β*(torch.bmm(self.W, reshaped_visible).reshape(self.Nh, visible.shape[1]) + self.hbias.repeat(self.Symmetry_cells).reshape(-1,1)))
        
        h_pos = torch.zeros((self.Nh, visible.shape[1]), device=self.device)
        for cell in range(self.Symmetry_cells):
            reshaped_W_h = torch.transpose(torch.roll(self.W, cell,0),0,1).reshape(self.h_units_per_cell, self.Nv)
            h_pos[cell*self.h_units_per_cell:(cell+1)*self.h_units_per_cell] = torch.matmul(reshaped_W_h, visible) + self.hbias.reshape(-1,1)
        '''
        
        mh = torch.sigmoid( β*(torch.matmul(self._W_v_temp, visible) + self.hbias.repeat(self.Symmetry_cells).reshape(-1,1)))
        h = torch.bernoulli(mh)
        return h,mh

    def _get_W_h_matrix(self, W_by_cells):
        flipped_W = torch.flip(W_by_cells,[0]) # Reverse order of matrix at index 0
        flipped_rolled_W_temp = torch.zeros(size=(self.Nv,self.Nh), device=self.device, dtype=self.dtype)
        for cell in range(self.Symmetry_cells):
            flipped_rolled_W = torch.roll(flipped_W, cell+1,0) # cell + 1 so that it begins with the first cell
            flipped_rolled_W_temp[cell*self.v_units_per_cell:(cell+1)*self.v_units_per_cell,:] = torch.transpose(torch.transpose(flipped_rolled_W,1,2),0,1).reshape(self.v_units_per_cell, self.Nh) #Good
        return flipped_rolled_W_temp

    def Sampler_Visibles_Symmetry(self, hiddens, β=1):
        # Matrix multiply W^TH+k Bmm is batch matrix multiplication
        '''
        mv = torch.sigmoid(β*(self.W.transpose(1,2).transpose(0,1).reshape(self.Nv, self.h_units_per_cell).mm(hiddens) + self.vbias.repeat_interleave(self.Symmetry_cells).reshape(-1,1)))
        reshaped_hidden = hiddens.reshape(self.Symmetry_cells,self.h_units_per_cell,hiddens.shape[1])
        mv = torch.sigmoid( β*(torch.bmm(torch.transpose(self.W,1,2), reshaped_hidden).reshape(self.Nv, hiddens.shape[1]) + self.vbias.repeat(self.Symmetry_cells).reshape(-1,1)))
        
        visible_activations_temp = torch.zeros((self.Nv,hiddens.shape[1]), device=self.device)
        for cell in range(self.Symmetry_cells):
            flipped_rolled_W = torch.roll(flipped_W, cell+1,0) # cell + 1 so that it begins with the first cell
            reshaped_W_v = torch.transpose(torch.transpose(flipped_rolled_W,1,2),0,1).reshape(self.v_units_per_cell, self.Nh) #Good
            visible_activations_temp[cell*self.v_units_per_cell:(cell+1)*self.v_units_per_cell] = torch.matmul(reshaped_W_v, hiddens) + self.vbias.reshape(-1,1)
        '''

        visible_activations_temp = torch.matmul(self._W_h_temp, hiddens) + self.vbias.repeat(self.Symmetry_cells).reshape(-1,1)
        mv = torch.sigmoid( β*visible_activations_temp)
        v = torch.bernoulli(mv)
        return v,mv

    # H is Nh X M
    # W is Nh x Nv
    # Return Visible sample and average value for binary variable
    def SampleVisibles01(self,H,β=1):
        mv = torch.sigmoid(β*(self.W.t().mm(H).t() + self.vbias).t())
        v = torch.bernoulli(mv)
        return v,mv

    # Compute the negative term for the gradient
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def GetAv(self,it_mcmc=0,βs=-1):
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps

        v = self.X_persistent_chain

        β=self.β 

        if self.anneal_steps > 0:
            β_init = 1
            β_end = 1
            _,s,_ = torch.svd(self.W) 
            smax = torch.max(s)
            if(smax > self.ann_threshold ):
                β_init = self.ann_threshold/smax
                β_end = 1

            Dβ = (β_end - β_init)/(self.anneal_steps-1)
            β = β_init  
            h, mh = self.SamplerHiddens(v,β=β)
            for t in range(self.anneal_steps):
                β += Dβ
                v, mv = self.SamplerVisibles(h,β=β)
                h, mh = self.SamplerHiddens(v,β=β)
            
        β=self.β 
        if βs != -1:
            β=βs
        h,mh_pos = self.SamplerHiddens(v,β=β)
        v,mv = self.SamplerVisibles(h,β=β)
        
        for t in range(1,it_mcmc):
            h,mh = self.SamplerHiddens(v,β=β)
            v,mv = self.SamplerVisibles(h,β=β)
            
        return self.X_persistent_chain,mh_pos,v,h

    # Return samples and averaged values
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def Sampling(self,X,it_mcmc=0,βs=1,anneal_steps=0,ann_threshold=4):
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps

        v = X
        β=self.β
        if anneal_steps > 0:
            β_init = 1
            β_end = 1
            _,s,_ = torch.svd(self.W) 
            smax = torch.max(s)
            if(smax > ann_threshold ):
                β_init = ann_threshold/smax
                β_end = 1

            Dβ = (β_end - β_init)/(anneal_steps-1)
            β = β_init  
            h, mh = self.SamplerHiddens(v,β=β)
            for t in range(anneal_steps):
                β += Dβ
                v, mv = self.SampleVisibles01(h,β=β)
                h, mh = self.SamplerHiddens(v,β=β)
            

        β=self.β 
        if βs != -1:
            β=βs
        h,mh = self.SamplerHiddens(v,β=β)
        v,mv = self.SampleVisibles01(h,β=β)
        
        for t in range(it_mcmc-1):
            h,mh = self.SamplerHiddens(v,β=β)
            v,mv = self.SampleVisibles01(h,β=β)
            
        return v,mv,h,mh

    # Update weights and biases
    def updateWeights(self,v_pos,h_pos,v_neg,h_neg):
        if self.DynBetaGradient:
            self.TempUpd = 1/self.ExpCos(self.up_tot,4,10000,100)

        lr_p = self.TempUpd*self.learning_rate/self.mini_batch_size
        lr_n = self.TempUpd*self.learning_rate/self.num_pcd
        lr_reg = self.TempUpd*self.learning_rate*self.regL2

        # Regular RBM (Not symmetric)
        if not self.Symmetry_training:        
            
            self.W += h_pos.mm(v_pos.t())*lr_p - h_neg.mm(v_neg.t())*lr_n - 2*lr_reg*self.W 
            if self.UpdFieldsVis:
                self.vbias += torch.sum(v_pos,1)*lr_p - torch.sum(v_neg,1)*lr_n
            if self.UpdFieldsHid:    
                self.hbias += torch.sum(h_pos,1)*lr_p - torch.sum(h_neg,1)*lr_n
        # RBM with symmetries
        if self.Symmetry_training:
            # To calculate the elements, it is convenient to first calculate the whole matrix,
            # then average the symmetric elements together
            '''
            temp_w_pos = torch.zeros(size=(self.Symmetry_cells, self.h_units_per_cell, self.v_units_per_cell), device=self.device, dtype=self.dtype)
            temp_w_neg = torch.zeros(size=(self.Symmetry_cells, self.h_units_per_cell, self.v_units_per_cell), device=self.device, dtype=self.dtype)
            for cell in range(self.Symmetry_cells):
                for s in range(self.h_units_per_cell):
                    for r in range(self.v_units_per_cell):
                        index_set_temp = self._symmetry_indexes_matrix[(self._symmetry_indexes_matrix[:,2]==cell) & (self._symmetry_indexes_matrix[:,3]==s) & (self._symmetry_indexes_matrix[:,4]==r)]
                        temp_w_pos[cell,s,r] = temp_w_full_pos[(index_set_temp[:,1]),(index_set_temp[:,0])].mean()
                        temp_w_neg[cell,s,r] = temp_w_full_neg[(index_set_temp[:,1]),(index_set_temp[:,0])].mean()
            '''
            # Full matrix calculation
            temp_w_full_pos = h_pos.mm(v_pos.t())
            temp_w_full_neg = h_neg.mm(v_neg.t())
            # Positive term
            reshaped_blocks_W_pos = torch.transpose(torch.transpose(torch.transpose(torch.transpose(temp_w_full_pos.reshape(self.Symmetry_cells, self.h_units_per_cell, self.Nv ),1,2).reshape(self.Symmetry_cells,self.Symmetry_cells,self.v_units_per_cell,self.h_units_per_cell),1,2),1,2),2,3)
            W_by_blocks_rolled_pos = torch.zeros(size=(self.Symmetry_cells, self.Symmetry_cells, self.h_units_per_cell, self.v_units_per_cell), device=self.device, dtype=self.dtype)
            for block_index,block_row in enumerate(reshaped_blocks_W_pos):
                W_by_blocks_rolled_pos[block_index] = torch.roll(block_row,-block_index,0)
            # negative term
            reshaped_blocks_W_neg = torch.transpose(torch.transpose(torch.transpose(torch.transpose(temp_w_full_neg.reshape(self.Symmetry_cells, self.h_units_per_cell, self.Nv ),1,2).reshape(self.Symmetry_cells,self.Symmetry_cells,self.v_units_per_cell,self.h_units_per_cell),1,2),1,2),2,3)            
            W_by_blocks_rolled_neg = torch.zeros(size=(self.Symmetry_cells, self.Symmetry_cells, self.h_units_per_cell, self.v_units_per_cell), device=self.device, dtype=self.dtype)
            for block_index,block_row in enumerate(reshaped_blocks_W_neg):
                W_by_blocks_rolled_neg[block_index] = torch.roll(block_row,-block_index,0)
            # Weight update
            self.W +=  W_by_blocks_rolled_pos.sum(axis=0)*lr_p - W_by_blocks_rolled_neg.sum(axis=0)*lr_n - 2*lr_reg*self.W
            if self.UpdFieldsVis:
                self.vbias += v_pos.reshape(self.Symmetry_cells, self.v_units_per_cell,v_pos.shape[1]).sum(axis=0).sum(axis=1)*lr_p - v_neg.reshape(self.Symmetry_cells, self.v_units_per_cell,v_neg.shape[1]).sum(axis=0).sum(axis=1)*lr_n
            if self.UpdFieldsHid:    
                self.hbias += h_pos.reshape(self.Symmetry_cells, self.h_units_per_cell,h_pos.shape[1]).sum(axis=0).sum(axis=1)*lr_p - h_neg.reshape(self.Symmetry_cells, self.h_units_per_cell,h_neg.shape[1]).sum(axis=0).sum(axis=1)*lr_n
    '''
    # Update only biases
    def updateFields(self,v_pos,h_pos,v_neg,h_neg):
        lr_p = self.learning_rate/self.mini_batch_size
        lr_n = self.learning_rate/self.num_pcd
        self.vbias += lr_a*(torch.sum(v_pos,1) - torch.sum(v_neg,1))
        self.hbias += lr_b*(torch.sum(h_pos,1) - torch.sum(h_neg,1))
    '''
    
    # Compute positive and negative term
    def fit_batch(self,X):
        _, h_pos = self.SamplerHiddens(X)
        _,_,self.X_persistent_chain,h = self.GetAv()
        self.updateWeights(X,h_pos,self.X_persistent_chain,h)

    '''
    def fit_batch_fields(self,X):
        _, h_pos = self.SamplerHiddens(X,W,b)
        _,_,self.X_persistent_chain,h = self.GetAv(self.X_persistent_chain)
        self.updateFields(X,h_pos,self.X_persistent_chain,h)
    '''
    # return the mininbatch
    # WARNING: does not handle the case where Ns % n_mb != 0
    def getMiniBatches(self,X,m):
        return X[:,m*self.mini_batch_size:(m+1)*self.mini_batch_size]

    # Iterating nMF fixed point
    def SamplingMF(self,X,it_mcmc=0):
        if it_mcmc==0:
            it_mcmc = self.gibbs_steps
        _,mh = self.SamplerHiddens(X)
        _,mv = self.SampleVisibles01(mh)

        for t in range(it_mcmc):
            _,mh = self.SamplerHiddens(mv)
            _,mv = self.SampleVisibles01(mh)
        
        return mv,mh

    def fit(self,X,v_pos,epochs_max=0):
        if epochs_max==0:
            epochs_max = self.epochs_max
        # Number of batches
        NB = int(X.shape[1]/self.mini_batch_size)
        '''
        if self.up_tot == 0:
            f = h5py.File('AllParameters.h5','w')
            f.create_dataset('alltime',data=self.list_save_time)
            f.close()
        '''
        # for t in tq.tqdm(range(epochs_max)): # Uncomment if using notebook
        for t in tqdm(range(epochs_max)): # Uncomment if using terminal
            #self.free_energies_.append(torch.mean(self.free_energy(X)).item())


            # Permute data            
            Xp = X[:,torch.randperm(X.size()[1])]
            for m in range(NB):
                Xb = self.getMiniBatches(Xp,m)
                # Calculate these matrices just once per batch train step
                if self.Symmetry_training:
                    self._W_v_temp = self._get_W_v_matrix(self.W)
                    self._W_h_temp = self._W_v_temp.clone().t()
                self.fit_batch(Xb)

                '''
                if self.up_tot in self.list_save_time:
                    print('Check cycle')
                    f = h5py.File('AllParameters.h5','a')
                    print('param'+str(self.up_tot))
                    f.create_dataset('paramW'+str(self.up_tot),data=self.W.cpu())
                    f.create_dataset('paramVB'+str(self.up_tot),data=self.vbias.cpu())
                    f.create_dataset('paramHB'+str(self.up_tot),data=self.hbias.cpu())
                    f.close()
                
                self.up_tot += 1
                '''
            # See W_matrix definition at beginning
            # self.W_matrix[self.total_epochs,:] = self.W.cpu().flatten()
            self.total_epochs += 1
            if self.total_epochs%self.Saving_interval==0:
                self.energies_train.append(torch.mean(self.energy(X)).item())
                self.energies_test.append(torch.mean(self.energy(v_pos)).item())
                self.log_likelihoods_train.append(self.log_likelihood(X))
                self.log_likelihoods_test.append(self.log_likelihood(v_pos))
                if self.Symmetry_training:
                    self.list_of_singular_values_by_cell.append(torch.linalg.svd(self.W)[1])
                self.list_of_W_matrices.append(self.W.clone())
                self.list_of_hbias.append(self.hbias.clone())
                self.list_of_vbias.append(self.vbias.clone())
            
            
            
        print("Ran for ",self.total_epochs, 'epochs')
        
    # Partially optimized function to calculate energy
    def energy(self, visible):
        hidden,_ = self.SamplerHiddens(visible)
        if not self.Symmetry_training:
            return - (visible.t().mm(self.W.t()) * hidden.t()).sum(dim=1) - torch.matmul(visible.t(), self.vbias) - torch.matmul(hidden.t(), self.hbias)
        if self.Symmetry_training:
            return - (torch.matmul(self._W_v_temp, visible) * hidden).sum(dim=0) - torch.matmul(visible.t(), self.vbias.repeat(self.Symmetry_cells)) - torch.matmul(hidden.t(), self.hbias.repeat(self.Symmetry_cells))
            
    def free_energy(self, visible):
        v_term = torch.matmul(visible.t(), self.vbias)
        w_x_h = torch.nn.functional.linear(visible.t(), self.W, self.hbias)
        # Softplus does log(1+x)
        h_term = torch.sum(torch.nn.functional.softplus(w_x_h), dim=1)
        return -h_term - v_term
    
    def log_likelihood(self, visible):
        energy = self.energy(visible)
        return -energy.mean() - torch.logsumexp(-energy,0) - torch.tensor(np.log(visible.shape[1]))
    
    def ComputeFreeEnergyAIS(self,nβ,nI):
        # nI should be the number of data points
        βlist = torch.arange(0,1.000001,1.0/nβ)
        x = torch.zeros(self.Nv,nI,device=self.device)
        H = torch.zeros(self.Nh,nI,device=self.device)
        E = torch.zeros(nI,device=self.device)

        # initialize xref
        x = torch.bernoulli(torch.sigmoid(self.vbias_prior).repeat(nI,1).t())
        H = torch.bernoulli(torch.rand((self.Nh,nI),device=self.device))
        E = self.computeE(x,H).double().to(self.device)  - self.computeE_prior(x)
        self.V_STATE = x
        self.H_STATE = H
        for idβ in range(1,nβ+1):
            H, _ = self.SamplerHiddens(x,β=βlist[idβ])
            x, _ = self.SampleVisibles01(H,β=βlist[idβ])
            E += self.computeE(x,H)

        Δ = 0
        # E = self.computeE(x,H) - self.computeE_prior(x)
        Δβ = 1.0/nβ
        Δ = -Δβ*E # torch.sum(E,1)
        #for idβ in range(nβ):
        #    Δβ = 1/nβ
        #    Δ += -Δβ*E[:,idβ]
        Δ = Δ.double()
        Δ0 = torch.mean(Δ)

        AIS = torch.log(torch.mean(torch.exp(Δ-Δ0).double()))+Δ0
        # AIS = torch.log(torch.mean(torch.exp(Δ)))
        return AIS
