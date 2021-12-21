import math
import time
import torch
from torch import nn
import torch.linalg as la

from functools import partial
from models.attention import SoftmaxAttention as SelfAttention
from models.attention_informer import ProbAttention as InformerAttention


def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0
    
def default(val, d):
    return val if exists(val) else d

def original_attention(q, k, v, mask): # for SM kernel
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(q, k.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :])
    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    out = torch.matmul(attention_probs, v)
    
    return out
    
def linear_attention(q, k, v): # for SM kernel
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    
    # temp = torch.einsum('...n,...nd,...md->...nm', D_inv, q, k)
    # print("Sketched SM:", la.norm(temp, ord=2, dim=(2,3)).mean())
    # del temp
    
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out
    
def kernel_SM(X1, X2=None, X2_accu=False):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...mnd', X1, X2)
        # product = torch.matmul(X1.unsqueeze(dim=2), torch.transpose(X2, 3, 4))
        result = torch.exp(product)
        
        result = result.sum(dim=2)
        # print(result.shape)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)
        
    return result
    # return result, product
    
    
def sketched(V, sketching_matrix, accu=True, random_sign=None):
    # compute S^T V, V: n x p
    B, H, n, p = V.shape
    if accu:
        # S: n x m x d
        STV = V[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], sketching_matrix] # bhmdp
        STV = torch.einsum('...mdp,...md->...dp', STV, random_sign)
    
    return STV
        
def kernel_RS_SM1(X1, X2=None, X2_accu=False, random_sign=None):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...nmd', X1, X2)
        # result = torch.exp(product - product.max())
        result = torch.exp(product)
        result = torch.einsum('bhnmd,...bmd->...bhnd', result, random_sign)
        # result = (result.transpose(0, 2) * random_sign).sum(-2).transpose(0, 2) # nhbmd -> nhbd -> bhnd

    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result   

def kernel_RS_SM(X1, X2=None, X2_accu=False, random_sign=None):
    # compute AS with random sign for softmax kernel
    if X2 is None:
        X2 = X1
        X2_accu = False
        
    if X2_accu:
        product = torch.einsum('...np,...mdp->...nmd', X1, X2)
        result = torch.exp(product)
        
        result = torch.einsum('...nmd,...md->...nd', result, random_sign)
        # result = result.sum(dim=3)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)
        
    return result
    
def kernel_RBF(X1, X2=None, X2_accu=False):

    # todo
    
    if X2 is None:
        X2 = X1
        X2_accu = False
    
    diag_X1 = torch.abs(X1) ** power
    diag_X1 = torch.sum(diag_X1, dim=-1) / scale
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = torch.abs(X2) ** power
    diag_X2 = torch.sum(diag_X2, dim=-1) / scale
    diag_X2 = diag_X2.unsqueeze(dim=-2)
    
    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        result = result.sum(dim=2)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        
    return result
    
def kernel_RS_RBF(X1, X2=None, X2_accu=False, random_sign=None):

    # todo
    power=2
    
    if X2 is None:
        X2 = X1
        X2_accu = False
    
    diag_X1 = torch.abs(X1) ** power
    diag_X1 = torch.sum(diag_X1, dim=-1) / scale
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = torch.abs(X2) ** power
    diag_X2 = torch.sum(diag_X2, dim=-1) / scale
    diag_X2 = diag_X2.unsqueeze(dim=-2)
    
    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = result * random_sign
        result = result.sum(dim=3)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        
    return result

def rbf_attention(q, k, v): # for rbf kernel
    # todo
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def kernel_sketch(q, k, *, kernel_fn, sketching_matrix, random_sign, normalize_data=False, eps=1e-4, device = None):
    # sketching_matrix: (self.M, self.d) tensor
    # sketching_matrix
    
    b, h, n, p = q.shape

    # data_normalizer = (p ** -0.25) if normalize_data else 1.
    # X = torch.cat([q, k], dim=2) * data_normalizer
    X = torch.cat([q, k], dim=2)
    
    AS = kernel_fn(X, X[:,:,sketching_matrix], X2_accu=True, random_sign=random_sign)
    # AS, p = kernel_fn(X, X[:,:,sketching_matrix], True)
    # if True in torch.isinf(AS):
        # print()
    # if True in torch.isnan(AS):
        # print()

    return AS.type_as(q)

def kernel_skein(q, k, *, kernel_fn, sketching_matrix, random_sign, normalize_data=False, eps=1e-4, device = None):
    
    B, H, n, p = q.shape
    AS = kernel_fn(q, k[torch.arange(B)[:, None, None, None], 
        torch.arange(H)[None, :, None, None], sketching_matrix], X2_accu=True, random_sign=random_sign)

    return AS.type_as(q)


    
def uniform_sketching(n, nb_rows, nb_columns):
    w = torch.ones(n)
    S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(nb_rows, nb_columns)
    random_sign = (torch.randint(2, S.shape) * 2 - 1) * math.sqrt(n / nb_rows / nb_columns)
    return S, random_sign
    
def importance_sketching(prob, nb_rows, nb_columns):
    # Output: S, random_sign: BHmd

    B, H, n = prob.shape
    sample_shape = (B, H, nb_rows, nb_columns)
    
    # if len(prob.shape) == 3: 
    prob = prob.reshape(B*H, n)
    w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
    S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
    # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
    w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
        torch.arange(H)[None, :, None, None], S]
    # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
    random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
    return S, random_sign


def pinv(X, hermitian = True, eps = 1e-4):
    if hermitian:
        Sigma, U = torch.symeig(X, eigenvectors=True)
        Sigma = torch.where(Sigma > eps, 1 / Sigma, torch.tensor(0.))
        # print(U.shape, Sigma.shape)
        res = torch.einsum('...md,...nd,...d->...mn', U, U, Sigma)
        return res
    else:
        return torch.pinverse(X)

def exp_kernel(q, k):
    res = torch.einsum('...np,...mp->...nm', q, k)
    res = torch.exp(res)
    
    return res

   

class SkeinAttention_ablation_nopilot(nn.Module):


    def __init__(self, config):
        super().__init__()

        self.device = config["device"]  if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM1

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns):
        S = torch.rand(nb_rows, nb_columns, device=self.device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()
        # random_sign = (torch.randint(2, S.shape) * 2 - 1)
        random_sign = torch.ones(S.shape, device=self.device)
        random_sign = torch.einsum('bmd,b->bmd', random_sign, 
            torch.sqrt(non_padding_num / nb_rows / nb_columns))
        
        return S, random_sign
   
    @torch.no_grad()   
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # Output: S, random_sign: BHmd

        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)
        
        # if len(prob.shape) == 3: 
        prob = prob.reshape(B*H, n)
        w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
        # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
        w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], S]
        # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
        random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v, mask):
        # Without row information, to show column is better than row in Informer
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * mask[:, None, :, None] * data_normalizer
        k = k * mask[:, None, :, None] * data_normalizer
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        nb_features = min(self.nb_features, torch.min(non_padding_num).long().item()-1)
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
    
            S0, rs0 = self.uniform_sketching(non_padding_num, 1, nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp
            QS0 = q.transpose(1, 2)[torch.arange(b)[:, None, None], S0].permute(0,3,1,2,4) # bmdhp -> bhmdp

            ATS0 = self.kernel_fn(k, QS0, True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd

            ATS0 = ATS0 * mask[:, None, :, None]

            D_inv0_partial = 1 / ATS0.sum(-2) # bhd

            
            Dinv_S0TA = torch.einsum('...d,...nd->...dn',  D_inv0_partial, ATS0) # bhdn
            

            prob_AV = torch.sqrt((Dinv_S0TA * Dinv_S0TA).sum(-2) * (v*v).sum(-1))
            
            ################################################## 
        
            S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows = self.accumulation, 
                nb_columns = nb_features)
                
            S1TV = v[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] # bhmdp
            
            K1 = k[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] #bhmdp
            

            qK1 = torch.einsum('...np,...mdp->...nmd', q, K1)
            AS1 = torch.exp(qK1) # bhnmd
             
            AV1 = torch.einsum('...nmd,...mdp->...np', AS1, S1TV) # bhnp
            A1_sum = AS1.reshape(b,h,n,-1).sum(-1) # bhn
            
            
            ##################################################
            
            model_column = torch.exp(qK1.reshape(b,h,n,-1).mean(-1))
            
            # print(model_column.shape, v.sum(-2).shape)
            
            V_sum = torch.einsum("...n,...p->...np", model_column, v.sum(-2)) # bhnp
            V1_sum = torch.einsum("...n,...p->...np", model_column, S1TV.reshape(b,h,-1,p).sum(-2))

            ##################################################
            
            D1 = A1_sum + torch.einsum('bhn,b->bhn', model_column, non_padding_num-nb_features) #bhn
                
            out1 = AV1 + (V_sum - V1_sum) # bhnp
            out1 = torch.einsum('...n,...np->...np', 1/D1, out1)
            
        return out1


class SkeinAttention_ablation_simple_normalization(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config["device"]  if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM1

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns):
        S = torch.rand(nb_rows, nb_columns, device=self.device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()
        # random_sign = (torch.randint(2, S.shape) * 2 - 1)
        random_sign = torch.ones(S.shape, device=self.device)
        random_sign = torch.einsum('bmd,b->bmd', random_sign, 
            torch.sqrt(non_padding_num / nb_rows / nb_columns))
        
        return S, random_sign
   
    @torch.no_grad()   
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # Output: S, random_sign: BHmd

        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)
        
        # if len(prob.shape) == 3: 
        prob = prob.reshape(B*H, n)
        w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
        # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
        w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], S]
        # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
        random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v, mask):
        # Without row information, to show column is better than row in Informer
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * mask[:, None, :, None] * data_normalizer
        k = k * mask[:, None, :, None] * data_normalizer
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        nb_features = min(self.nb_features, torch.min(non_padding_num).long().item()-1)
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
    
            S0, rs0 = self.uniform_sketching(non_padding_num, 1, nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp
            QS0 = q.transpose(1, 2)[torch.arange(b)[:, None, None], S0].permute(0,3,1,2,4) # bmdhp -> bhmdp

            ATS0 = self.kernel_fn(k, QS0, True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd

            ATS0 = ATS0 * mask[:, None, :, None]
            ###
            D_inv0_partial = 1 / ATS0.sum(-2) # bhd

            
            Dinv_S0TA = torch.einsum('...d,...nd->...dn',  D_inv0_partial, ATS0) # bhdn
            
            # out0 = torch.matmul(Dinv_S0TA, v)
            

            prob_AV = torch.sqrt((Dinv_S0TA * Dinv_S0TA).sum(-2) * (v*v).sum(-1))
            
            ################################################## 
        
            S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows = self.accumulation, 
                nb_columns = nb_features)
                
            S1TV = v[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1].sum(-3) # bhmdp -> bhdp
            
            K1 = k[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] #bhmdp
            

            qK1 = torch.einsum('...np,...mdp->...nmd', q, K1).sum(-2) # bhnmd
            # AS1 = torch.exp(qK1) # bhnd
            
            attn = nn.functional.softmax(qK1, dim = -1) # bhnd
            # attn = self.drop_attn(attn)
            out1 = torch.matmul(attn, S1TV)
            
        return out1

class SkeinAttention_ablation_no_normalization(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config["device"]  if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM1

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns):
        S = torch.rand(nb_rows, nb_columns, device=self.device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()
        # random_sign = (torch.randint(2, S.shape) * 2 - 1)
        random_sign = torch.ones(S.shape, device=self.device)
        random_sign = torch.einsum('bmd,b->bmd', random_sign, 
            torch.sqrt(non_padding_num / nb_rows / nb_columns))
        
        return S, random_sign
   
    @torch.no_grad()   
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # Output: S, random_sign: BHmd

        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)
        
        # if len(prob.shape) == 3: 
        prob = prob.reshape(B*H, n)
        w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
        # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
        w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], S]
        # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
        random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v, mask):
        # Without row information, to show column is better than row in Informer
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * mask[:, None, :, None] * data_normalizer
        k = k * mask[:, None, :, None] * data_normalizer
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        nb_features = min(self.nb_features, torch.min(non_padding_num).long().item()-1)
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
    
            S0, rs0 = self.uniform_sketching(non_padding_num, 1, nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp
            QS0 = q.transpose(1, 2)[torch.arange(b)[:, None, None], S0].permute(0,3,1,2,4) # bmdhp -> bhmdp

            ATS0 = self.kernel_fn(k, QS0, True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd

            ATS0 = ATS0 * mask[:, None, :, None]

            D_inv0_partial = 1 / ATS0.sum(-2) # bhd

            
            Dinv_S0TA = torch.einsum('...d,...nd->...dn',  D_inv0_partial, ATS0) # bhdn
            
            # out0 = torch.matmul(Dinv_S0TA, v)
            

            prob_AV = torch.sqrt((Dinv_S0TA * Dinv_S0TA).sum(-2) * (v*v).sum(-1))
            
            ################################################## 
        
            S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows = self.accumulation, 
                nb_columns = nb_features)
                
            S1TV = v[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1].sum(-3) # bhmdp -> bhdp
            
            K1 = k[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] #bhmdp
            

            qK1 = torch.einsum('...np,...mdp->...nmd', q, K1) # bhnmd
            AS1 = torch.exp(qK1).sum(-2) # bhnd
            
            ##################################################
            
            
            dot = torch.matmul(q, torch.transpose(k, -2, -1))
            dot = dot - 1e9 * (1 - mask[:, None, None, :])           
            D = torch.exp(dot).sum(-1)
            
            out1 = torch.einsum("...n,...nd,...dp->...np", 1/D, AS1, S1TV)
            
        return out1
        
class SkeinAttention_ablation_uniform(nn.Module):


    def __init__(self, config):
        super().__init__()

        self.device = config["device"]  if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM1

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns):
        S = torch.rand(nb_rows, nb_columns, device=self.device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()
        # random_sign = (torch.randint(2, S.shape) * 2 - 1)
        random_sign = torch.ones(S.shape, device=self.device)
        random_sign = torch.einsum('bmd,b->bmd', random_sign, 
            torch.sqrt(non_padding_num / nb_rows / nb_columns))
        
        return S, random_sign
   
    @torch.no_grad()   
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # Output: S, random_sign: BHmd

        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)
        
        # if len(prob.shape) == 3: 
        prob = prob.reshape(B*H, n)
        w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
        # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
        w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], S]
        # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
        random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v, mask):
        # Without row information, to show column is better than row in Informer
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * mask[:, None, :, None] * data_normalizer
        k = k * mask[:, None, :, None] * data_normalizer
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        nb_features = min(self.nb_features, torch.min(non_padding_num).long().item()-1)
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            
            prob_AV = torch.einsum("h,bn->bhn", torch.ones(h, device=self.device), mask)
        
            S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows = self.accumulation, 
                nb_columns = nb_features)
                
            S1TV = v[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] # bhmdp
            
            K1 = k[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] #bhmdp
            

            qK1 = torch.einsum('...np,...mdp->...nmd', q, K1)
            AS1 = torch.exp(qK1) # bhnmd
             
            AV1 = torch.einsum('...nmd,...mdp->...np', AS1, S1TV) # bhnp
            A1_sum = AS1.reshape(b,h,n,-1).sum(-1) # bhn
            
            
            ##################################################
            
            model_column = torch.exp(qK1.reshape(b,h,n,-1).mean(-1))
            
            # print(model_column.shape, v.sum(-2).shape)
            
            V_sum = torch.einsum("...n,...p->...np", model_column, v.sum(-2)) # bhnp
            V1_sum = torch.einsum("...n,...p->...np", model_column, S1TV.reshape(b,h,-1,p).sum(-2))

            ##################################################
            
            D1 = A1_sum + torch.einsum('bhn,b->bhn', model_column, non_padding_num-nb_features) #bhn
                
            out1 = AV1 + (V_sum - V1_sum) # bhnp
            out1 = torch.einsum('...n,...np->...np', 1/D1, out1)
            
        return out1

class SkeinAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device = config["device"]  if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM1

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns):
        S = torch.rand(nb_rows, nb_columns, device=self.device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()
        # random_sign = (torch.randint(2, S.shape) * 2 - 1)
        random_sign = torch.ones(S.shape, device=self.device)
        random_sign = torch.einsum('bmd,b->bmd', random_sign, 
            torch.sqrt(non_padding_num / nb_rows / nb_columns))
        
        return S, random_sign
   
    @torch.no_grad()   
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # Output: S, random_sign: BHmd

        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)
        
        # if len(prob.shape) == 3: 
        prob = prob.reshape(B*H, n)
        w = torch.einsum('...n,...->...n', prob, 1 / prob.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(sample_shape)
        # S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)
        w = w.reshape(B,H,-1)[torch.arange(B)[:, None, None, None], 
            torch.arange(H)[None, :, None, None], S]
        # random_sign = (torch.randint(2, S.shape) * 2 - 1) / torch.sqrt(w * nb_rows * nb_columns)
        random_sign = torch.ones(S.shape, device=self.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v, mask):
        # Without row information, to show column is better than row in Informer
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * mask[:, None, :, None] * data_normalizer
        k = k * mask[:, None, :, None] * data_normalizer
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        nb_features = min(self.nb_features, torch.min(non_padding_num).long().item()-1)
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
    
            S0, rs0 = self.uniform_sketching(non_padding_num, 1, nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp
            QS0 = q.transpose(1, 2)[torch.arange(b)[:, None, None], S0].permute(0,3,1,2,4) # bmdhp -> bhmdp

            ATS0 = self.kernel_fn(k, QS0, True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd

            ATS0 = ATS0 * mask[:, None, :, None]
            ###
            D_inv0_partial = 1 / ATS0.sum(-2) # bhd

            
            Dinv_S0TA = torch.einsum('...d,...nd->...dn',  D_inv0_partial, ATS0) # bhdn
            # [ 0.1986, -0.1007,  0.1131,  ...,  0.0362, -0.0098,  0.1269]
            # [ 0.0355, -0.0180,  0.0202,  ...,  0.0065, -0.0018,  0.0227]
            out0 = torch.matmul(Dinv_S0TA, v)
            # print(S0, out0[0, 0])
            
            

            prob_AV = torch.sqrt((Dinv_S0TA * Dinv_S0TA).sum(-2) * (v*v).sum(-1))
            
            ################################################## 
        
            S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows = self.accumulation, 
                nb_columns = nb_features)
                
            S1TV = v[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] # bhmdp
            
            K1 = k[torch.arange(b)[:, None, None, None], 
                torch.arange(h)[None, :, None, None], S1] #bhmdp
            

            qK1 = torch.einsum('...np,...mdp->...nmd', q, K1)
            AS1 = torch.exp(qK1) # bhnmd
             
            AV1 = torch.einsum('...nmd,...mdp->...np', AS1, S1TV) # bhnp
            A1_sum = AS1.reshape(b,h,n,-1).sum(-1) # bhn
            
            
            ##################################################
            
            model_column = torch.exp(qK1.reshape(b,h,n,-1).mean(-1))
            
            # print(model_column.shape, v.sum(-2).shape)
            
            V_sum = torch.einsum("...n,...p->...np", model_column, v.sum(-2)) # bhnp
            V1_sum = torch.einsum("...n,...p->...np", model_column, S1TV.reshape(b,h,-1,p).sum(-2))

            ##################################################
            
            D1 = A1_sum + torch.einsum('bhn,b->bhn', model_column, non_padding_num-nb_features) #bhn
                
            out1 = AV1 + (V_sum - V1_sum) # bhnp
            out1 = torch.einsum('...n,...np->...np', 1/D1, out1)
            
            out1 = out1.transpose(1, 2)
            out1[torch.arange(b)[:, None], S0.reshape(b, -1)] = out0.transpose(1, 2)
            out1 = out1.transpose(1, 2)
            
            
        return out1
        
        
      
class Vmean(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.device = config["device"] if "device" in config else "cuda"
        n = config["max_seq_len"]
        accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]
        
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features
        self.accumulation = accumulation
        if config["sketched_kernel"] == "kernel_RS_SM":
            self.kernel_fn = kernel_RS_SM

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
        self.no_projection = config["no_projection"]

    @torch.no_grad()
    def redraw_sketching_matrix(self, device):
        sketching, random_sign = self.create_sketching(device = device)
        self.sketching_matrix.copy_(sketching)
        self.random_sign.copy_(random_sign)
        del sketching
        del random_sign

    def forward(self, q, k, v, mask):
        # The same as forward2. Remove the irrelevant part.
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        v = v * mask[:, None, :, None]
        non_padding_num = mask.sum(-1) # b
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            out1 = torch.ones(b, h, n, device=self.device)
            v_mean = torch.einsum('bhp,b->bhp', v.sum(-2), 1/non_padding_num)
            out1 = torch.einsum('...n,...p->...np', out1, v_mean)
            
        return out1
        
    def forward2(self, q, k, v, mask):
        # prob_AV = D_inv0 * torch.sqrt((ATS0*ATS0).sum(-1) * (v*v).sum(-1)) 错误
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * data_normalizer
        k = k * data_normalizer
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            S0, rs0 = uniform_sketching(n, 1, self.nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp            
            ATS0 = self.kernel_fn(k, q[:, :, S0], True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd
            D_inv0_partial = 1 / ATS0.sum(-2) # bhd
            out0 = torch.einsum('...d,...nd,...np->...dp',  D_inv0_partial, ATS0, v)
            
            AS0 = self.kernel_fn(q, k[:, :, S0], True, rs0)
            S0T1 = rs0.sum(-2)
            D_inv0 = 1 / torch.einsum('...nd,...d->...n', AS0, S0T1)
            
            prob_AV = D_inv0 * torch.sqrt((ATS0*ATS0).sum(-1) * (v*v).sum(-1))
            
            ##################################################
        
            self.create_sketching = partial(importance_sketching, 
                prob=prob_AV, nb_rows = self.accumulation, nb_columns = self.nb_features)
            sketching_matrix, random_sign = self.create_sketching()
            self.register_buffer('sketching_matrix', sketching_matrix)
            self.register_buffer('random_sign', random_sign)
           
            STV = sketched(v, sketching_matrix = self.sketching_matrix, 
                    accu=True, random_sign=self.random_sign)
            
            create_kernel_sketch = partial(kernel_skein, kernel_fn = self.kernel_fn, 
                sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
            AS = create_kernel_sketch(q, k)  # b,h,n,nb_feat
            
            ##################################################
            
            ST1 = self.random_sign.sum(-2)
            D_inv1 = 1 / torch.einsum('...nd,...d->...n', AS, ST1)
            ASSTV = torch.matmul(AS, STV)
            
            
            # out1 = torch.einsum('...np,...n->...np', ASSTV, D_inv1)
            out1 = torch.ones(b, h, n)
            out1 = torch.einsum('...n,...p->...np', out1, v.mean(-2))
            
            ''''''
            prob2 = torch.sqrt((ASSTV * ASSTV).sum(-1)) * D_inv1.abs()
            # w = prob2.reshape(b*h, -1)
            # S2 = torch.multinomial(w, self.nb_features, replacement=False).reshape(b, h, -1)
            # Q2 = q[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2]
            # out2 = original_attention(Q2, k, v, mask)
            
            S_sparse = prob2.topk(self.nb_features)[1]
            Q2 = q[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S_sparse]
            out2 = original_attention(Q2, k, v, mask)
            
            ##################################################
            # out1[:, :, S0.reshape(-1)] = out0
            # out1[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2] = out2
            # out1[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S_sparse] = out2
            
        return out1
        
    def forwardi(self, q, k, v):
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * data_normalizer
        k = k * data_normalizer
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            out1 = torch.ones(b, h, n)
            out1 = torch.einsum('...n,...p->...np', out1, v.mean(-2))
            
            S0, rs0 = uniform_sketching(n, 1, self.nb_features) # bhmd
            
            AS0 = self.kernel_fn(q, k[:, :, S0], True, rs0) # bhnd
            S0T1 = rs0.sum(-2)
            D_inv0 = 1 / torch.einsum('...nd,...d->...n', AS0, S0T1)
            
            prob_DiA = D_inv0 * torch.sqrt((AS0*AS0).sum(-1))
            
            w = prob_DiA.reshape(b*h, -1)
            S2 = torch.multinomial(w, self.nb_features, replacement=False).reshape(b, h, -1)
            Q2 = q[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2]
            out2 = original_attention(Q2, k, v, mask)
            
            ##################################################
            out1[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2] = out2
            
        return out1
        
    def forward1(self, q, k, v):
        device = q.device
        b, h, n, p = q.shape
        data_normalizer = (p ** -0.25)
        q = q * data_normalizer
        k = k * data_normalizer
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            S0, rs0 = uniform_sketching(n, 1, self.nb_features) # bhmd
            # Q0 = q[:, :, S0] # bhmdp            
            ATS0 = self.kernel_fn(k, q[:, :, S0], True, rs0).reshape(b, h, n, -1) # bhnmd -> bhnd
            D_inv0_partial = 1 / ATS0.sum(-2) # bhd
            out0 = torch.einsum('...d,...nd,...np->...dp',  D_inv0_partial, ATS0, v)
            
            AS0 = self.kernel_fn(q, k[:, :, S0], True, rs0)
            S0T1 = rs0.sum(-2)
            D_inv0 = 1 / torch.einsum('...nd,...d->...n', AS0, S0T1)
            
            prob_AV = D_inv0 * torch.sqrt((ATS0*ATS0).sum(-1) * (v*v).sum(-1))
            
            ##################################################
        
            self.create_sketching = partial(importance_sketching, 
                prob=prob_AV, nb_rows = self.accumulation, nb_columns = self.nb_features)
            sketching_matrix, random_sign = self.create_sketching()
            self.register_buffer('sketching_matrix', sketching_matrix)
            self.register_buffer('random_sign', random_sign)
           
            STV = sketched(v, sketching_matrix = self.sketching_matrix, 
                    accu=True, random_sign=self.random_sign)
            
            create_kernel_sketch = partial(kernel_skein, kernel_fn = self.kernel_fn, 
                sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
            AS = create_kernel_sketch(q, k)  # b,h,n,nb_feat
            
            ##################################################
            
            ST1 = self.random_sign.sum(-2)
            D_inv1 = 1 / torch.einsum('...nd,...d->...n', AS, ST1)
            ASSTV = torch.matmul(AS, STV)
            out1 = torch.einsum('...np,...n->...np', ASSTV, D_inv1)
            
            ##################################################
            ''''''
            prob2 = torch.sqrt((ASSTV * ASSTV).sum(-1)) * D_inv1.abs()
            w = prob2.reshape(b*h, -1)
            S2 = torch.multinomial(w, self.nb_features, replacement=False).reshape(b, h, -1)
            Q2 = q[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2]
            out2 = original_attention(Q2, k, v, mask)
            
            ##################################################
            out1[:, :, S0.reshape(-1)] = out0
            out1[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], S2] = out2
            
        return out1
        
    def forward0(self, q, k, v):
        device = q.device
        b, h, n, d = q.shape
        data_normalizer = (d ** -0.25)
        q = q * data_normalizer
        k = k * data_normalizer
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        else:
            self.create_sketching = partial(importance_sketching, 
                prob=(v*v).sum(-1), nb_rows = self.accumulation, nb_columns = self.nb_features)
            sketching_matrix, random_sign = self.create_sketching()
            self.register_buffer('sketching_matrix', sketching_matrix)
            self.register_buffer('random_sign', random_sign)
           
            STV = sketched(v, sketching_matrix = self.sketching_matrix, 
                    accu=True, random_sign=self.random_sign)
            
            create_kernel_sketch = partial(kernel_skein, kernel_fn = self.kernel_fn, 
                sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
            AS = create_kernel_sketch(q, k)  # b,h,n,nb_feat
            
            ##################################################
            
            ST1 = self.random_sign.sum(-2)
            D_inv = 1 / torch.einsum('...nd,...d->...n', AS, ST1)
            ASSTV = torch.matmul(AS, STV)
            out1 = torch.einsum('...np,...n->...np', ASSTV, D_inv)
            
            ##################################################
            ''''''
            prob2 = torch.sqrt((ASSTV * ASSTV).sum(-1)) * D_inv.abs()
            B, H, _ = prob2.shape
            w = prob2.reshape(B*H, -1)
            S2 = torch.multinomial(w, self.nb_features, replacement=False).reshape(B, H, -1)
            Q2 = q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], S2]
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores2 = torch.matmul(Q2, k.transpose(-1, -2))
            # Normalize the attention scores to probabilities.
            attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
            out2 = torch.matmul(attention_probs2, v)
            
            ##################################################
            
            out1[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], S2] = out2
            
        return out1



