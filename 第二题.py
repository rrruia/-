import torch
import math
import torch.nn.functional as F

def self_attention(
        query:torch.Tensor,
        key:torch.Tensor,
        value:torch.Tensor
) -> tuple[torch.Tensor,torch.Tensor]:
        
        if query.dim()==2:
                query=query.unsqueeze(0)
                key=key.unsqueeze(0)
                value=value.unsqueeze(0)
                add_batch=True
        else:
                add_batch=False

        B,l_q,d_k=query.shape

        scale=math.sqrt(d_k)
        logits=torch.matmul(query,key.transpose(-1,-2))/scale
        weights=F.softmax(logits,dim=-1)
        output=torch.matmul(weights,value)

        if add_batch==True:
                output=output.squeeze(0)
                weights=weights.squeeze(0)

        return output,weights