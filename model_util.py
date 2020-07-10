import torch.optim as optim
import torch

def get_optimizer(model, lr):
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    return optimizer

def cov(m, rowvar=False):

    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1))
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
    
def loss_fn():
    
    def custom_loss(encoder_output, trade_off):
        
        loss = torch.norm(encoder_output-torch.mean(encoder_output,0),p=2,dim=1) + trade_off*torch.norm(cov(encoder_output) - torch.eye(encoder_output.shape[1]), p='fro')
        
        return torch.sum(loss)/encoder_output.shape[0]
    
    return custom_loss
    
    