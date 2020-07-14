import torch.optim as optim
import torch
import torch.nn as nn

def get_optimizer(model, lr):
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    return optimizer
    
def loss_fn():
    
    def covariance(m):

        m = m.t()
        m_mean = m-torch.mean(m,dim=1,keepdim=True)
        mt = m.t()
        return (m_mean.matmul(mt)/m.shape[1]).squeeze()

    def cos_similarity():
    
        return nn.CosineSimilarity(dim=1)
    
    def custom_loss(encoder_output, num_augments, trade_off):
        
        image_loss = []
#        for i in range(int(encoder_output.shape[0]/num_augments)):
        for i in range(encoder_output.shape[0]):
        
            encoder_output_image = encoder_output[i,:,:]
            
            loss = torch.sum(torch.norm(encoder_output_image-torch.mean(encoder_output_image,dim=0),p=2,dim=1) + \
                                        trade_off*torch.norm(covariance(encoder_output_image)-torch.eye(encoder_output_image.shape[1]), p='fro')) / \
                                        num_augments
                             
            image_loss.append(loss)
        
        return torch.stack(image_loss)
    
    def contrastive_loss(encoder_output, num_augments, temperature):
        
        encoder_output = encoder_output.view(encoder_output.shape[0]*encoder_output.shape[1],2048)
        cos_sim = cos_similarity()
        single_loss = []
        image_loss = []
        counter = 0
        for i in range(encoder_output.shape[0]):
                
            loss = torch.sum(-torch.log(torch.exp((cos_sim(torch.cat((encoder_output[:i],encoder_output[i+1:])), encoder_output[i].unsqueeze(0))/temperature)[counter*num_augments:counter*num_augments+4]) / \
                                        torch.sum(torch.exp(cos_sim(torch.cat((encoder_output[:i],encoder_output[i+1:])), encoder_output[i].unsqueeze(0))/temperature))))
            
            single_loss.append(loss)
            
            if i%num_augments == num_augments-1:
                counter += 1
                image_loss.append(torch.sum(torch.stack(single_loss)))
                single_loss = []
            
        return torch.stack(image_loss)
             
    def cum_loss(encoder_output, num_augments, trade_off = 0.25, temperature = 0.05):
#        
#        custom = custom_loss(encoder_output, num_augments, trade_off)    
#        contrastive = contrastive_loss(encoder_output, num_augments, temperature)
        
        return torch.sum(custom_loss(encoder_output, num_augments, trade_off) + contrastive_loss(encoder_output, num_augments, temperature))

    return cum_loss
    
    