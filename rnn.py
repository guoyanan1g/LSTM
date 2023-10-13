import torch
from torch import nn



class LSTM(nn.Module):
    def __init__(self,vocalb_size,embedding_dim,hidden_dim,out_dim):
        super(LSTM,self).__init__()
        self.embedding=nn.EmbeddingBag(vocalb_size,embedding_dim,mode='sum')
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers=2,bidirectional=False,dropout=0.5)
        self.fc=nn.Linear(hidden_dim,out_dim) #全连接层，生成输出
        #self.dropout=nn.Dropout(0.5)

    def forward(self,x,offset):
        embed_x=self.embedding(x,offset)
        out,(ht,ct)=self.lstm(embed_x)
        #h/c:[num_layer,b,h]
        #out:[seq,b,h]
        return self.fc(out[-1,:,:])


if __name__=='__main__':
    model=LSTM(10000,100,20,4)
