import torch
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.data.utils import get_tokenizer #分词工具
from torchtext.vocab import  build_vocab_from_iterator #构建词表
from rnn import LSTM
from torch import optim

"""
1.加载数据集。每一条是(label,text)的形式
2.分词器、生成词表
3.对于数据集的每一个text，处理成词向量的形式
4.dataloader，获得label、text、offset三种数据，offset是给embeddingbag用的，能够索引到不同长度的text
5.定义神经网络
"""
device=torch.device('mps') #mac M1
batch_size=10
tokenizer=get_tokenizer('basic_english')
def yield_tokens(train_ier):
    for label,text in train_ier:
        yield tokenizer(text)

train_iter=datasets.AG_NEWS(split='train').__iter__()
print(train_iter.__iter__())
#print(next(train_iter))
vocalb=build_vocab_from_iterator(yield_tokens(train_iter),specials=[])
vocalb.insert_token('<unk>', 0)  # 手动添加 <unk> 标记并设置索引为 0
vocalb.set_default_index(0)
text_process=lambda x:vocalb(tokenizer(x))
def my_collate_fn(batch):
    x=[]
    y=[]
    offset=[]
    for label,text in batch:
        y.append(label)
        tmp=torch.Tensor(text_process(text),dtype=torch.int64)
        x.append(tmp)
        offset.append(tmp.size(0))
    y=torch.Tensor(y,dtype=torch.int64)
    x=torch.cat(x)
    offset=torch.tensor(offset[:-1]).cumsum(dim=0) #累计和，得到每个text的索引
    return x.to(device),y.to(device),offset.to(device)
def main():
    vocal_size=len(vocalb)
    dataloader=DataLoader(train_iter.__iter__(),batch_size,collate_fn=my_collate_fn)#设置shuffle为True会报错

    model=LSTM(vocal_size,100,20,1).to(device)
    total=0
    total_correct=0
    optimizer=optim.Adam(model.parameters(),lr=1e-3)

    criteon=torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(100):
        model.train()
        for idx,(x,y,offset) in enumerate(dataloader):
            optimizer.zero_grad()
            pred=model(x,offset)
            loss=criteon(pred,y)
            loss.backward()
            optimizer.step()
        print(epoch, " train cross entropy:", loss.item())





if __name__=='__main__':
    main()