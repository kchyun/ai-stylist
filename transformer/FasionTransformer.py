import torch
import torch.nn as nn

class FashionTransformer(nn.Module):
    def __init__(self, ModelArgs, device):
        super().__init__()
        self.modelargs = ModelArgs
        self.mlm_probability = 0.15
        self.special_tokens = {"[MASK]" : torch.LongTensor(0).to(device), "[CLS]" : torch.LongTensor(1).to(device)}
        self.mask_token = nn.Embedding(2, 512)
        self.device = device
        
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncoderLayer, 1) 
        self.fc1 = nn.Linear(512, 256)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(256, 1)

    def get_mask_tokens(self, inputs):
        '''
        MLM방식을 사용한다는 가정하에 작성된 코드입니다

        inputs : (N, C, 512) N은 Batch_size, C는 category 개수를 의미함. 

        outputs 
            inputs : (N, C)의 형태로, Mask하기로 한값은,
            labels : (N, C)의 형태로, 변경되지 않는 label은 -100으로 설정된다.

        '''    
        N, C, S = inputs.shape
        
        mask = torch.zeros(N,C)
        labels = inputs.clone() # labels는 임시로 넣은 것이라 조만간 빠질겁니당

        
        
        probability_matrix = torch.full(torch.shape, self.mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 

        indices_replaced = torch.bernoulli(torch.full(mask.shape, 0.8)).bool() & masked_indices
        apply_indices_replaced = indices_replaced.unsqueeze(-1).repeat(1,1,S)
        inputs[apply_indices_replaced] = self.mask_token(torch.Tensor(self.special_tokens['[MASK]']).to(self.device))

        '''
        Random하게 embedding 바꾸는 것입니다. (MLM의 15퍼센트는 이렇게 바꾼다길래 해보았습니다.)

        indices_random = torch.bernoulli(torch.full(mask.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        apply_indices_random = indices_random.unsqueeze(-1).repeat(1,1,S)
        # 파일 가져오기
        #random_embeds  #random으로 바꿔줄 값을 설정해줍시다.
        inputs[apply_indices_random] = random_embeds[random_embeds]
        '''

        return inputs, labels

        
    def concat_embed(self, embeds, context_embeds=None): 
        '''
        

        input
            embeds : (N, 512)의 Tensor c-1개로 이루어진 list
            context_embeds : 말 그대로 context_embeds

        output
            concat_embeds : (C, N, 512)로 값을 넣음. 이때 embeds에 None이 있다면, Mask token으로 넣어줌.
        '''

        ### token에 대한 이해가 적기 때문에 이상하면 말해주세요!

        N, D = 2, 512
        concat_embed = None
        concat_embed = self.mask_token(self.special_tokens["[CLS]"].repeat(N)).reshape(1,N,D)

        if context_embeds == None :#실험 전 용도로 만들었습니다. 아직 context_embeds가 없기 때문에
            #concat_embed = torch.cat([concat_embed, self.mask_token(self.special_tokens['[MASK]'].repeat(N)).reshape(1, N, D)], dim = 0)
            #mask에 대한 학습이 아직 안이루어졌기 때문에 위 코드는 잠시 묶어두겠습니다.

            for embed in embeds:
                if embed is None:
                    embed = self.mask_token(self.special_tokens['[MASK]'].repeat(N)).reshape(1, N, D)
                else:
                    embed = embed.reshape(1, N, D)

                concat_embed = torch.cat([concat_embed, embed], dim = 0)

        else:
            concat_embed = torch.cat([concat_embed, context_embeds.reshape(1,N,D)], dim=0)
            for embed in embeds:
                if embed is None:
                    embed = self.mask_token(self.special_tokens['[MASK]'].repeat(N)).reshape(1, N, D)
                else:
                    embed = embed.reshape(1, N, D)

                concat_embed = torch.cat([concat_embed, embed], dim = 0)
        
        return concat_embed # (C, N, 512로 return)
    def forward(self, embeds, context_embeds = None, mlm = False):
        
        concat_embeds = self.concat_embed(embeds, context_embeds)
        C, N, S = concat_embeds.shape

        if mlm:
            
            mask_inputs, labels = self.get_mask_tokens(concat_embeds)
            output = self.transformerEncoder(mask_inputs) #이미 자체로 mask를 주었기 때문에, 굳이 또 하지 않았습니다.
            
            idx = (labels != -100)
            labels = labels[idx].to(self.device)
            output = output[idx].to(self.device) #MASK 처리 된거에 대해서만 학습을 진행해줍니다.
            
            return labels, output
        
        else:

            output = self.transformerEncoder(concat_embeds) #이미 자체로 mask를 주었기 때문에, 굳이 또 하지 않았습니다.
            output = output[0, :, :]
            
            output = self.fc1(output)
            output = self.gelu(output)
            output = self.fc2(output)


            return None, output

       
            