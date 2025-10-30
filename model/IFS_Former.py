import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import math

# blocks

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        heads = 8,
        dim = 64,
        dropout = 0.,
        inner_dim = 0, 
    ):
        super().__init__()
        self.heads = heads
        if inner_dim == 0:
            inner_dim = dim
        self.scale = (inner_dim/heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, attn_out = False):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            mask_value = -torch.finfo(sim.dtype).max
            attn_mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(attn_mask, mask_value)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        if attn_out:
            return out, attn
        else:
            return out

class MemoryBlock(nn.Module):
    def __init__(
            self, 
            token_num, 
            heads, 
            dim, 
            attn_dropout, 
            cluster, 
            target_mode, 
            groups, 
            num_per_group,
            use_cls_token,
            sum_or_prod = None,
            qk_relu = False) -> None:
        super().__init__()
        if num_per_group == -1:
            self.num_per_group = -1
        else:
            self.num_per_group = max(math.ceil(token_num/groups), num_per_group)
            num_per_group = max(math.ceil(token_num/groups), num_per_group)
            self.gather_layer = nn.Conv1d((groups+int(use_cls_token)) * num_per_group, groups+int(use_cls_token), groups=groups+int(use_cls_token), kernel_size=1)
        self.soft = nn.Softmax(dim=-1)
        self.qk_relu = qk_relu
        self.dropout = nn.Dropout(attn_dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        ) 
        self.groups = groups
        self.use_cls_token = int(use_cls_token)
        self.heads = heads
        self.target_mode = target_mode
        self.cluster = cluster
        if cluster :
            if target_mode == 'mix':
                self.target_token = nn.Parameter(torch.rand([groups, dim]))
                self.to_target = nn.Linear(groups+token_num+int(use_cls_token), groups+int(use_cls_token))
            else:
                self.target_token = nn.Parameter(torch.rand([groups+int(use_cls_token), dim]))
        if sum_or_prod not in ['sum', 'prod']:
            raise ValueError
        self.sum_or_prod = sum_or_prod
        self.scale = dim/heads

    def forward(self, x, mask=None, return_attention=False):
        b,l,d = x.shape
        h = self.heads
        if self.sum_or_prod == 'prod':
            x = torch.log(nn.ReLU()(x) + 1)
        target = self.target_token
        target = target.reshape(1, -1, d).repeat((b,1,1))
        if self.cluster:
            if self.target_mode == 'mix':
                target = torch.cat([target, x], dim=-2)
                target = self.to_target(target.transpose(-1, -2)).transpose(-1, -2)
            q = self.q(target)
        else:
            q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
        k = k.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, h, d//h).permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-1,-2)) * (self.scale ** -0.5)

        
        if mask is not None:
            mask_value = -torch.finfo(attn_scores.dtype).max
            attn_mask = rearrange(mask, 'b j -> b 1 1 j')
            attn_scores = attn_scores.masked_fill(attn_mask, mask_value)

        attn = self.soft(attn_scores)
        attn = self.dropout(attn)
        if self.num_per_group == -1:
            x = einsum('b h i j, b h j d -> b h i d', attn, v)
        else:
            value, idx_original = torch.topk(attn, dim=-1, k=self.num_per_group)
            idx = idx_original.unsqueeze(-1).repeat((1,1,1,1,d // h))
            vv = v.unsqueeze(-2).repeat((1,1,1,self.num_per_group,1))
            xx_ = torch.gather(vv, 2, idx)
            x = self.gather_layer(xx_.reshape(b*h, -1, d//h)).reshape(b, h, -1, d//h)
        if self.sum_or_prod == 'prod':
            x = (x - x.min())/ (x.max()-x.min())
            x = torch.exp(x)
        out = rearrange(x, 'b h n d -> b n (h d)', h = h)
        out = self.out(out)

        if return_attention:
            return out, attn
        else:
            return out

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        attn_dropout,
        ff_dropout,
        use_cls_token,
        groups,
        sum_num_per_group,
        prod_num_per_group,
        cluster,
        target_mode,
        token_num,
        token_descent=False,
        use_prod=True,
        qk_relu = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        flag = int(use_cls_token)
        if not token_descent:
            groups = [token_num for _ in groups]
        for i in range(depth):
            token_num_use = token_num if i == 0 else groups[i-1]
            self.layers.append(nn.ModuleList([
                MemoryBlock(
                    token_num=token_num_use, 
                    heads=heads, 
                    dim=dim, 
                    attn_dropout=attn_dropout, 
                    cluster=cluster, 
                    target_mode=target_mode, 
                    groups=groups[i], 
                    num_per_group=prod_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='prod',
                    qk_relu=qk_relu) if use_prod else nn.Identity(),
                MemoryBlock(
                    token_num=token_num_use, 
                    heads=heads, 
                    dim=dim, 
                    attn_dropout=attn_dropout, 
                    cluster=cluster, 
                    target_mode=target_mode, 
                    groups=groups[i], 
                    num_per_group=sum_num_per_group[i],
                    use_cls_token=use_cls_token,
                    sum_or_prod='sum',
                    qk_relu=qk_relu) if token_descent else Attention(heads=heads, dim=dim, dropout=attn_dropout),
                nn.Linear(2*(groups[i] + flag), groups[i] + flag),
                nn.Linear(token_num_use + flag, groups[i] + flag) if token_descent else nn.Identity(),
                FeedForward(dim, dropout = ff_dropout),
            ]))   
        self.use_prod = use_prod

    def forward(self, x, mask=None, return_attention=False):
        all_attentions = []

        for toprod, tosum, down, downx, ff in self.layers:
            if isinstance(tosum, Attention):
                attn_out, attn_map = tosum(x, mask=mask, attn_out=True)
            else: # MemoryBlock
                attn_out, attn_map = tosum(x, mask=mask, return_attention=True)
            if return_attention:
                all_attentions.append(attn_map)
            if self.use_prod:
                prod = toprod(x, mask=mask)
                attn_out = down(torch.cat([attn_out, prod], dim=1).transpose(2,1)).transpose(2,1)

            x = attn_out + downx(x.transpose(-1, -2)).transpose(-1, -2)
            x = ff(x) + x

        if return_attention:
            return x, all_attentions
        else:
            return x

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class IFS_Former(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim=192,
        dim_out=1,
        depth=6,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
        use_cls_token=True,
        groups=[128, 128, 128, 128, 128, 128],
        sum_num_per_group=[32, 32, 32, 32, 32, 32],
        prod_num_per_group=[4, 4, 4, 4, 4, 4],
        cluster=True,
        target_mode='mix',
        token_descent=False,
        use_prod=True,
        num_special_tokens=2,
        use_sigmoid=True,
        qk_relu=False,
        mask_specific_columns: list = None  
    ):
        super().__init__()
        num_categories = len(categories)  
        self.num_categories = num_categories
        self.num_continuous = num_continuous
        num_features = num_categories + num_continuous
        token_num = num_features

        self.use_cls_token = use_cls_token
        self.dim_out = dim_out
        self.use_sigmoid = use_sigmoid

        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert num_features > 0, 'input shape must not be null'

        self.num_unique_categories = sum(categories)
        
        if self.num_categories > 0:
            total_tokens = self.num_unique_categories + num_special_tokens + 1
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.missing_feature_embedding = nn.Parameter(torch.randn(num_features+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        if mask_specific_columns is not None and num_features > 0:
            assert all(0 <= i < num_features for i in mask_specific_columns), "column index out of range!"
            static_mask = torch.zeros(num_features, dtype=torch.bool)
            static_mask[mask_specific_columns] = True
            self.register_buffer('static_column_mask', static_mask, persistent=False)
        else:
            self.register_buffer('static_column_mask', torch.zeros(num_features, dtype=torch.bool), persistent=False)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups,
            sum_num_per_group=sum_num_per_group,
            prod_num_per_group=prod_num_per_group,
            cluster=cluster,
            target_mode=target_mode,
            token_num=token_num,
            token_descent=token_descent,
            use_prod=use_prod,
            qk_relu=qk_relu,
        )

        self.pool = nn.Linear(num_features, 1)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim_out)
        )
    
    def forward(self, x_categ, x_numer, return_attention=False):
        b = x_categ.shape[0] if self.num_categories > 0 else x_numer.shape[0]
        device = x_categ.device if self.num_categories > 0 else x_numer.device
        
        xs = []

        if self.num_categories > 0:
            categ_mask = (x_categ == -5)
            x_categ_imputed = torch.where(categ_mask, torch.zeros_like(x_categ), x_categ).long()
            embedded_categ = self.categorical_embeds(x_categ_imputed + self.categories_offset)
            missing_categ_embeds = self.missing_feature_embedding[:self.num_categories].unsqueeze(0).expand(b, -1, -1)
            final_categ_embed = torch.where(categ_mask.unsqueeze(-1), missing_categ_embeds, embedded_categ)
            xs.append(final_categ_embed)

        if self.num_continuous > 0:
            numer_mask = torch.isnan(x_numer)
            x_numer_imputed = torch.nan_to_num(x_numer, nan = 0.0)
            embedded_numer = self.numerical_embedder(x_numer_imputed)
            missing_numer_embeds = self.missing_feature_embedding[self.num_categories:-1].unsqueeze(0).expand(b, -1, -1)
            final_numer_embed = torch.where(numer_mask.unsqueeze(-1), missing_numer_embeds, embedded_numer)
            xs.append(final_numer_embed)

        x = torch.cat(xs, dim = 1)
        preserved_placeholders = x[:, self.static_column_mask].detach()
        attention_mask = self.static_column_mask.unsqueeze(0).expand(b, -1)
        placeholder_mask_for_restoration = self.static_column_mask
        
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            placeholder_cls_mask = torch.zeros(1, dtype=torch.bool, device=device)
            placeholder_mask_for_restoration = torch.cat((placeholder_cls_mask, placeholder_mask_for_restoration))
        transformer_output = self.transformer(x, mask=attention_mask, return_attention=return_attention)

        if return_attention:
            x, attentions = transformer_output
        else:
            x = transformer_output

        x[..., placeholder_mask_for_restoration, :] = preserved_placeholders # back to initial embeddings

        if self.use_cls_token:
            x = x[:, 0]
        else:
            if attention_mask is not None:
                valid_tokens_mask = ~attention_mask
                x_masked = x.masked_fill(~valid_tokens_mask.unsqueeze(-1), -1e9) 
                x = self.pool(x_masked.transpose(-1, -2)).squeeze(-1)
            else:
                x = self.pool(x.transpose(-1, -2)).squeeze(-1)

        logit = self.to_logits(x)

        if return_attention:
            return logit, attentions
        else:
            return logit