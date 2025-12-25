import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class CAN_Layer(nn.Module):
    """
    https://github.com/ZhaohanM/FusionDTI/blob/main/utils/metric_learning_models.py
    """
    def __init__(self, hidden_dim, num_heads, agg_mode='mean', group_size=1):
        super(CAN_Layer, self).__init__()
        self.agg_mode = agg_mode #args.
        self.group_size = group_size  #args.  Control Fusion Scale
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Group embeddings before applying multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size)
        
        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)

        # Compute attention scores
        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2
        
        # Continue as usual with the aggregation mode
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(-1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
            
        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        return query_embed


class BANLayer(nn.Module):
    """
    https://github.com/peizhenbai/DrugBAN/blob/main/ban.py
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        logits = torch.clamp(logits, min=-1000000.0, max=1000000.0)  # NOTE: YQY added
        return logits, att_maps


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MAN(torch.nn.Module):
    """
    Modified from https://github.com/yydhYYDH/MutualDTA/blob/master/models/model.py
    """
    def __init__(self, drug_hidden_dim: int = 128, protein_hidden_dim: int = 128):
        super(MAN, self).__init__()
        k, self.beta_p, self.beta_x = 64, 0.5, 0.99

        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(protein_hidden_dim, drug_hidden_dim)))
        self.W_x = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, drug_hidden_dim)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, protein_hidden_dim)))
        self.w_hx = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, 1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, 1)))

    def forward(self, drug: torch.Tensor, target: torch.Tensor):  # drug : B x 45 x 128, target : B x L x 128
        drug = drug.permute(0, 2, 1)

        C = F.tanh(torch.matmul(target, torch.matmul(self.W_b, drug))) # B x L x 45

        H_c = F.tanh(torch.matmul(self.W_x, drug) + torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))  # B x k x 45
        H_p = F.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_x, drug), C.permute(0, 2, 1)))  # B x k x L

        a_c_weight = torch.matmul(torch.t(self.w_hx), H_c)
        a_p_weight = torch.matmul(torch.t(self.w_hp), H_p)

        a_c = F.softmax(a_c_weight, dim=2) # B x 1 x 45
        a_p = F.softmax(a_p_weight, dim=2) # B x 1 x L

        c = torch.squeeze(torch.matmul(a_c, drug.permute(0, 2, 1)))      # B x 128
        p = torch.squeeze(torch.matmul(a_p, target))                  # B x 128

        return c, p


class MANnew(nn.Module):
    """
    Refined MAN with mask.
    """
    def __init__(
        self,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 128,
        k: int = 64,
        beta_p: float = 0.5,
        beta_x: float = 0.99,
        dropout_rate: float = 0.2,
        use_layernorm: bool = True
    ):
        super(MANnew, self).__init__()
        self.k = k
        self.beta_p = beta_p
        self.beta_x = beta_x

        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(protein_hidden_dim, drug_hidden_dim)))
        self.W_x = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, drug_hidden_dim)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, protein_hidden_dim)))
        self.w_hx = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, 1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, 1)))

        self.dropout = nn.Dropout(dropout_rate)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm_drug = nn.LayerNorm(drug_hidden_dim)
            self.norm_target = nn.LayerNorm(protein_hidden_dim)
            self.norm_c = nn.LayerNorm(drug_hidden_dim)
            self.norm_p = nn.LayerNorm(protein_hidden_dim)

    def forward(
        self,
        drug: torch.Tensor,           # (B, D, dim)
        target: torch.Tensor,         # (B, L, dim)
        drug_mask: torch.Tensor = None,   # (B, D), bool or byte
        target_mask: torch.Tensor = None  # (B, L), bool or byte
    ):
        # Optional LayerNorm on input
        if self.use_layernorm:
            drug = self.norm_drug(drug)
            target = self.norm_target(target)

        drug_t = drug.permute(0, 2, 1)  # B x dim x D

        # Cross interaction: attention logits
        C = F.tanh(torch.matmul(target, torch.matmul(self.W_b, drug_t)))  # B x L x D

        # Hidden fusion representations
        H_c = F.tanh(torch.matmul(self.W_x, drug_t) +
                     torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))  # B x k x D

        H_p = F.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) +
                     torch.matmul(torch.matmul(self.W_x, drug_t), C.permute(0, 2, 1)))  # B x k x L

        # Attention logits
        a_c_weight = torch.matmul(self.w_hx.T, H_c)  # B x 1 x D
        a_p_weight = torch.matmul(self.w_hp.T, H_p)  # B x 1 x L

        # === ✅ Dropout → Mask → Softmax ===
        a_c_weight = self.dropout(a_c_weight)
        a_p_weight = self.dropout(a_p_weight)

        if drug_mask is not None:
            a_c_weight = a_c_weight.masked_fill(~drug_mask.unsqueeze(1).bool(), float('-inf'))
        if target_mask is not None:
            a_p_weight = a_p_weight.masked_fill(~target_mask.unsqueeze(1).bool(), float('-inf'))

        a_c = F.softmax(a_c_weight, dim=2)  # B x 1 x D
        a_p = F.softmax(a_p_weight, dim=2)  # B x 1 x L

        # Weighted sum
        c = torch.matmul(a_c, drug).squeeze(1)   # B x dim
        p = torch.matmul(a_p, target).squeeze(1) # B x dim

        if self.use_layernorm:
            c = self.norm_c(c)
            p = self.norm_p(p)

        return c, p, C.clone().detach()
