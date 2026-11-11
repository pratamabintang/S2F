import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PositionEmbeddingSine
from .transformer import Transformer


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat


class ProtoFormer(nn.Module):
    def __init__(self, shot=1, reduce_dim=64):
        super(ProtoFormer, self).__init__()
        self.shot = shot
        self.reduce_dim = reduce_dim
        # query & support sama-sama sudah difuse rgb+topo (level 1,2) lewat fuse -> 64+64
        in_fea_dim = 64 + 64
        drop_out = 0.5

        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        ) 
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )
        self.query_embed=nn.Embedding(reduce_dim,1) 
        self.pe_layer=PositionEmbeddingSine(reduce_dim//2, normalize=True)
        self.transformer = Transformer(
            d_model=reduce_dim,
            dropout=0.1,
            nhead=4,
            dim_feedforward=reduce_dim//4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        self.high_avg_pool = nn.Identity()        
        prior_channel = 1 
        self.pyramid_bins = [60, 30, 15, 8]
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )
        self.init_merge = []
        self.beta_conv = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2+prior_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True)
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True)
            ))            

        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False)                         
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )
        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(2*reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)    

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, q_feats, s_feats, s_y):
        q2, q3, q4 = q_feats[1], q_feats[2], q_feats[3]
        s2, s3, s4 = s_feats[1], s_feats[2], s_feats[3]
        batch_size = q3.size(0)
        fts_size = q3.shape[-2:]
        q2_dn = F.interpolate(q2, size=fts_size, mode='bilinear', align_corners=True)
        s2_dn = F.interpolate(s2, size=fts_size, mode='bilinear', align_corners=True)
        query_feat = torch.cat([q2_dn, q3], dim=1)
        supp_feat = torch.cat([s2_dn, s3], dim=1)
        query_feat = self.adjust_feature_qry(query_feat)
        supp_feat = self.adjust_feature_supp(supp_feat) 
        supp_feat_list = []
        r_supp_feat = supp_feat.view(batch_size, self.shot, -1, fts_size[0], fts_size[1])
        for st in range(self.shot):
            mask = (s_y[:,st,:,:]==1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            tmp_supp_feat = r_supp_feat[:,st,...]
            tmp_supp_feat = Weighted_GAP(tmp_supp_feat, mask)
            supp_feat_list.append(tmp_supp_feat)
        global_supp_pp = supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_list)):
                global_supp_pp += supp_feat_list[i]
            global_supp_pp /= len(supp_feat_list)
                        
        query_embed = global_supp_pp.squeeze(-1)
        query_pos = self.query_embed.weight
        key_pos=self.pe_layer(query_feat)
        key_embed=query_feat
        masking=None
        fg_embed=self.transformer(key_embed,masking,query_embed,query_pos,key_pos)
        supp_feat_high = s4.view(batch_size, self.shot, s4.size(1), s4.size(2), s4.size(3))
        corr_query_mask = self.generate_prior(q4, supp_feat_high, s_y, fts_size)
        
        pyramid_feat_list = []
        out_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = global_supp_pp.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = merge_feat_bin
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat)   
        # fg_embed: (B, C, L=1) -> pixel-wise similarity with the prototype
        sim = torch.einsum("bchw,bcl->bhw", query_feat, fg_embed).unsqueeze(1)
        out = query_feat * torch.sigmoid(sim)
        return out

    def generate_prior(self, query_feat_high, supp_feat_high, s_y, fts_size):
        bsize, _, sp_sz, _ = query_feat_high.size()[:]
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for st in range(self.shot):
            tmp_mask = (s_y[:,st,:,:] == 1).float().unsqueeze(1)
            tmp_mask = F.interpolate(tmp_mask, size=(sp_sz, sp_sz), mode='bilinear', align_corners=True)

            tmp_supp_feat = supp_feat_high[:,st,...] * tmp_mask               
            q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]

            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous() 
            tmp_supp = tmp_supp.contiguous() 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(fts_size[0], fts_size[1]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        return corr_query_mask
