import torch

class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = torch.nn.functional.scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::50, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = torch.mean(sim, -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
        topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=topk_ind)
        k0 = torch.gather(k, dim=-2, index=topk_ind)
        return k0, v0


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::30, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = sim.softmax(-1)
        sim = torch.mean(sim, 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = torch.unique(activated_token, return_counts=True)[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        index = index.expand(-1, v.shape[1], -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=index)
        k0 = torch.gather(k, dim=-2, index=index)
        return k0, v0 