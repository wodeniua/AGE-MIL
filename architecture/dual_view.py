import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MaxMIL(nn.Module):
    def __init__(self, in_dim=1024, repr_dim=512, dropout=True, act="relu"):
        super().__init__()
        layers = [nn.Linear(in_dim, repr_dim)]
        if act.lower() == "relu":
            layers.append(nn.ReLU())
        elif act.lower() == "gelu":
            layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.encoder = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x):
        h = self.encoder(x)
        patient_repr, _ = h.max(dim=0)
        return patient_repr


class TopKPatchSelectorMixCos(nn.Module):
    def __init__(self, d_patch, d_patient, k=512, d_hidden=256, tau=0.2, use_ln=True, stopgrad_patient=True):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)
        self.stopgrad_patient = bool(stopgrad_patient)
        self.ln_p = nn.LayerNorm(d_patient) if use_ln else nn.Identity()
        self.ln_x = nn.LayerNorm(d_patch) if use_ln else nn.Identity()
        self.scorer_u = nn.Linear(d_patch, 1, bias=False)
        self.wp = nn.Linear(d_patient, d_hidden, bias=False)
        self.wx = nn.Linear(d_patch, d_hidden, bias=False)
        self.alpha = nn.Sequential(nn.Linear(d_patient, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))

    def forward(self, patches, patient_repr):
        if self.stopgrad_patient:
            patient_repr = patient_repr.detach()

        x = self.ln_x(patches)
        p = self.ln_p(patient_repr)
        score_uncond = self.scorer_u(x).squeeze(-1)
        x_proj = F.normalize(self.wx(x), dim=-1)
        p_proj = F.normalize(self.wp(p), dim=-1)
        score_cond = (x_proj * p_proj.unsqueeze(0)).sum(dim=-1) / self.tau
        alpha = torch.sigmoid(self.alpha(p)).squeeze(-1)
        scores = (1.0 - alpha) * score_uncond + alpha * score_cond

        k = min(self.k, patches.size(0))
        idx = torch.topk(scores, k=k, largest=True, sorted=False).indices
        return patches.index_select(0, idx), idx


class PatientPatchCrossAttnMultiQ(nn.Module):
    def __init__(self, d_query, d_kv, d_attn=384, use_ln=True):
        super().__init__()
        self.d_attn = int(d_attn)
        self.ln_q = nn.LayerNorm(d_query) if use_ln else nn.Identity()
        self.ln_kv = nn.LayerNorm(d_kv) if use_ln else nn.Identity()
        self.wq = nn.Linear(d_query, d_attn, bias=False)
        self.wk = nn.Linear(d_kv, d_attn, bias=False)
        self.wv = nn.Linear(d_kv, d_attn, bias=False)

    def forward(self, q_tokens, top_patches):
        q = self.wq(self.ln_q(q_tokens))
        kv = self.ln_kv(top_patches)
        k = self.wk(kv)
        v = self.wv(kv)
        attn_logits = (q @ k.transpose(0, 1)) / (self.d_attn ** 0.5)
        attn = torch.softmax(attn_logits, dim=1)
        evidence_repr = attn @ v
        return evidence_repr, attn


class DualViewPatientClassifier(nn.Module):
    def __init__(self, conf, top_k=256):
        super().__init__()
        self.top_k = int(getattr(conf, "K_PATCH", top_k))
        self.maxmil = MaxMIL(in_dim=conf.D_feat, repr_dim=conf.D_inner)
        self.topk_selector = TopKPatchSelectorMixCos(
            d_patch=conf.D_feat,
            d_patient=conf.D_inner,
            k=self.top_k,
            d_hidden=256,
            tau=0.2,
            stopgrad_patient=True,
        )
        self.cross_attn = PatientPatchCrossAttnMultiQ(conf.D_inner, conf.D_feat, conf.D_inner)
        self.gate = nn.Linear(conf.D_inner, 1)
        self.fuse = nn.Linear(conf.D_inner + conf.D_inner, conf.D_inner)
        self.head = nn.Linear(conf.D_inner, conf.n_class)

    def forward(self, tritan_feats, conch_patches, clinical=None):
        patient_repr = self.maxmil(tritan_feats)
        logits_base = self.head(patient_repr)

        top_patches, _ = self.topk_selector(conch_patches, patient_repr)
        norm_tritan_feats = self.maxmil.encoder(tritan_feats)
        evidence_repr, attn = self.cross_attn(norm_tritan_feats, top_patches)
        gate = torch.sigmoid(self.gate(evidence_repr))
        evidence_patient = (gate * evidence_repr).max(dim=0).values
        fused = self.fuse(torch.cat([patient_repr, evidence_patient], dim=-1))
        logits_factual = self.head(fused)

        slice_logits = self.head(norm_tritan_feats)
        logits_risk = torch.logsumexp(slice_logits, dim=0) - math.log(max(slice_logits.size(0), 1))
        return logits_factual, logits_base, gate, attn, patient_repr, evidence_repr, logits_risk
