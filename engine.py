import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate


class Loss_sum_v1_risk_align_weights(nn.Module):
    def __init__(self, w_cons=0.1, w_risk_align=0.02, T=2.0):
        super().__init__()
        self.register_buffer("ce_w", torch.tensor([6.0, 1.0], dtype=torch.float32))
        self.w_cons = float(w_cons)
        self.w_risk_align = float(w_risk_align)
        self.T = float(T)

    def forward(self, logits_factual, labels, logits_base, logits_risk=None):
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        labels = labels.long()

        weight = self.ce_w.to(logits_factual.device)
        loss_sup = F.cross_entropy(logits_factual, labels, weight=weight)

        p = F.log_softmax(logits_factual, dim=1)
        q = F.softmax(logits_base, dim=1)
        kl_raw = F.kl_div(p, q, reduction="none").sum(dim=1)
        loss_cons = (kl_raw * weight[labels]).mean()

        loss = loss_sup + self.w_cons * loss_cons
        loss_risk_align = logits_factual.new_tensor(0.0)

        if logits_risk is not None:
            pos_mask = labels == 1
            if pos_mask.any():
                teacher = F.softmax((logits_factual[pos_mask] / self.T).detach(), dim=1)
                student_logp = F.log_softmax(logits_risk[pos_mask] / self.T, dim=1)
                loss_risk_align = F.kl_div(student_logp, teacher, reduction="batchmean") * (self.T * self.T)
                loss = loss + self.w_risk_align * loss_risk_align

        return loss, loss_sup, loss_cons, loss_risk_align


def _forward_dual_batch(net, tritan_feats, conch_patches, clinical):
    clinical_iter = list(clinical) if clinical is not None else [None] * len(tritan_feats)
    logits_f_list = []
    logits_b_list = []
    logits_r_list = []
    gates = []

    for t_i, c_i, cli_i in zip(tritan_feats, conch_patches, clinical_iter):
        logits_f, logits_b, gate, _attn, _patient_repr, _evidence_repr, logits_r = net(t_i, c_i, cli_i)
        logits_f_list.append(logits_f.unsqueeze(0) if logits_f.dim() == 0 else logits_f)
        logits_b_list.append(logits_b.unsqueeze(0) if logits_b.dim() == 0 else logits_b)
        logits_r_list.append(logits_r.unsqueeze(0) if logits_r.dim() == 0 else logits_r)
        gates.append(gate.detach().cpu())

    return torch.stack(logits_f_list, dim=0), torch.stack(logits_b_list, dim=0), torch.stack(logits_r_list, dim=0), gates


def train_one_epoch(net, criterion, data_loader, optimizer, device, epoch, conf, log_writer=None):
    net.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for data_it, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
        adjust_learning_rate(optimizer, epoch + data_it / len(data_loader), conf)
        optimizer.zero_grad()

        labels = data["label"].to(device)
        tritan_feats = [x.to(device, dtype=torch.float32) for x in data["tritan_feats"]]
        conch_patches = [x.to(device, dtype=torch.float32) for x in data["conch_patches"]]
        clinical = data.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device, dtype=torch.float32)

        logits_f, logits_b, logits_r, _gates = _forward_dual_batch(net, tritan_feats, conch_patches, clinical)
        loss, loss_sup, loss_cons, loss_risk = criterion(logits_f, labels, logits_b, logits_r)
        loss.backward()
        optimizer.step()

        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"],
            loss=loss.item(),
            loss_sup=loss_sup.item(),
            loss_cons=loss_cons.item(),
            loss_risk=loss_risk.item(),
        )

        if log_writer is not None:
            log_writer.log("loss", loss)


@torch.no_grad()
def dual_view_evaluate(net, criterion, data_loader, device, conf, header):
    net.eval()
    metric_logger = MetricLogger(delimiter="  ")
    y_pred = []
    y_true = []
    gates = []

    for data in metric_logger.log_every(data_loader, 100, header):
        labels = data["label"].to(device)
        tritan_feats = [x.to(device, dtype=torch.float32) for x in data["tritan_feats"]]
        conch_patches = [x.to(device, dtype=torch.float32) for x in data["conch_patches"]]
        clinical = data.get("clinical")
        if clinical is not None:
            clinical = clinical.to(device, dtype=torch.float32)

        logits_f, logits_b, logits_r, batch_gates = _forward_dual_batch(net, tritan_feats, conch_patches, clinical)
        loss, _loss_sup, _loss_cons, _loss_risk = criterion(logits_f, labels, logits_b, logits_r)
        preds_use = logits_f

        if conf.n_class == 2:
            prob_pos = F.softmax(preds_use, dim=1)[:, 1]
            pred_label = preds_use.argmax(dim=1)
        else:
            prob_pos = torch.sigmoid(preds_use).squeeze(-1)
            pred_label = (prob_pos >= 0.5).long()

        acc1 = (pred_label == labels.long()).float().mean() * 100.0
        metric_logger.update(loss=loss.item(), acc1=acc1.item())
        y_pred.append(prob_pos.detach())
        y_true.append(labels.detach().long())
        gates.extend(batch_gates)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    auroc = torchmetrics.AUROC(task="binary").to(device)(y_pred.to(device), y_true.to(device)).item()
    f1_score = torchmetrics.F1Score(task="binary").to(device)((y_pred >= 0.5).long().to(device), y_true.to(device)).item()

    g_stats = {}
    if gates:
        g_all = torch.cat([g.reshape(-1) for g in gates])
        g_stats = {
            "mean": g_all.mean().item(),
            "median": g_all.median().item(),
            "p90": g_all.quantile(0.90).item(),
        }

    print("* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}".format(
        top1=metric_logger.acc1,
        losses=metric_logger.loss,
        AUROC=auroc,
        F1=f1_score,
    ))
    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, g_stats
