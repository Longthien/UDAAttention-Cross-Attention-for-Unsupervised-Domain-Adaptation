import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05, max_samples=1024):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, feat1, feat2):
        """
        feat1: [B, C, H, W] - Features from view 1
        feat2: [B, C, H, W] - Features from view 2
        """
        B, C, H, W = feat1.shape
        
        # L2 Normalize along channel dimension
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # Flatten spatial dimensions
        feat1 = feat1.view(B, C, -1)
        feat2 = feat2.view(B, C, -1)
        
        N = H * W
        K = min(self.max_samples, N)
        
        # Subsample spatial locations to save memory
        if K < N:
            idx = torch.randperm(N, device=feat1.device)[:K]
            feat1 = feat1[:, :, idx]
            feat2 = feat2[:, :, idx]
        
        # [B*K, C]
        feat1 = feat1.transpose(1, 2).reshape(B * K, C)
        feat2 = feat2.transpose(1, 2).reshape(B * K, C)
        
        # Compute similarity map [B*K, B*K]
        sim = torch.matmul(feat1, feat2.T) / self.temperature
        
        # Labels for cross entropy (diagonal elements are positives)
        labels = torch.arange(B * K, device=sim.device)
        
        loss = F.cross_entropy(sim, labels)
        return loss

class SpatialKLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_log, target):
        """
        pred_log: log_softmax predictions [B, Num_Classes, H, W]
        target: softmax probabilities [B, Num_Classes, H, W]
        """
        return F.kl_div(pred_log, target, reduction='batchmean')

class CentroidAwareInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, f_aug, f_t, source_gt, target_pseudo):
        """
        f_aug: [B, C, H, W] - Source features (augmented via cross attention)
        f_t: [B, C, H, W] - Target features
        source_gt: [B, H_gt, W_gt] or [B, 1, H_gt, W_gt] - Source ground truth
        target_pseudo: [B, H_gt, W_gt] or [B, 1, H_gt, W_gt] - Target pseudo labels
        """
        B, C, H, W = f_aug.shape
        
        # Squeeze channel dim if present
        if source_gt.dim() == 4:
            source_gt = source_gt.squeeze(1)
        if target_pseudo.dim() == 4:
            target_pseudo = target_pseudo.squeeze(1)
            
        # Downsample labels to match feature spatial dimensions
        # Use nearest neighbor to preserve class indices
        source_gt_down = F.interpolate(source_gt.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
        target_pseudo_down = F.interpolate(target_pseudo.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
        
        # Normalize features
        f_aug = F.normalize(f_aug, dim=1) # [B, C, H, W]
        f_t = F.normalize(f_t, dim=1) # [B, C, H, W]
        
        # Flatten spatial dimensions
        f_aug_flat = f_aug.view(B, C, -1).transpose(1, 2).reshape(-1, C) # [B*H*W, C]
        f_t_flat = f_t.view(B, C, -1).transpose(1, 2).reshape(-1, C) # [B*H*W, C]
        
        source_gt_flat = source_gt_down.view(-1) # [B*H*W]
        target_pseudo_flat = target_pseudo_down.view(-1) # [B*H*W]
        
        # Find unique classes in the current batch (ignoring ignore_index)
        unique_classes = torch.unique(target_pseudo_flat)
        unique_classes = unique_classes[unique_classes != self.ignore_index]
        
        if len(unique_classes) == 0:
            return torch.tensor(0.0, device=f_aug.device, requires_grad=True)
            
        # Compute all valid centroids from target
        centroids = []
        centroid_classes = []
        for cls in unique_classes:
            t_mask = (target_pseudo_flat == cls)
            if t_mask.any():
                centroid = f_t_flat[t_mask].mean(dim=0)
                centroid = F.normalize(centroid, dim=0) # [C]
                centroids.append(centroid)
                centroid_classes.append(cls)
                
        if len(centroids) == 0:
            return torch.tensor(0.0, device=f_aug.device, requires_grad=True)
            
        centroids = torch.stack(centroids) # [Num_Classes, C]
        centroid_classes = torch.tensor(centroid_classes, device=f_aug.device)
        
        # We only want to compute loss for source pixels whose class has a target centroid!
        s_class_has_centroid = (source_gt_flat.unsqueeze(1) == centroid_classes.unsqueeze(0)).any(dim=1)
        valid_s_mask = (source_gt_flat != self.ignore_index) & s_class_has_centroid
        
        valid_source_gt = source_gt_flat[valid_s_mask] # [N_s]
        valid_src_feats = f_aug_flat[valid_s_mask] # [N_s, C]
        
        if valid_src_feats.shape[0] == 0:
            return torch.tensor(0.0, device=f_aug.device, requires_grad=True)
            
        # Subsample source pixels to save memory
        MAX_SAMPLES = 4096
        if valid_src_feats.shape[0] > MAX_SAMPLES:
            idx = torch.randperm(valid_src_feats.shape[0], device=f_aug.device)[:MAX_SAMPLES]
            valid_src_feats = valid_src_feats[idx]
            valid_source_gt = valid_source_gt[idx]
            
        # Compute similarities [N_s, Num_Classes]
        sim = torch.matmul(valid_src_feats, centroids.T) / self.temperature
        
        # Labels for cross entropy (index of the matching class in centroid_classes)
        labels = (valid_source_gt.unsqueeze(1) == centroid_classes.unsqueeze(0)).long().argmax(dim=1)
        
        loss = F.cross_entropy(sim, labels)
        return self.loss_weight * loss
