import torch
import torch.nn as nn
from itertools import permutations


class SISNRLoss(nn.Module):
    """
    Scale Invariant Signal to Noise Ratio with Permutation Invariant
    Adapted from:
        https://github.com/kaituoxu/Conv-TasNet/blob/master/src/pit_criterion.py
    """
    
    def __init__(self):
        super(SISNRLoss, self).__init__()
        self.epsilon = 1e-16 # use epsilon for prevent  gradient explosion
    
    def get_mask(self, source, source_lengths):
        """
        Args:
            source: [B, C, T]
            source_lengths: [B]
        Returns:
            mask: [B, 1, T]
        """
        B, _, T = source.size()
        mask = source.new_ones((B, 1, T))
        for i in range(B):
            mask[i, :, source_lengths[i]:] = 0
        return mask

    def forward(self, estimate_source, source, source_lengths):
        """
        Args:
            source: [B, C, T], B is batch size
            estimate_source: [B, C, T]
            source_lengths: [B], each item is between [0, T]
        """
        assert source.size() == estimate_source.size()
        B, C, T = source.size()
        # mask padding position along T
        mask = self.get_mask(source, source_lengths)
        estimate_source *= mask

        # Step 1. Zero-mean norm
        num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
        mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
        mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
        zero_mean_target = source - mean_target
        zero_mean_estimate = estimate_source - mean_estimate
        # mask padding position along T
        zero_mean_target *= mask
        zero_mean_estimate *= mask

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=1)      # [B, 1, C, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
        
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.epsilon  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.epsilon)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.epsilon)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
        
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)

        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= C

        loss = 20 - torch.mean(max_snr) # 20 is an arbitrary high value
        return loss



class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.
    Note that this returns the negative of the SI-SDR loss.
    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)
    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = losses.mean()
        return -losses



class PowerlawCompressedLoss(nn.Module):
    def __init__(self, power=0.3, complex_loss_ratio=0.113):
        super(PowerlawCompressedLoss, self).__init__()
        self.power = power
        self.complex_loss_ratio = complex_loss_ratio
        self.criterion = nn.MSELoss()
        self.epsilon = 1e-16 # use epsilon for prevent  gradient explosion

    def forward(self, prediction, target, seq_len=None, spec_phase=None):
        # prevent NAN loss
        prediction = prediction + self.epsilon
        target = target + self.epsilon

        prediction = torch.pow(prediction, self.power)
        target = torch.pow(target, self.power)

        spec_loss = self.criterion(torch.abs(target), torch.abs(prediction))
        complex_loss = self.criterion(target, prediction)

        loss = spec_loss + (complex_loss * self.complex_loss_ratio)
        return loss