import os
import torchaudio
import numpy as np
import torch
import pytorch_lightning as pl
from utils.adabound import AdaBound
from utils.audio import Audio
from utils.loss import SISNRLoss

from torchmetrics import \
    SignalDistortionRatio as SDR, \
    ScaleInvariantSignalNoiseRatio as SI_SNR, \
    WordErrorRate


class VoiceSeparationPL(pl.LightningModule):
    def __init__(self, hp, model, embedder) -> None:
        super().__init__()

        self.hp = hp
        self.model = model
        self.embedder = embedder
        self.optimizer = hp.train.optimizer
        self.criterion = SISNRLoss()

        self.sdr = SDR()
        self.si_snr = SI_SNR()
        self.wer = WordErrorRate()
        self.audio = Audio(hp)

        # freeze params of the embeddings
        for param in self.embedder.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        """ The complete training loop """
        refer_wav_list, target_wav, mixed_wav = batch

        dvec_list = list()
        for wav in refer_wav_list:
            mel = self.audio.mel_spectrogram(wav)
            dvec = self.embedder(mel)
            dvec_list.append(dvec)
        dvec = torch.stack(dvec_list, dim=0)

        # target_mag, _ = self.audio.wav2spec(target_wav)
        mixed_mag, mixed_phase = self.audio.wav2spec(mixed_wav)
        pred_mask = self.model(mixed_mag, dvec)
        output = mixed_mag * pred_mask

        # SI_SNR
        output_wav = self.audio.spec2wav( output, mixed_phase)   # reconstruct waveform
        # append channel dim
        target = target_wav.unsqueeze(1)
        # append channel dim
        output = output_wav.unsqueeze(1)
        # length of wav sequence
        seqlen = torch.tensor(mixed_wav.shape).cuda()

        loss = self.criterion(output, target, seqlen)
        self.log('train/loss', loss, on_epoch=True,
                 batch_size=self.hp.train.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """ The complete validation loop """
        refer_wav_list, target_wav, mixed_wav = batch

        dvec_list = list()
        for wav in refer_wav_list:
            mel = self.audio.mel_spectrogram(wav)
            dvec = self.embedder(mel)
            dvec_list.append(dvec)
        dvec = torch.stack(dvec_list, dim=0)

        mixed_mag, mixed_phase = self.audio.wav2spec(mixed_wav)
        est_mask = self.model(mixed_mag, dvec)
        est_mag = est_mask * mixed_mag
        est_wav = self.audio.spec2wav(est_mag, mixed_phase)

        sdr_score = self.sdr(est_wav, target_wav).item()
        si_snr_score = self.si_snr(est_wav, target_wav).item()

        self.log('val/sdr', sdr_score, on_epoch=True,
                 batch_size=self.hp.train.batch_size)
        self.log('val/si_snr', si_snr_score, on_epoch=True,
                 batch_size=self.hp.train.batch_size)

    def test_step(self, batch, batch_idx):
        """ The complete test loop """

        refer_wav, target_wav, mixed_wav = batch[0]

        dvec_mel = self.audio.mel_spectrogram(refer_wav)
        dvec_emb = self.embedder(dvec_mel.squeeze())
        dvec_emb = dvec_emb.unsqueeze(0)

        mixed_mag, mixed_phase = self.audio.wav2spec(mixed_wav)
        pred_mask = self.model(mixed_mag, dvec_emb)
        pred_mag = pred_mask * mixed_mag
        pred_wav = self.audio.spec2wav(pred_mag, mixed_phase)

        return {
            'sdr': self.sdr(pred_wav, target_wav).item(),
            'si_snr': self.si_snr(pred_wav, target_wav).item()
        }

    def test_epoch_end(self, outputs):
        self.log('test/sdr', np.mean([o['sdr'] for o in outputs]))
        self.log('test/si_snr', np.mean([o['si_snr'] for o in outputs]))



    def forward(self, mixed_wav, target_wav, dvec_wav, mask="reg"):
        """ predict the waveform """

        # get mel spectrogram of ref
        dvec_mel = self.audio.mel_spectrogram(dvec_wav)
        dvec = self.embedder(dvec_mel.cpu().squeeze(0))
        dvec = dvec.unsqueeze(0)

        mixed_mag, mixed_phase = self.audio.wav2spec(mixed_wav)

        # predict the mask and the waveform
        mixed_mag, mixed_phase = mixed_mag.cpu(), mixed_phase.cpu()
        pred_mask = self.model(mixed_mag, dvec)

        if mask == "reg":
            pred_mag = mixed_mag * pred_mask
        elif mask == "inv":
            pred_mag = mixed_mag * \
                (torch.ones_like(pred_mask) - pred_mask)

        # pred_mag = mixed_mag * pred_mask
        pred_wav = self.audio.spec2wav(pred_mag, mixed_phase)
        pred_wav = pred_wav.detach().cpu()

        return pred_wav

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.hp.train.adam)

        elif self.optimizer == 'adabound':
            return AdaBound(self.model.parameters(),
                            lr=self.hp.train.adabound.initial,
                            final_lr=self.hp.train.adabound.final)

        else:
            raise Exception("%s optimizer not supported" % self.optimizer)