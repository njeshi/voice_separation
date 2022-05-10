import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.sox_effects import apply_effects_tensor as sox_fx
import speech_recognition as sr

class Audio():
    def __init__(self, hp):
        self.hp = hp
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def load(self, audio_file_path):
        wav, _ = torchaudio.load(audio_file_path)
        target_sr = self.hp.audio.sample_rate

        # Normalize volume, trim leading and trailing silence and resample
        effects = [
            ["gain", "-n"],
            ['silence', '-l', '1', '0.1', '-50d'], ['reverse'],
            ['silence', '-l', '1', '0.1', '-50d'], ['reverse'],
            ['rate', str(target_sr)]
        ]

        wav, _ = sox_fx(wav, target_sr, effects, channels_first = True)

        return wav

    def amp_to_db(self, x):
        return F.amplitude_to_DB(x, multiplier=20,
                                 db_multiplier=0,
                                 amin=1e-5)

    def db_to_amp(self, x):
        return F.DB_to_amplitude(x, ref=1, power=0.5)

    def normalize(self, spec):
        return torch.clip(spec / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, spec):
        return (torch.clip(spec, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db

    def stft(self, y):
        """ Short time Fourier Transform """
        window = torch.hann_window(self.hp.audio.win_length).to(self.device)
        y = y.to(self.device)
        return torch.stft(y, n_fft=self.hp.audio.n_fft,
                          hop_length=self.hp.audio.hop_length,
                          win_length=self.hp.audio.win_length,
                          window = window,
                          pad_mode='constant',
                          normalized=False,
                          return_complex=True)

    def istft(self, y):
        """ Inverse short time Fourier Transform """
        window = torch.hann_window(self.hp.audio.win_length).to(self.device)
        y = y.to(self.device)
        return torch.istft(y, n_fft=self.hp.audio.n_fft,
                           hop_length=self.hp.audio.hop_length,
                           win_length=self.hp.audio.win_length,
                           window = window)
    
    def mel_spectrogram(self, waveform):
        """ Create mel spectrogram for a raw audio signal. """
        mel_spectrogram = T.MelSpectrogram(sample_rate=self.hp.audio.sample_rate,
                                           n_fft=self.hp.embedder.n_fft,
                                           hop_length=self.hp.audio.hop_length,
                                           win_length=self.hp.audio.win_length,
                                           n_mels=self.hp.embedder.num_mels,
                                           pad_mode='constant',
                                           mel_scale='slaney',
                                           norm='slaney')

        # TODO: window should be on cpu
        melspec = mel_spectrogram(waveform.cpu()) 
        melspec = torch.log10(melspec + 1e-6)
        return melspec.cuda()

    def spectrogram(self, waveform):
        """ Create a spectrogram from a audio signal."""

        transform = T.Spectrogram(n_fft=self.hp.audio.n_fft,
                                    hop_length=self.hp.audio.hop_length,
                                    win_length=self.hp.audio.win_length,
                                    pad_mode='constant',
                                    normalized=False,
                                    power = None)
        spec = self.stft(waveform)
        # spec = transform(waveform)  # stft
        mag, phase = torch.abs(spec), torch.angle(spec)
        return mag, phase

    def inverse_spectrogram(self, spectrogram, phase):
        """ Inverse spectrogram to recover an audio signal from a spectrogram."""
        stft_matrix = spectrogram * torch.exp(phase * 1j)
        transform = T.InverseSpectrogram(n_fft=self.hp.audio.n_fft,
                                        hop_length=self.hp.audio.hop_length,
                                        win_length=self.hp.audio.win_length,
                                        window_fn = torch.hann_window,
                                        center=True)
        waveform = self.istft(stft_matrix)
        # waveform = transform(stft_matrix)
        return waveform

    def wav2spec(self, waveform):
        """ convert to spectrogram and normalize """
        mag, phase = self.spectrogram(waveform)

        mag = self.amp_to_db(mag) - self.hp.audio.ref_level_db
        mag = self.normalize(mag)
        mag = mag.transpose(2, 1)
        phase = phase.transpose(2, 1)

        return mag, phase

    def spec2wav(self, spec, phase=None):
        """ denormalize and convert to waveform """
        spec = spec.transpose(2, 1)
        phase = phase.transpose(2, 1)

        spec = self.denormalize(spec) + self.hp.audio.ref_level_db
        spec = self.db_to_amp(spec)
        waveform = self.inverse_spectrogram(spec, phase)

        return waveform


    def recognize(self, file_path):
        """ Speech to text """
        r = sr.Recognizer()

        with sr.AudioFile(file_path) as source:
            audio = r.record(source)
        try:
            pred_text = r.recognize_google(audio)
            return pred_text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return ""