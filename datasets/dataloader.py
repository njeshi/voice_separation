import os
from tqdm.auto import tqdm
import glob
import random
import torch
import torchaudio
from utils.audio import Audio


class LibriSpeech(torchaudio.datasets.LIBRISPEECH):

    def __init__(self, hp, root, train, download=True) -> None:
        """
        Args:
            hp: hyperparameters
            root: root dir to save the data
            train: either True or False
            download: if True, download LibriSpeech
        """
        self.url = hp.data.train_url if train else hp.data.test_url
        super().__init__(root=root, url=self.url, download=download)

        self.hp = hp
        self.train = train
        self.audio = Audio(hp)

        self.target_sr = hp.audio.sample_rate
        self.target_len = int(hp.audio.sample_rate * hp.data.audio_len)
        self.dvec_len = hp.embedder.window * hp.audio.hop_length * 1.1

        self.mixed_out_dir = self._generate_mixed_dataset()
        self._load_files_list()

        # loop over files to count total length
        self.hours = 0
        for filename in self.mixed_wav_list:
            meta = torchaudio.info(filename)
            self.hours += (meta.num_frames/meta.sample_rate) / 3600

        print(
            f"Located {len(self.mixed_wav_list)} examples totaling {self.hours:0.1f} hr in the {'train' if train else 'test'} subset.")

    def _generate_mixed_dataset(self):
        # create directory
        mode = 'train' if self.train else 'test'
        out_dir = os.path.dirname(self._path)
        mixed_out_dir = os.path.join(out_dir, mode)
        if os.path.exists(mixed_out_dir):
            if os.listdir(mixed_out_dir):
                return mixed_out_dir

        print('Generating mixed dataset...')
        os.makedirs(mixed_out_dir, exist_ok=True)

        # get all speakers folders
        dir_ids = os.path.join(self._path, '*')
        folders = [x for x in glob.glob(dir_ids) if os.path.isdir(x)]
        speakers = [glob.glob(os.path.join(spk, '**', self.hp.form.input), recursive=True)
                    for spk in folders]
        speakers = [x for x in speakers if len(x) >= 2]

        # choose randomly speeches from two speakers
        def mix_wrapper(idx):
            speaker1, speaker2 = random.sample(speakers, 2)
            reference, clean = random.sample(speaker1, 2)
            interference = random.choice(speaker2)
            return self._mix(idx, reference, clean, interference,
                             out_dir=mixed_out_dir)

        arr_size = self.hp.data.train_size if self.train else self.hp.data.test_size
        progress_bar = tqdm(total=arr_size)

        idx = 0
        while idx < arr_size:
            out = mix_wrapper(idx)
            if out == -1:
                continue
            idx += 1
            progress_bar.update(1)
        progress_bar.close()

        return mixed_out_dir

    def _mix(self, idx, reference, clean, interference, out_dir):
        """
        Args:
        ----
            num: (int) index number; needed for naming the files
            reference: (path) reference audio for speaker 1; d-vector
            clean: (path) target audio for speaker 1; will be mixed
            interference: (path) audio from speaker 2; interference audio 
            out_dir: (path) path to save training and testing files
        """

        # Load audio files
        d_vector = self.audio.load(reference)
        waveform1 = self.audio.load(clean)
        waveform2 = self.audio.load(interference)

        # if d_vector or audio length is too short, discard it
        if d_vector.shape[-1] < self.dvec_len:
            return -1
        if waveform1.shape[-1] < self.target_len:
            return -1
        if waveform2.shape[-1] < self.target_len:
            return -1

        # make audios the same length
        waveform1 = waveform1.narrow(-1, 0, self.target_len)
        waveform2 = waveform2.narrow(-1, 0, self.target_len)

        # mixed audio
        mixed_wav = waveform1 + waveform2

        # normalize
        norm_factor = torch.max(mixed_wav.abs()) * 1.1
        waveform1 /= norm_factor
        waveform2 /= norm_factor
        mixed_wav /= norm_factor

        # format path
        def formatter(form):
            return os.path.join(out_dir, form.replace('*', '%06d' % idx))

        refer_wav_path = formatter(self.hp.form.refer)
        target_wav_path = formatter(self.hp.form.target)
        mixed_wav_path = formatter(self.hp.form.mixed)

        # try waveform2 when noise becomes the target
        target_wav = waveform1

        # save audio files
        torchaudio.save(refer_wav_path, d_vector, self.target_sr)
        torchaudio.save(target_wav_path, target_wav, self.target_sr)
        torchaudio.save(mixed_wav_path, mixed_wav, self.target_sr)

    def __len__(self):
        return len(self.refer_wav_list)

    def __getitem__(self, idx):
        refer_wav, _ = torchaudio.load(self.refer_wav_list[idx])
        target_wav, _ = torchaudio.load(self.target_wav_list[idx])
        mixed_wav, _ = torchaudio.load(self.mixed_wav_list[idx])

        return refer_wav, target_wav, mixed_wav

    def _load_files_list(self):
        def find_all(file_format):
            return sorted(
                glob.glob(os.path.join(self.mixed_out_dir, file_format)))

        self.refer_wav_list = find_all(self.hp.form.refer)
        self.mixed_wav_list = find_all(self.hp.form.mixed)
        self.target_wav_list = find_all(self.hp.form.target)    

        assert len(self.refer_wav_list) == \
            len(self.target_wav_list) == \
            len(self.mixed_wav_list), "number of training files must match"

    def train_collate_fn(self, batch):
        refer_wav_list = list()
        target_wav_list = list()
        mixed_wav_list = list()

        for refer_wav, target_wav, mixed_wav in batch:
            refer_wav_list.append(refer_wav.squeeze(0))
            target_wav_list.append(target_wav.squeeze(0))
            mixed_wav_list.append(mixed_wav.squeeze(0))

        # refer_wav_list = torch.stack(refer_wav_list, dim=0)
        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_wav_list = torch.stack(mixed_wav_list, dim=0)

        return refer_wav_list, target_wav_list, mixed_wav_list

    def test_collate_fn(self, batch):
        return batch
