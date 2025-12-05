import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"  # hides TF-related warnings

from transformers import HubertModel
from transformers import (
    pipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WavLMModel,
    WavLMConfig
)

# import torchaudio
import torch
# import json
import argparse
# from tqdm import tqdm
import random
# import numpy as np
import os
import time
import datetime
# Use a pipeline as a high-level helper
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
# import IPython.display as ipd
import soundfile as sf
from torch import nn
from torch.nn import functional as F
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
# from deepspeech_pytorch.decoder import Decoder
# from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
# from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model
import torchaudio.transforms as Trans_audio
# from api.iflytek_ASR import iflytek_ASR
# from api.Tencent_ASR import tencent_ASR
# from api.Baidu_ASR import baidu_ASR
# from api.Aliyun_ASR import aliyun_ASR
from torch.cuda.amp import autocast
import whisper
import metrics
import pickle
#from WavLM.WavLM import WavLM, WavLMConfig

path = "datasets/librispeech_train_clean100_waveform_spec_speaker_text.pkl"
with open(path, "rb") as f:
    dataset = pickle.load(f)
dataset = dataset[:200]

path = "librispeech_test_clean_waveform_spec_speaker_text.pkl"
with open(path, "rb") as f:
    tgt_dataset = pickle.load(f)
tgt_dataset = tgt_dataset[:50]


class Attacker:
    def __init__(self, args):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        '''
        model_path = 'hugging_face/hubert'
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.hubert = HubertModel.from_pretrained(model_path).eval().to(self.device)
        '''
        # replaced by:
        hubert_model_id = "facebook/hubert-large-ls960-ft"  # or "facebook/hubert-base-ls960"
        self.hubert = HubertModel.from_pretrained(hubert_model_id).eval().to(self.device)
        
        self.target_layer = args.tgt_layer - 1
        self.tgt_model = args.tgt_model
        self.wav_input = args.wav_input
        self.attack_iters = args.attack_iters
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.segment_size = args.segment_size
        self.tgt_text = None
        self.ori_text = args.ori_text
        self.if_slm_loss = args.if_slm_loss
        '''
        model_path = "hugging_face/wav2vec2_base_960h"
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        '''
        # replaced by:
        wav2vec2_model_id = "facebook/wav2vec2-base-960h"
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_id)
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_id)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model_id)

        self.wav2vec2_pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.wav2vec2,
            tokenizer=self.processor,
            feature_extractor=self.feature_extractor,
            device=self.device
        )
        self.wav2vec2.to(self.device)
        '''
        model_path = "hugging_face/whisper_small"
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(model_path)
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        '''
        # replaced by:
        whisper_model_id = "openai/whisper-small"
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_id)
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_id)

        # WavLM-Large
        '''
        checkpoint = torch.load('WavLM/WavLM-Large.pt')
        cfg = WavLMConfig(checkpoint['cfg'])
        self.WavLM = WavLM(cfg)
        self.WavLM.load_state_dict(checkpoint['model'])
        self.WavLM.eval()
        self.WavLM.to(self.device)
        '''
        # replaced by:
        model_name = "microsoft/wavlm-large"  # HF hub ID
        self.wavlm_config = WavLMConfig.from_pretrained(model_name)
        self.WavLM = WavLMModel.from_pretrained(model_name)
        self.WavLM.eval()
        self.WavLM.to(self.device)

        self.whisper_pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.whisper,
            tokenizer=self.whisper_tokenizer,
            feature_extractor=self.whisper_feature_extractor,
            device=self.device
        )
        self.whisper.to(self.device)
        self.whisper_encoder = self.whisper.get_encoder()
        ########################### for DeepSpeech #################################
        self.labels = ['_', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
        self.labels_map = dict([(self.labels[i], i) for i in range(len(self.labels))])
        self.attack_criterion = nn.CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        tgt_model_path = 'pretrained/deepspeech/librispeech_pretrained_v3.ckpt'
        self.deepspeech = load_model(device=self.device, model_path=tgt_model_path)
        self.deep_speech_cfg = TranscribeConfig()
        self.decoder = load_decoder(labels=self.deepspeech.labels, cfg=self.deep_speech_cfg.lm)
        self.sample_rate = 16000
        self.window_size = .02
        self.window_stride = .01
        self.precision = 32
        self.wav2spec = Trans_audio.Spectrogram(n_fft=int(self.sample_rate * self.window_size),
                                                win_length=int(self.sample_rate * self.window_size),
                                                hop_length=int(self.sample_rate * self.window_stride)).to(self.device)

    def parse_transcript(self, transcript):
        """
            Parses a transcript into a numerical tensor.
            :param transcript: The original transcript string.
            :return: A tensor containing the index of each character in the transcript.
        """
        transcript = transcript.replace('\n', '')
        transcript_n = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        transcript_tensor = torch.tensor(transcript_n, dtype=torch.long)
        return transcript_tensor

    def test_wav(self, x, target_model):
        """
            Tests a given audio waveform x with different ASR models and prints the results.
            :param x: The input audio waveform tensor.
        """
        if 'wav2vec2' in target_model:
            txt = self.wav2vec2_pipe(x.detach().cpu().numpy())['text']
        if 'whisper' in target_model:
            txt = self.whisper_pipe(x.detach().cpu().numpy())['text']
        if 'aliyun' in target_model:
            audio = x.detach().cpu().numpy()
            path = "cache/tmp.wav"
            sf.write(path, audio, 22050)
            txt = aliyun_ASR(path)
        return txt

    def get_feat(self, wav, tgt_model):
        """
            Extracts audio features based on the target model.
            :param wav: The input audio waveform tensor.
            :param tgt_model: The name of the target model, which determines the feature extraction method.
            :return: The extracted audio features.
        """
        #if 'deepspeech' in tgt_model:
        #    out_spec = self.calc_spectrogram(wav)
        #    input_lengths = torch.LongTensor(out_spec.size(0) * [out_spec.size(2)])
        #    _, _, hs = self.spec2trans(out_spec, input_lengths)
        if 'hubert' in tgt_model:
            ouput = self.hubert(wav.to(self.hubert.device), output_hidden_states=True)
            hs = ouput.hidden_states
        elif 'whisper' in tgt_model:
            mel = whisper.log_mel_spectrogram(wav, n_mels=80, device=self.device)
            hs = self.whisper_encoder(mel, output_hidden_states=True).hidden_states
            '''
            elif 'wavlm' in tgt_model:
                # extract the representation of each layer
                if self.WavLM.cfg.normalize:
                    wav = torch.nn.functional.layer_norm(wav, wav.shape)
                rep, layer_results = \
                    self.WavLM.extract_features(wav, output_layer=self.WavLM.cfg.encoder_layers, ret_layer_results=True)[0]
                hs = [x.transpose(0, 1) for x, _ in layer_results]
            '''
        # replaced by: 
        elif 'wavlm' in tgt_model:
            # HF WavLMModel interface
            outputs = self.WavLM(
                wav.to(self.WavLM.device),
                output_hidden_states=True,
                return_dict=True,
                )
            hs = outputs.hidden_states  # tuple of [batch, seq_len, hidden_dim]

            
        else:
            ouput = self.wav2vec2(wav.to(self.wav2vec2.device), output_hidden_states=True)
            hs = ouput.hidden_states
        if self.target_layer == 'avg':
            wav_feat = torch.mean(torch.stack(hs), axis=0)
        else:
            wav_feat = hs[self.target_layer]
            if 'deepspeech' in tgt_model:
                wav_feat = wav_feat[1]
        return wav_feat

    def calc_spectrogram(self, y):
        """
            Calculates the spectrogram of an audio signal, followed by log and normalization.
            :param y: The input audio waveform tensor.
            :return: The normalized log-magnitude spectrogram.
        """
        spectrogram = self.wav2spec(y.to(self.device, torch.float32))
        tmp1 = spectrogram.clone()
        magnitudes = torch.abs(tmp1)

        tmp2 = magnitudes.clone()
        spec = torch.log1p(tmp2)
        spect = spec.clone()
        mean = spect.mean()
        std = spect.std()
        tmp3 = spect.clone()
        tmp3.add_(-mean)
        tmp4 = tmp3.clone()
        tmp4.div_(std)

        return tmp4

    def spec2trans(self, spec, input_sizes):
        """
            Converts a spectrogram to text using the DeepSpeech model.
            :param spec: The input spectrogram tensor.
            :param input_sizes: The input size of each sequence.
            :return: A tuple containing the DeepSpeech model's output, decoded text, and hidden states.
        """
        hs = None
        spec = spec.contiguous()
        spect = spec.view(spec.size(0), 1, spec.size(1), spec.size(2))
        spect = spect.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes, hs = self.deepspeech(spect, input_sizes, hs)
        decoded_output, _ = self.decoder.decode(out)
        txts = []
        for txt_l in decoded_output:
            txts.append(txt_l[0])
        return out, txts, hs

    def calc_ds_loss(self, syn_audio):
        """
            Calculates the DeepSpeech loss between the synthesized audio and the target text.
            :param syn_audio: The input synthesized audio tensor.
            :return: A tuple containing the calculated loss and the recognized text.
        """
        trans_tgt = self.parse_transcript(self.tgt_text.upper()).to(self.device)

        batch_size = syn_audio.shape[0]
        trans_tgt = trans_tgt.repeat(batch_size).view(batch_size, -1)
        x = syn_audio.to(self.device)
        out_spec = self.calc_spectrogram(x).to(self.device)
        input_lengths = torch.LongTensor(out_spec.size(0) * [out_spec.size(2)]).to(self.device)
        out_trans, txt, _ = self.spec2trans(out_spec, input_lengths)
        input_lengths = torch.tensor(out_trans.size(0) * [out_trans.size(1)]).to(self.device)
        target_lengths = torch.tensor(batch_size * [trans_tgt.size(1)]).to(self.device)

        out_trans = out_trans.transpose(0, 1)
        out_trans = out_trans.log_softmax(-1)

        attack_loss = self.attack_criterion(out_trans, trans_tgt, input_lengths, target_lengths)
        del out_spec
        del out_trans
        return attack_loss, txt

    def attack(self, test_models, idx):
        sample_rate = 16000
        wav = self.wav_input.to(self.device)
        wav = wav.unsqueeze(0)
        wav_tgt, _, _, self.tgt_text = random.choice(tgt_dataset)
        wav_tgt = wav_tgt.unsqueeze(0)

        if 'whisper' not in self.tgt_model:
            self.segment_size = max(wav.size(-1), wav_tgt.size(-1))
        if wav.size(-1) < self.segment_size:
            wav = torch.nn.functional.pad(wav, (0, self.segment_size - wav.size(-1)), 'constant')
        if wav_tgt.size(-1) < self.segment_size:
            wav_tgt = torch.nn.functional.pad(wav_tgt, (0, self.segment_size - wav_tgt.size(-1)), 'constant')

        wav_tgt = wav_tgt.to(self.device)
        logger.info(wav.shape)
        logger.info(wav_tgt.shape)
        ptb = torch.randn_like(wav).to(self.device) * 0.005
        ptb = torch.nn.Parameter(ptb.requires_grad_(True))
        optimizer = torch.optim.Adam([ptb], 0.001, (0.9, 0.999), weight_decay=1e-4)
        org_feat = self.get_feat(wav, self.tgt_model)
        tgt_feat = self.get_feat(wav_tgt, self.tgt_model)

        logger.info(f'tgt_shape = {tgt_feat.shape}, org_shape = {org_feat.shape}')
        tau = 8 / 255
        tot_loss = 0
        start_time = time.time()
        loss_dict = {"ASR_loss": None, "sim_loss": None}

        adv_wav = None
        for i in range(self.attack_iters):
            adv_wav = wav + ptb
            attack_loss, txts = self.calc_ds_loss(adv_wav)
            if self.if_slm_loss:
                wav_feat = self.get_feat(adv_wav, self.tgt_model)
                ts_tgt = tgt_feat.reshape(tgt_feat.shape[2], -1)
                ts_wav = wav_feat.reshape(wav_feat.shape[2], -1)
                cos_sim = torch.nn.functional.cosine_similarity(ts_tgt, ts_wav, dim=1)
                sim_loss = 1.0 - torch.mean(cos_sim)
            else:
                sim_loss = torch.tensor(0.0)

            suc_flag = (metrics.clean_text(txts[0]) == metrics.clean_text(self.tgt_text))
            if self.if_slm_loss:
                suc_flag = suc_flag and (sim_loss < 0.05)
            if suc_flag:
                logger.info('attack success!')
                out_log = f'deepspeech: {txts[0]}'
                for tgt_model in test_models:
                    txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
                    out_log += f'\n{tgt_model}: {txt}'
                logger.info(out_log)
                break

            loss = attack_loss
            if self.if_slm_loss:
                loss += 20 * sim_loss

            loss_dict["ASR_loss"] = attack_loss.item()
            loss_dict["sim_loss"] = sim_loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            ptb.grad = torch.sign(ptb.grad)
            optimizer.step()
            ptb.data = torch.clamp(ptb.data, -tau, tau)

            tot_loss += loss.item()

            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                now_loss = tot_loss / 10
                tot_loss = 0
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                out_log = f'Elapsed [{et}], loss [{i + 1}] = {now_loss:.7f}'
                for key, value in loss_dict.items():
                    out_log += f', {key} = {value:.7f}'
                logger.info(out_log)

            if (i + 1) % 100 == 0:
                """test"""
                out_log = f'deepspeech: {txts[0]}'
                for tgt_model in test_models:
                    txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
                    out_log += f'\n{tgt_model}: {txt}'
                logger.info(out_log)

        path = os.path.join(self.output_dir, f'{idx}_SLM_{i}_{self.ori_text}_{self.tgt_text}.wav')
        self.save_wav(adv_wav.squeeze(0).detach().cpu().numpy(), path)
        res = {}
        for tgt_model in test_models:
            res[tgt_model] = []
        for tgt_model in test_models:
            txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
            res[tgt_model].append((self.ori_text, txt))
        return res, path

    def save_wav(self, wav, path):
        output_file = path
        import soundfile as sf
        sf.write(output_file, wav, 16000)


import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial attack on ASR models.')
    parser.add_argument('--wav_input', type=str, help='Path to the original audio file (overridden by dataset).')
    parser.add_argument('--ori_text', type=str, help='Original transcript (overridden by dataset).')
    parser.add_argument('--tgt_layer', type=int, default=9, help='Target layer for feature extraction.')
    parser.add_argument('--tgt_model', type=str, default='hubert', choices=['hubert', 'wav2vec2', 'whisper', 'deepspeech', 'wavlm'], help='Target model for semantic feature matching.')
    parser.add_argument('--attack_iters', type=int, default=1500, help='Number of attack iterations.')
    parser.add_argument('--segment_size', type=int, default=480000, help='Segment size for audio processing.')
    parser.add_argument('--if_slm_loss', action='store_true', help='Enable semantic loss matching.')
    parser.add_argument('--output_dir', type=str, default='experiment', help='Base directory to save adversarial examples.')
    parser.add_argument('--test_models', nargs='+', default=['wav2vec2', 'whisper'], help='List of models to test the adversarial examples on.')
    parser.add_argument('--prefix_name', type=str, default='run_main_', help='Prefix for the log file name.')

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    output_dir = os.path.join(args.output_dir, args.tgt_model + '_' + str(args.tgt_layer))
    today_date = datetime.datetime.now().strftime('%m%d')
    log_path = args.prefix_name + output_dir.split('/')[-1] + f'_{today_date}.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    res = {}
    for tgt_model in args.test_models:
        res[tgt_model] = []
    wav_paths = []
    for i, (wav_input, spec, spk_id, txt) in enumerate(dataset):
        logger.info(
            f"\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            f"===============================================================================================================================\n\n")

        # Update args for the current iteration
        args.wav_input = wav_input
        args.ori_text = txt  # this is the actual transcript
        args.output_dir = output_dir # Use the constructed output_dir

        attacker = Attacker(args)
        ans_now, path = attacker.attack(args.test_models, i)
        wav_paths.append((path, txt))
        for tgt_model in args.test_models:
            res[tgt_model] += ans_now[tgt_model]

        torch.cuda.empty_cache()

    out_res = {}
    for test_model in args.test_models:
        cnt, num, sr, cer, wer = metrics.calc_metrics(res[test_model])
        out_res[test_model] = (cnt, num, sr, cer, wer)
    for key, value in out_res.items():
        logger.info(f'Model: {key}:\nSR = {value[0]}/{value[1]} = {value[2]}\nCER = {value[3]}\nWER = {value[4]}\n')

    save_path = 'exp_file_results/' + 'cw_' + output_dir.split('/')[-1] + '.pkl'
    with open(save_path, "wb") as f:
        pickle.dump(wav_paths, f)
