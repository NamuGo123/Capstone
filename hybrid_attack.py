import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"  # hides TF-related warnings

import re
import argparse
import random
import time
import datetime
import pickle

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from transformers import (
    HubertModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WavLMModel,
    WavLMConfig,
)

import torchaudio.transforms as Trans_audio
import whisper  # for log_mel_spectrogram in semantic loss
from torch.cuda.amp import autocast

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.utils import load_decoder, load_model

import metrics

# SMACK synthesis
from smack_synthesis import audio_synthesis


# ==========================
# Dataset loading
# ==========================
path = "datasets/librispeech_train_clean100_waveform_spec_speaker_text.pkl"
with open(path, "rb") as f:
    dataset = pickle.load(f)
dataset = dataset[:200]

path = "datasets/librispeech_test_clean_waveform_spec_speaker_text.pkl"
with open(path, "rb") as f:
    tgt_dataset = pickle.load(f)
tgt_dataset = tgt_dataset[:50]


class Attacker:
    def __init__(self, args):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # ------------- Hubert (semantic) -------------
        hubert_model_id = "facebook/hubert-large-ls960-ft"
        self.hubert = HubertModel.from_pretrained(hubert_model_id).eval().to(self.device)

        self.target_layer = args.tgt_layer - 1
        self.tgt_model = args.tgt_model
        self.wav_input = args.wav_input
        self.attack_iters = args.attack_iters
        self.output_dir = args.output_dir
        self.smack_tmp_dir = os.path.join(self.output_dir, "smack_tmp")
        os.makedirs(self.smack_tmp_dir, exist_ok=True)
        self.segment_size = args.segment_size
        self.tgt_text = None
        self.ori_text = args.ori_text
        self.if_slm_loss = args.if_slm_loss

        # SMACK-style init flag
        self.smack_init = getattr(args, "smack_init", False)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # ------------- Wav2Vec2 (ASR) -------------
        wav2vec2_model_id = "facebook/wav2vec2-base-960h"
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_id)
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_id)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_model_id)
        self.wav2vec2.to(self.device)

        # ------------- Whisper (semantic + optional ASR) -------------
        whisper_model_id = "openai/whisper-small"
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_id)
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_id)
        self.whisper.to(self.device)
        self.whisper_encoder = self.whisper.get_encoder()

        # ------------- WavLM (semantic) -------------
        model_name = "microsoft/wavlm-large"
        self.wavlm_config = WavLMConfig.from_pretrained(model_name)
        self.WavLM = WavLMModel.from_pretrained(model_name)
        self.WavLM.eval()
        self.WavLM.to(self.device)

        # ------------- DeepSpeech (target ASR) -------------
        self.labels = [
            "_",
            "'",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            " ",
        ]
        self.labels_map = {c: i for i, c in enumerate(self.labels)}
        self.attack_criterion = nn.CTCLoss(
            blank=self.labels.index("_"), reduction="sum", zero_infinity=True
        )
        tgt_model_path = "pretrained/deepspeech/librispeech_pretrained_v3.ckpt"
        self.deepspeech = load_model(device=self.device, model_path=tgt_model_path)
        self.deep_speech_cfg = TranscribeConfig()
        self.decoder = load_decoder(labels=self.deepspeech.labels, cfg=self.deep_speech_cfg.lm)

        self.sample_rate = 16000
        self.window_size = 0.02
        self.window_stride = 0.01
        self.precision = 32
        self.wav2spec = Trans_audio.Spectrogram(
            n_fft=int(self.sample_rate * self.window_size),
            win_length=int(self.sample_rate * self.window_size),
            hop_length=int(self.sample_rate * self.window_stride),
        ).to(self.device)

    # ==========================
    # Utility helpers
    # ==========================

    @staticmethod
    def _slugify(text: str, max_len: int = 80) -> str:
        if text is None:
            return "none"
        # Remove non-alnum, collapse to underscores
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.upper()).strip("_")
        if not slug:
            slug = "TXT"
        return slug[:max_len]

    def parse_transcript(self, transcript):
        transcript = transcript.replace("\n", "")
        transcript_n = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        transcript_tensor = torch.tensor(transcript_n, dtype=torch.long)
        return transcript_tensor

    # ==========================
    # ASR evaluation (wav2vec2 / whisper)
    # ==========================

    def test_wav(self, x: torch.Tensor, target_model: str):
        """
        Run ASR on a single waveform tensor (1D or [T] / [1, T]) using a target model name.
        """
        # Ensure 1D torch float on CPU
        x_cpu = x.detach().cpu().float()
        if x_cpu.ndim > 1:
            x_cpu = x_cpu.squeeze(0)

        if "wav2vec2" in target_model:
            # Direct PyTorch path: no HF pipeline, no numpy types passed to feature extractor
            input_values = x_cpu.unsqueeze(0).to(self.device)  # [1, T]
            with torch.no_grad():
                logits = self.wav2vec2(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            txt = self.processor.batch_decode(pred_ids)[0]
            return txt

        elif "whisper" in target_model:
            # In this environment Whisper ASR is unstable, so we disable it for now.
            logger.warning(
                "Whisper evaluation is disabled in this environment; returning empty string for whisper transcript."
            )
            return ""

        elif "aliyun" in target_model:
            # Placeholder: the original code calls aliyun_ASR on a saved wav file.
            # You can re-enable this once the aliyun API is available in this environment.
            logger.warning("Aliyun ASR is disabled/not implemented in this environment.")
            return ""

        else:
            return ""

    # ==========================
    # Feature extractors (semantic)
    # ==========================

    def get_feat(self, wav, tgt_model):
        if "hubert" in tgt_model:
            output = self.hubert(wav.to(self.device), output_hidden_states=True)
            hs = output.hidden_states

        elif "whisper" in tgt_model:
            # OpenAI whisper log-mel spec
            mel = whisper.log_mel_spectrogram(wav, n_mels=80, device=self.device)
            hs = self.whisper_encoder(mel, output_hidden_states=True).hidden_states

        elif "wavlm" in tgt_model:
            outputs = self.WavLM(
                wav.to(self.device),
                output_hidden_states=True,
                return_dict=True,
            )
            hs = outputs.hidden_states  # tuple of [batch, seq_len, hidden_dim]

        else:
            # default: wav2vec2 hidden states
            output = self.wav2vec2(wav.to(self.device), output_hidden_states=True)
            hs = output.hidden_states

        if self.target_layer == "avg":
            wav_feat = torch.mean(torch.stack(hs), dim=0)
        else:
            wav_feat = hs[self.target_layer]
            if "deepspeech" in tgt_model:
                wav_feat = wav_feat[1]
        return wav_feat

    # ==========================
    # DeepSpeech helpers
    # ==========================

    def calc_spectrogram(self, y):
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

    # ==========================
    # SMACK-style ptb init
    # ==========================

    def smack_init_ptb(self, wav, idx, tau):
        # Convert reference waveform to a 1D float32 numpy array
        if isinstance(wav, torch.Tensor):
            # [T] or [1, T] tensor on whatever device
            wav_np = wav.detach().cpu().numpy()
        else:
            # e.g. already a numpy array or list
            wav_np = np.asarray(wav)
    
        # Make sure it is numeric and not object
        wav_np = np.asarray(wav_np, dtype=np.float32)
    
        # Squeeze batch/channel dims if present
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
    
        ref_path = os.path.join(self.smack_tmp_dir, f"ref_{idx}.wav")
    
        # --- STRICT WAV NORMALIZATION ---
        # wav is usually a torch tensor [1, T] or [T]; normalize to 1D float32 ndarray
        wav_np = wav.detach().cpu().numpy()
        wav_np = np.asarray(wav_np)
    
        # squeeze any leading singleton dims ([1, T] -> [T])
        if wav_np.ndim > 1:
            wav_np = np.squeeze(wav_np)
    
        # if something upstream produced dtype=object, fix it
        if wav_np.dtype == np.object_:
            wav_np = wav_np.astype(np.float32)
        else:
            wav_np = wav_np.astype(np.float32, copy=False)
    
        sf.write(ref_path, wav_np, self.sample_rate)

        # 2) Build a random prosody vector p_0 ~ SMACK-style (8 x 32)
        exp_p0_tmp = np.exp(np.random.randn(8, 32).astype(np.float32) * 1.0)
        softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
        p_0 = softmax_p0_tmp * 0.25  # shape (8, 32), float32

        try:
            # 3) Call SMACK’s synthesis
            audio_numpy = audio_synthesis(p_0, ref_path, self.tgt_text)
            # int16 -> float32 in [-1, 1]-ish
            adv_float = audio_numpy.astype(np.float32) / 32768.0

            # 4) Length-align via simple interpolation (no extra deps)
            target_len = wav.shape[-1]
            src_len = adv_float.shape[0]
            if src_len != target_len:
                # Linear interpolation onto [0, target_len)
                src_idx = np.linspace(0, src_len - 1, num=target_len, dtype=np.float32)
                base_idx = np.arange(src_len, dtype=np.float32)
                adv_float = np.interp(src_idx, base_idx, adv_float).astype(np.float32)

            adv_tensor = torch.from_numpy(adv_float).to(self.device).unsqueeze(0)  # [1, T]

            ptb_init = adv_tensor - wav
            ptb_init = torch.clamp(ptb_init, -tau, tau)
            ptb = torch.nn.Parameter(ptb_init.requires_grad_(True))
            logger.info("ptb initialized using SMACK-style synthesis.")
            return ptb
        except Exception as e:
            logger.warning(
                f"SMACK-style initialization failed with error {e!r}; falling back to random ptb."
            )
            ptb = torch.randn_like(wav).to(self.device) * 0.005
            ptb = torch.nn.Parameter(ptb.requires_grad_(True))
            return ptb

    # ==========================
    # Main attack loop
    # ==========================

    def attack(self, test_models, idx):
        # Convert input wav to tensor on device
        if isinstance(self.wav_input, torch.Tensor):
            wav = self.wav_input.to(self.device).float()
        else:
            wav = torch.tensor(self.wav_input, dtype=torch.float32, device=self.device)

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        # Sample target sentence from target dataset
        wav_tgt, _, _, self.tgt_text = random.choice(tgt_dataset)
        if isinstance(wav_tgt, torch.Tensor):
            wav_tgt = wav_tgt.float()
        else:
            wav_tgt = torch.tensor(wav_tgt, dtype=torch.float32)
        if wav_tgt.ndim == 1:
            wav_tgt = wav_tgt.unsqueeze(0)
        wav_tgt = wav_tgt.to(self.device)

        # Pad both to same segment_size
        if "whisper" not in self.tgt_model:
            self.segment_size = max(wav.size(-1), wav_tgt.size(-1))
        if wav.size(-1) < self.segment_size:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)), "constant")
        if wav_tgt.size(-1) < self.segment_size:
            wav_tgt = F.pad(wav_tgt, (0, self.segment_size - wav_tgt.size(-1)), "constant")

        wav_tgt = wav_tgt.to(self.device)
        logger.info(wav.shape)
        logger.info(wav_tgt.shape)

        tau = 8 / 255.0

        # ---- ptb initialization (random vs SMACK) ----
        if self.smack_init:
            ptb = self.smack_init_ptb(wav, idx, tau)
        else:
            ptb = torch.randn_like(wav).to(self.device) * 0.005
            ptb = torch.nn.Parameter(ptb.requires_grad_(True))

        optimizer = torch.optim.Adam([ptb], 0.001, (0.9, 0.999), weight_decay=1e-4)

        org_feat = self.get_feat(wav, self.tgt_model)
        tgt_feat = self.get_feat(wav_tgt, self.tgt_model)

        logger.info(f"tgt_shape = {tgt_feat.shape}, org_shape = {org_feat.shape}")
        tot_loss = 0.0
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
                sim_loss = torch.tensor(0.0, device=self.device)

            suc_flag = (
                metrics.clean_text(txts[0]) == metrics.clean_text(self.tgt_text)
            )
            if self.if_slm_loss:
                suc_flag = suc_flag and (sim_loss < 0.05)
            if suc_flag:
                logger.info("attack success!")
                out_log = f"deepspeech: {txts[0]}"
                for tgt_model in test_models:
                    txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
                    out_log += f"\n{tgt_model}: {txt}"
                logger.info(out_log)
                break

            loss = attack_loss
            if self.if_slm_loss:
                loss = loss + 20.0 * sim_loss

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
                now_loss = tot_loss / 10.0
                tot_loss = 0.0
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                out_log = f"Elapsed [{et}], loss [{i + 1}] = {now_loss:.7f}"
                for key, value in loss_dict.items():
                    out_log += f", {key} = {value:.7f}"
                logger.info(out_log)

            if (i + 1) % 100 == 0:
                out_log = f"deepspeech: {txts[0]}"
                for tgt_model in test_models:
                    txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
                    out_log += f"\n{tgt_model}: {txt}"
                logger.info(out_log)

        # ==========================
        # Save final adversarial audio
        # ==========================
        ori_slug = self._slugify(self.ori_text, max_len=40)
        tgt_slug = self._slugify(self.tgt_text, max_len=40)
        filename = f"{idx}_SLM_{i}_{ori_slug}_{tgt_slug}.wav"
        path = os.path.join(self.output_dir, filename)

        self.save_wav(adv_wav.squeeze(0).detach().cpu().numpy(), path)

        # Evaluate on test_models
        res = {m: [] for m in test_models}
        for tgt_model in test_models:
            txt = self.test_wav(adv_wav.squeeze(0), tgt_model)
            res[tgt_model].append((self.ori_text, txt))
        return res, path

    def save_wav(self, wav, path):
        """
        Robust WAV saver:
        - Forces `wav` to a 1D float32 numpy array
        - Handles cases where `wav` is a list, tuple, or dtype=object array
        - Falls back to a shorter filename if the primary path fails
        """
        import numpy as np
        import os
        import logging
        import soundfile as sf
    
        # --- Normalize to numpy array ---
        wav_np = np.asarray(wav)
    
        # Handle pathological case: array of objects (e.g., array of arrays)
        if wav_np.dtype == np.object_:
            try:
                # Try to flatten by concatenating all sub-arrays
                parts = []
                for x in wav_np:
                    parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
                wav_np = np.concatenate(parts, axis=0)
            except Exception as e:
                logging.warning(
                    "save_wav: failed to concatenate object array, "
                    "falling back to simple flatten with best-effort cast (%r)",
                    e,
                )
                wav_np = np.array(wav_np.tolist(), dtype=np.float32).reshape(-1)
        else:
            # Regular numeric array: just cast
            wav_np = wav_np.astype(np.float32)
    
        # Ensure mono 1-D
        wav_np = np.squeeze(wav_np)
        if wav_np.ndim > 1:
            wav_np = wav_np.reshape(-1)
    
        # --- Try primary path ---
        try:
            sf.write(path, wav_np, self.sample_rate)
            return path
        except Exception as e:
            logging.warning(
                "Primary save_wav failed for %r with error %r; "
                "retrying with a shorter filename.",
                path,
                e,
            )
    
        # --- Fallback: shorter / safer filename ---
        try:
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            # Truncate basename to something safe
            safe_basename = basename[-80:]  # last 80 chars
            fallback_path = os.path.join(dirname, f"clip_{safe_basename}")
            sf.write(fallback_path, wav_np, self.sample_rate)
            return fallback_path
        except Exception as e:
            logging.error(
                "Fallback save_wav also failed for %r with error %r. "
                "Waveform will not be saved.",
                path,
                e,
            )
            # Last resort: don’t crash the attack, just return original path
            return path


# ==========================
# Script entry point
# ==========================

import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial attack on ASR models.")
    parser.add_argument("--wav_input", type=str, help="Path to the original audio file (overridden by dataset).")
    parser.add_argument("--ori_text", type=str, help="Original transcript (overridden by dataset).")
    parser.add_argument(
        "--tgt_layer", type=int, default=9, help="Target layer for feature extraction."
    )
    parser.add_argument(
        "--tgt_model",
        type=str,
        default="hubert",
        choices=["hubert", "wav2vec2", "whisper", "deepspeech", "wavlm"],
        help="Target model for semantic feature matching.",
    )
    parser.add_argument(
        "--attack_iters", type=int, default=1500, help="Number of attack iterations."
    )
    parser.add_argument(
        "--segment_size", type=int, default=480000, help="Segment size for audio processing."
    )
    parser.add_argument(
        "--if_slm_loss", action="store_true", help="Enable semantic loss matching."
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment", help="Base directory to save adversarial examples."
    )
    parser.add_argument(
        "--test_models",
        nargs="+",
        default=["wav2vec2", "whisper"],
        help="List of models to test the adversarial examples on.",
    )
    parser.add_argument(
        "--prefix_name", type=str, default="run_main_", help="Prefix for the log file name."
    )
    parser.add_argument(
        "--smack_init",
        action="store_true",
        help="Use SMACK-style TTS synthesis to initialize ptb instead of random noise.",
    )

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    output_dir = os.path.join(args.output_dir, args.tgt_model + "_" + str(args.tgt_layer))
    today_date = datetime.datetime.now().strftime("%m%d")
    log_path = args.prefix_name + output_dir.split("/")[-1] + f"_{today_date}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    res = {m: [] for m in args.test_models}
    wav_paths = []

    for i, (wav_input, spec, spk_id, txt) in enumerate(dataset):
        logger.info(
            "\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            "===============================================================================================================================\n\n"
        )

        # Override args for this sample
        args.wav_input = wav_input
        args.ori_text = txt
        args.output_dir = output_dir

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
        logger.info(
            f"Model: {key}:\nSR = {value[0]}/{value[1]} = {value[2]}\nCER = {value[3]}\nWER = {value[4]}\n"
        )

    save_path = "exp_file_results/" + "cw_" + output_dir.split("/")[-1] + ".pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(wav_paths, f)