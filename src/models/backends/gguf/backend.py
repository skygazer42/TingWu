"""GGUF 后端主实现

移植自 CapsWriter-Offline，封装为 TingWu ASRBackend 接口。
"""

import os
import time
import ctypes
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from ..base import ASRBackend
from .dataclasses import GGUFConfig, Timings
from .onnx_utils import load_onnx_models, encode_audio
from .ctc_utils import load_ctc_tokens, decode_ctc, align_timestamps
from .llama_cpp import llama_lib, llama_token, llama_batch, ByteDecoder, get_token_embeddings_gguf

logger = logging.getLogger(__name__)


class GGUFBackend(ASRBackend):
    """GGUF 量化模型后端

    使用 ONNX (encoder/CTC) + llama.cpp (GGUF decoder) 进行语音识别。
    移植自 CapsWriter-Offline。

    需要的模型文件:
    - encoder ONNX: Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx
    - CTC ONNX: Fun-ASR-Nano-CTC.int8.onnx
    - decoder GGUF: Fun-ASR-Nano-Decoder.q8_0.gguf
    - tokens.txt: CTC 词表

    还需要 llama.cpp 动态库:
    - Windows: ggml.dll, ggml-base.dll, llama.dll
    - Linux: libggml.so, libggml-base.so, libllama.so
    """

    def __init__(
        self,
        encoder_path: str,
        ctc_path: str,
        decoder_path: str,
        tokens_path: str,
        lib_dir: Optional[str] = None,
        n_predict: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_ubatch: int = 512,
        **kwargs
    ):
        """初始化 GGUF 后端

        Args:
            encoder_path: Encoder ONNX 模型路径
            ctc_path: CTC ONNX 模型路径
            decoder_path: Decoder GGUF 模型路径
            tokens_path: CTC tokens 文件路径
            lib_dir: llama.cpp 库目录
            n_predict: 最大生成 token 数
            n_threads: 推理线程数
            n_threads_batch: 批处理线程数
            n_ubatch: llama.cpp ubatch 大小
        """
        self.config = GGUFConfig(
            encoder_onnx_path=encoder_path,
            ctc_onnx_path=ctc_path,
            decoder_gguf_path=decoder_path,
            tokens_path=tokens_path,
            n_predict=n_predict,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ubatch=n_ubatch,
        )

        # 默认库目录为模型目录下的 bin 子目录
        if lib_dir is None:
            lib_dir = Path(decoder_path).parent / "bin"
        self.lib_dir = Path(lib_dir)

        # 运行时对象
        self._encoder_sess = None
        self._ctc_sess = None
        self._model = None
        self._ctx = None
        self._vocab = None
        self._eos_token = None
        self._embedding_table = None
        self._ctc_id2token = None

        self._loaded = False
        self._stop_tokens = [151643, 151645]  # Qwen2.5 stop tokens

    def load(self) -> None:
        """加载模型"""
        if self._loaded:
            return

        t_start = time.perf_counter()

        # 1. 加载 ONNX 模型
        logger.info("[1/5] Loading ONNX models...")
        self._encoder_sess, self._ctc_sess, _ = load_onnx_models(
            self.config.encoder_onnx_path,
            self.config.ctc_onnx_path
        )

        # 2. 初始化 llama.cpp
        logger.info("[2/5] Initializing llama.cpp...")
        llama_lib.init(self.lib_dir)

        # 3. 加载 GGUF 模型
        logger.info("[3/5] Loading GGUF decoder...")
        model_params = llama_lib.llama_model_default_params()
        model_path = Path(self.config.decoder_gguf_path).resolve()
        self._model = llama_lib.llama_model_load_from_file(
            str(model_path).encode('utf-8'),
            model_params
        )
        if not self._model:
            raise RuntimeError(f"Failed to load GGUF model: {model_path}")

        self._vocab = llama_lib.llama_model_get_vocab(self._model)
        self._eos_token = llama_lib.llama_vocab_eos(self._vocab)

        # 4. 加载 embedding 权重
        logger.info("[4/5] Loading embeddings...")
        self._embedding_table = get_token_embeddings_gguf(self.config.decoder_gguf_path)

        # 5. 创建上下文
        logger.info("[5/5] Creating LLM context...")
        self._ctx = self._create_context()

        # 加载 CTC 词表
        self._ctc_id2token = load_ctc_tokens(self.config.tokens_path)

        self._loaded = True
        t_cost = time.perf_counter() - t_start
        logger.info(f"GGUF backend loaded in {t_cost:.2f}s")

    def _create_context(self):
        """创建 LLM 上下文"""
        ctx_params = llama_lib.llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_batch = 2048
        ctx_params.n_ubatch = self.config.n_ubatch
        ctx_params.embeddings = False
        ctx_params.no_perf = True
        ctx_params.n_threads = self.config.n_threads or (os.cpu_count() // 2)
        ctx_params.n_threads_batch = self.config.n_threads_batch or os.cpu_count()
        return llama_lib.llama_init_from_model(self._model, ctx_params)

    def _build_prompt(self, hotwords: Optional[List[str]] = None):
        """构建 prompt embeddings"""
        prefix_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

        if hotwords:
            hotwords_str = ", ".join(hotwords[:20])  # 最多 20 个热词
            prefix_prompt += f"热词列表：[{hotwords_str}]\n"

        prefix_prompt += "语音转写："
        suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_tokens = llama_lib.text_to_tokens(self._vocab, prefix_prompt)
        suffix_tokens = llama_lib.text_to_tokens(self._vocab, suffix_prompt)

        prefix_embd = self._embedding_table[prefix_tokens].astype(np.float32)
        suffix_embd = self._embedding_table[suffix_tokens].astype(np.float32)

        return prefix_embd, suffix_embd, len(prefix_tokens), len(suffix_tokens)

    def _decode_llm(self, full_embd: np.ndarray, n_input_tokens: int) -> tuple:
        """执行 LLM 解码"""
        # 清空 KV cache
        mem = llama_lib.llama_get_memory(self._ctx)
        llama_lib.llama_memory_clear(mem, True)

        # 注入 embeddings
        t_inject_start = time.perf_counter()
        batch_embd = llama_lib.llama_batch_init(n_input_tokens, full_embd.shape[1], 1)
        batch_embd.n_tokens = n_input_tokens
        batch_embd.token = ctypes.cast(None, ctypes.POINTER(llama_token))

        if not full_embd.flags['C_CONTIGUOUS']:
            full_embd = np.ascontiguousarray(full_embd)
        ctypes.memmove(batch_embd.embd, full_embd.ctypes.data, full_embd.nbytes)

        for k in range(n_input_tokens):
            batch_embd.pos[k] = k
            batch_embd.n_seq_id[k] = 1
            batch_embd.seq_id[k][0] = 0
            batch_embd.logits[k] = 1 if k == n_input_tokens - 1 else 0

        ret = llama_lib.llama_decode(self._ctx, batch_embd)
        llama_lib.llama_batch_free(batch_embd)
        if ret != 0:
            raise RuntimeError(f"LLM decode failed (ret={ret})")

        t_inject = time.perf_counter() - t_inject_start

        # 生成循环
        t_gen_start = time.perf_counter()
        vocab_size = llama_lib.llama_vocab_n_tokens(self._vocab)
        batch_text = llama_lib.llama_batch_init(1, 0, 1)
        batch_text.n_tokens = 1

        generated_text = ""
        current_pos = n_input_tokens
        decoder_utf8 = ByteDecoder()
        tokens_generated = 0

        for step in range(self.config.n_predict):
            logits_ptr = llama_lib.llama_get_logits(self._ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))

            if token_id == self._eos_token or token_id in self._stop_tokens:
                break

            raw_bytes = llama_lib.token_to_bytes(self._vocab, token_id)
            text_piece = decoder_utf8.decode(raw_bytes)
            generated_text += text_piece
            tokens_generated += 1

            # 熔断检测
            if step == 0:
                last_token_id = token_id
                consecutive_cnt = 1
            elif token_id == last_token_id:
                consecutive_cnt += 1
                if consecutive_cnt > 20:
                    logger.warning("Detected abnormal repetition, breaking")
                    break
            else:
                last_token_id = token_id
                consecutive_cnt = 1

            batch_text.token[0] = token_id
            batch_text.pos[0] = current_pos
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1

            if llama_lib.llama_decode(self._ctx, batch_text) != 0:
                break
            current_pos += 1

        remaining = decoder_utf8.flush()
        generated_text += remaining

        llama_lib.llama_batch_free(batch_text)
        t_gen = time.perf_counter() - t_gen_start

        return generated_text.strip(), tokens_generated, t_inject, t_gen

    def transcribe(
        self,
        audio_input: Union[bytes, str, Path, np.ndarray],
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写

        Args:
            audio_input: 音频输入
            hotwords: 热词字符串（换行分隔）
            **kwargs: 其他参数

        Returns:
            转写结果字典
        """
        if not self._loaded:
            self.load()

        timings = Timings()
        t_start = time.perf_counter()

        # 1. 加载音频
        audio = self._load_audio(audio_input)

        # 2. 音频编码
        t_enc = time.perf_counter()
        audio_embd, enc_output = encode_audio(audio, self._encoder_sess)
        timings.encode = time.perf_counter() - t_enc

        # 3. CTC 解码
        t_ctc = time.perf_counter()
        ctc_logits = self._ctc_sess.run(None, {"enc_output": enc_output})[0]
        ctc_text, ctc_results = decode_ctc(ctc_logits, self._ctc_id2token)
        timings.ctc = time.perf_counter() - t_ctc

        # 4. 准备 Prompt
        t_prep = time.perf_counter()
        hotwords_list = None
        if hotwords:
            hotwords_list = [hw.strip() for hw in hotwords.split('\n') if hw.strip()]

        prefix_embd, suffix_embd, n_prefix, n_suffix = self._build_prompt(hotwords_list)
        full_embd = np.concatenate([prefix_embd, audio_embd.astype(np.float32), suffix_embd], axis=0)
        timings.prepare = time.perf_counter() - t_prep

        # 5. LLM 解码
        text, n_gen, t_inject, t_gen = self._decode_llm(full_embd, full_embd.shape[0])
        timings.inject = t_inject
        timings.llm_generate = t_gen

        # 6. 时间戳对齐
        t_align = time.perf_counter()
        aligned = []
        if ctc_results:
            aligned = align_timestamps(ctc_results, text)
        timings.align = time.perf_counter() - t_align

        timings.total = time.perf_counter() - t_start

        # 构建句子信息
        sentence_info = []
        if aligned:
            sentence_info.append({
                "text": text,
                "start": aligned[0]["start"] if aligned else 0,
                "end": aligned[-1]["start"] + 0.1 if aligned else 0,
            })

        return {
            "text": text,
            "sentence_info": sentence_info,
            "ctc_text": ctc_text,
            "timings": {
                "encode": timings.encode,
                "ctc": timings.ctc,
                "prepare": timings.prepare,
                "inject": timings.inject,
                "llm_generate": timings.llm_generate,
                "align": timings.align,
                "total": timings.total,
            }
        }

    def _load_audio(self, audio_input) -> np.ndarray:
        """加载音频为 numpy 数组"""
        if isinstance(audio_input, np.ndarray):
            return audio_input.astype(np.float32)

        if isinstance(audio_input, (bytes, bytearray)):
            data = bytes(audio_input)

            # TingWu's HTTP layer uses raw PCM16LE (16kHz, mono) bytes.
            # Keep compatibility with WAV container bytes too.
            is_wav = len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"
            if is_wav:
                import io
                import soundfile as sf

                audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                return audio

            # Raw PCM16LE 16k mono.
            if len(data) % 2 != 0:
                data = data[: len(data) - 1]
            audio_i16 = np.frombuffer(data, dtype=np.int16)
            return audio_i16.astype(np.float32) / 32768.0

        if isinstance(audio_input, (str, Path)):
            import soundfile as sf
            audio, sr = sf.read(str(audio_input), dtype='float32')
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return audio

        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_hotwords(self) -> bool:
        return True

    @property
    def supports_speaker(self) -> bool:
        return False

    def unload(self) -> None:
        """卸载模型"""
        if self._ctx:
            llama_lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            llama_lib.llama_model_free(self._model)
            llama_lib.llama_backend_free()
            self._model = None
        self._loaded = False
        logger.info("GGUF backend unloaded")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "GGUFBackend",
            "type": "gguf",
            "supports_streaming": self.supports_streaming,
            "supports_hotwords": self.supports_hotwords,
            "supports_speaker": self.supports_speaker,
            "loaded": self._loaded,
        }
