"""llama.cpp 绑定 (ctypes)

移植自 CapsWriter-Offline nano_llama.py
提供 llama.cpp 的 Python 绑定，用于 GGUF 模型推理。
"""

import os
import sys
import ctypes
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Type Definitions
llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


class LlamaLibrary:
    """llama.cpp 库封装"""

    def __init__(self):
        self.llama = None
        self.ggml = None
        self.ggml_base = None
        self._initialized = False
        self._log_callback_ref = None

        # Function pointers
        self.llama_log_set = None
        self.llama_backend_init = None
        self.llama_backend_free = None
        self.llama_model_default_params = None
        self.llama_model_load_from_file = None
        self.llama_model_free = None
        self.llama_model_get_vocab = None
        self.llama_context_default_params = None
        self.llama_init_from_model = None
        self.llama_free = None
        self.llama_batch_init = None
        self.llama_batch_free = None
        self.llama_decode = None
        self.llama_get_logits = None
        self.llama_tokenize = None
        self.llama_vocab_n_tokens = None
        self.llama_vocab_eos = None
        self.llama_token_to_piece = None
        self.llama_get_memory = None
        self.llama_memory_clear = None

    def init(self, lib_dir: Path):
        """初始化 llama.cpp 库"""
        if self._initialized:
            return

        original_cwd = Path.cwd()

        try:
            os.chdir(lib_dir)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(os.getcwd())
            os.environ['PATH'] = os.getcwd() + os.pathsep + os.environ['PATH']

            # 根据平台加载库
            if sys.platform == 'win32':
                self.ggml = ctypes.CDLL("./ggml.dll")
                self.ggml_base = ctypes.CDLL("./ggml-base.dll")
                self.llama = ctypes.CDLL("./llama.dll")
            else:
                self.ggml = ctypes.CDLL("./libggml.so")
                self.ggml_base = ctypes.CDLL("./libggml-base.so")
                self.llama = ctypes.CDLL("./libllama.so")

            self._setup_functions()
            self._initialized = True
            logger.info(f"llama.cpp library initialized from {lib_dir}")

        finally:
            os.chdir(original_cwd)

    def _setup_functions(self):
        """设置函数签名"""
        # Log callback
        LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
        self.llama_log_set = self.llama.llama_log_set
        self.llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
        self.llama_log_set.restype = None

        # 设置日志回调
        self._log_callback_ref = LOG_CALLBACK(self._log_callback)
        self.llama_log_set(self._log_callback_ref, None)

        # Load all backends
        ggml_backend_load_all = self.ggml.ggml_backend_load_all
        ggml_backend_load_all.argtypes = []
        ggml_backend_load_all.restype = None
        ggml_backend_load_all()

        # Backend init
        self.llama_backend_init = self.llama.llama_backend_init
        self.llama_backend_init.argtypes = []
        self.llama_backend_init.restype = None
        self.llama_backend_init()

        self.llama_backend_free = self.llama.llama_backend_free
        self.llama_backend_free.argtypes = []
        self.llama_backend_free.restype = None

        # Model
        self.llama_model_default_params = self.llama.llama_model_default_params
        self.llama_model_default_params.argtypes = []
        self.llama_model_default_params.restype = llama_model_params

        self.llama_model_load_from_file = self.llama.llama_model_load_from_file
        self.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        self.llama_model_load_from_file.restype = ctypes.c_void_p

        self.llama_model_free = self.llama.llama_model_free
        self.llama_model_free.argtypes = [ctypes.c_void_p]
        self.llama_model_free.restype = None

        self.llama_model_get_vocab = self.llama.llama_model_get_vocab
        self.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self.llama_model_get_vocab.restype = ctypes.c_void_p

        # Context
        self.llama_context_default_params = self.llama.llama_context_default_params
        self.llama_context_default_params.argtypes = []
        self.llama_context_default_params.restype = llama_context_params

        self.llama_init_from_model = self.llama.llama_init_from_model
        self.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
        self.llama_init_from_model.restype = ctypes.c_void_p

        self.llama_free = self.llama.llama_free
        self.llama_free.argtypes = [ctypes.c_void_p]
        self.llama_free.restype = None

        # Batch
        self.llama_batch_init = self.llama.llama_batch_init
        self.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.llama_batch_init.restype = llama_batch

        self.llama_batch_free = self.llama.llama_batch_free
        self.llama_batch_free.argtypes = [llama_batch]
        self.llama_batch_free.restype = None

        # Decode
        self.llama_decode = self.llama.llama_decode
        self.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
        self.llama_decode.restype = ctypes.c_int32

        # Logits
        self.llama_get_logits = self.llama.llama_get_logits
        self.llama_get_logits.argtypes = [ctypes.c_void_p]
        self.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        # Tokenize
        self.llama_tokenize = self.llama.llama_tokenize
        self.llama_tokenize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.POINTER(llama_token), ctypes.c_int32,
            ctypes.c_bool, ctypes.c_bool,
        ]
        self.llama_tokenize.restype = ctypes.c_int32

        # Vocab
        self.llama_vocab_n_tokens = self.llama.llama_vocab_n_tokens
        self.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        self.llama_vocab_n_tokens.restype = ctypes.c_int32

        self.llama_vocab_eos = self.llama.llama_vocab_eos
        self.llama_vocab_eos.argtypes = [ctypes.c_void_p]
        self.llama_vocab_eos.restype = llama_token

        self.llama_token_to_piece = self.llama.llama_token_to_piece
        self.llama_token_to_piece.argtypes = [
            ctypes.c_void_p, llama_token, ctypes.c_char_p,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
        ]
        self.llama_token_to_piece.restype = ctypes.c_int

        # Memory
        self.llama_get_memory = self.llama.llama_get_memory
        self.llama_get_memory.argtypes = [ctypes.c_void_p]
        self.llama_get_memory.restype = ctypes.c_void_p

        self.llama_memory_clear = self.llama.llama_memory_clear
        self.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.llama_memory_clear.restype = None

    @staticmethod
    def _log_callback(level, message, user_data):
        """llama.cpp 日志回调"""
        if not message:
            return
        try:
            msg_str = message.decode('utf-8', errors='replace').strip()
            if not msg_str or msg_str in ['.', '\n']:
                return

            if level == 2:
                logger.error(f"[llama.cpp] {msg_str}")
            elif level == 3:
                logger.warning(f"[llama.cpp] {msg_str}")
            elif level == 4:
                logger.info(f"[llama.cpp] {msg_str}")
            else:
                logger.debug(f"[llama.cpp] {msg_str}")
        except Exception:
            pass

    def text_to_tokens(self, vocab, text: str):
        """文本分词"""
        text_bytes = text.encode("utf-8")
        n_tokens_max = len(text_bytes) + 32
        tokens = (llama_token * n_tokens_max)()

        n = self.llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, False, True)
        if n < 0:
            return []
        return [tokens[i] for i in range(n)]

    def token_to_bytes(self, vocab, token_id: int) -> bytes:
        """将 token 转换为字节"""
        buf = ctypes.create_string_buffer(256)
        n = self.llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
        if n > 0:
            return buf.raw[:n]
        return b""


class ByteDecoder:
    """字节级解码器，用于处理 BPE 拆分的 UTF-8 字符"""

    def __init__(self):
        self.buffer = b""

    def decode(self, raw_bytes: bytes) -> str:
        self.buffer += raw_bytes
        result = ""
        while self.buffer:
            try:
                result += self.buffer.decode('utf-8')
                self.buffer = b""
                break
            except UnicodeDecodeError as e:
                if e.reason == 'unexpected end of data' or 'invalid continuation' in e.reason:
                    if e.start > 0:
                        result += self.buffer[:e.start].decode('utf-8', errors='replace')
                        self.buffer = self.buffer[e.start:]
                    break
                else:
                    result += self.buffer[:1].decode('utf-8', errors='replace')
                    self.buffer = self.buffer[1:]
        return result

    def flush(self) -> str:
        if self.buffer:
            result = self.buffer.decode('utf-8', errors='replace')
            self.buffer = b""
            return result
        return ""


def get_token_embeddings_gguf(model_path: str, cache_dir: str = None) -> np.ndarray:
    """从 GGUF 读取 token_embd.weight

    支持 F16/F32 和 Q8_0 量化格式，使用缓存机制。
    """
    try:
        import gguf
    except ImportError:
        raise ImportError("GGUF 后端需要 gguf: pip install gguf")

    if cache_dir is None:
        cache_dir = os.path.dirname(model_path)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cache_path = os.path.join(cache_dir, f"{model_name}.embd.npy")

    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(model_path):
            return np.load(cache_path)

    reader = gguf.GGUFReader(model_path, mode='r')

    for t in reader.tensors:
        if t.name == "token_embd.weight":
            if t.tensor_type == 8:  # GGML_TYPE_Q8_0
                block_size_bytes = 34
                num_values_per_block = 32

                raw_data = t.data
                data_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                n_blocks = data_u8.size // block_size_bytes

                blocks = data_u8.reshape(n_blocks, block_size_bytes)
                deltas = blocks[:, :2].view(np.float16).flatten()
                quants = blocks[:, 2:].view(np.int8)

                data = (deltas[:, np.newaxis] * quants).flatten().astype(np.float32).reshape(-1, 1024)
            else:
                data = t.data
                if data.dtype == np.float16:
                    data = data.astype(np.float32)

            np.save(cache_path, data)
            return data

    return None


# 全局实例
llama_lib = LlamaLibrary()
