"""ONNX 模型加载和音频编码

移植自 CapsWriter-Offline
"""

import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_onnx_models(encoder_path: str, ctc_path: str):
    """加载 ONNX 音频编码器和 CTC Head

    Args:
        encoder_path: Encoder ONNX 模型路径
        ctc_path: CTC ONNX 模型路径

    Returns:
        (encoder_sess, ctc_sess, load_time)
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError("GGUF 后端需要 onnxruntime: pip install onnxruntime")

    t_start = time.perf_counter()
    session_opts = onnxruntime.SessionOptions()
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder_sess = onnxruntime.InferenceSession(
        encoder_path,
        sess_options=session_opts,
        providers=['CPUExecutionProvider']
    )

    ctc_sess = onnxruntime.InferenceSession(
        ctc_path,
        sess_options=session_opts,
        providers=['CPUExecutionProvider']
    )

    t_cost = time.perf_counter() - t_start
    logger.info(f"ONNX models loaded in {t_cost:.2f}s")

    return encoder_sess, ctc_sess, t_cost


def encode_audio(audio: np.ndarray, encoder_sess) -> tuple:
    """使用 ONNX Encoder 获取 LLM 嵌入和 CTC 特征

    Args:
        audio: 音频数据 (float32, 16kHz)
        encoder_sess: ONNX InferenceSession

    Returns:
        (audio_embd, enc_output)
        - audio_embd: LLM 嵌入 [T_llm, 1024]
        - enc_output: CTC 特征 [1, T_enc, 512]
    """
    import onnxruntime

    # Reshape: (1, 1, audio_len) and cast to float32
    audio_input = audio.astype(np.float32).reshape(1, 1, -1)

    in_names = [x.name for x in encoder_sess.get_inputs()]
    out_names = [x.name for x in encoder_sess.get_outputs()]

    input_feed = {
        in_names[0]: onnxruntime.OrtValue.ortvalue_from_numpy(audio_input, 'cpu', 0)
    }

    outputs = encoder_sess.run_with_ort_values(out_names, input_feed)

    # Output 0: enc_output [1, T_enc, 512] (For CTC)
    enc_output = outputs[0].numpy()

    # Output 1: adaptor_output [1, T_llm, 1024] (For LLM)
    audio_embd = outputs[1].numpy().squeeze(0)

    return audio_embd, enc_output
