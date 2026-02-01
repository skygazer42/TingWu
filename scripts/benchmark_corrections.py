# coding: utf-8
"""
纠错效果 A/B 评估框架

对比不同纠错管线配置的 CER/WER，生成对比报告。

用法:
    python scripts/benchmark_corrections.py --audio-dir data/benchmark/audio --ref-dir data/benchmark/reference
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管线配置"""
    name: str
    correction_pipeline: str
    text_correct_enable: bool = False
    text_correct_backend: str = "kenlm"
    punc_restore_enable: bool = False
    punc_merge_enable: bool = False


@dataclass
class BenchmarkResult:
    """评估结果"""
    pipeline_name: str
    avg_cer: float
    avg_wer: float
    total_samples: int
    details: List[Dict] = field(default_factory=list)


def calculate_cer(hypothesis: str, reference: str) -> float:
    """计算字符错误率 (CER)"""
    if not reference:
        return 0.0 if not hypothesis else 1.0

    import numpy as np

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    m, n = len(ref_chars), len(hyp_chars)
    if m == 0:
        return 1.0 if n > 0 else 0.0

    # DP 计算编辑距离
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / m


def calculate_wer(hypothesis: str, reference: str) -> float:
    """计算词错误率 (WER)"""
    if not reference:
        return 0.0 if not hypothesis else 1.0

    import numpy as np

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return 1.0 if n > 0 else 0.0

    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / m


def load_reference(ref_path: Path) -> Dict[str, str]:
    """加载参考文本"""
    references = {}
    if ref_path.is_file():
        # 单文件，每行格式: filename|text
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    name, text = line.strip().split('|', 1)
                    references[name] = text
    elif ref_path.is_dir():
        # 目录，每个 .txt 文件对应一个参考文本
        for txt_file in ref_path.glob("*.txt"):
            name = txt_file.stem
            references[name] = txt_file.read_text(encoding='utf-8').strip()
    return references


def transcribe_with_pipeline(
    audio_path: Path,
    pipeline_config: PipelineConfig,
) -> str:
    """使用指定管线配置进行转写"""
    from src.config import settings

    # 临时修改配置
    original_pipeline = settings.correction_pipeline
    original_text_correct = settings.text_correct_enable
    original_punc_restore = settings.punc_restore_enable
    original_punc_merge = settings.punc_merge_enable

    try:
        settings.correction_pipeline = pipeline_config.correction_pipeline
        settings.text_correct_enable = pipeline_config.text_correct_enable
        settings.text_correct_backend = pipeline_config.text_correct_backend
        settings.punc_restore_enable = pipeline_config.punc_restore_enable
        settings.punc_merge_enable = pipeline_config.punc_merge_enable

        # 重新创建引擎以应用新配置
        from src.core.engine import TranscriptionEngine
        engine = TranscriptionEngine()
        engine.load_all()

        result = engine.transcribe(str(audio_path), apply_hotword=True, apply_llm=False)
        return result.get("text", "")

    finally:
        # 恢复原配置
        settings.correction_pipeline = original_pipeline
        settings.text_correct_enable = original_text_correct
        settings.punc_restore_enable = original_punc_restore
        settings.punc_merge_enable = original_punc_merge


def benchmark_pipeline(
    audio_dir: Path,
    references: Dict[str, str],
    pipeline_config: PipelineConfig,
) -> BenchmarkResult:
    """评估单个管线配置"""
    logger.info(f"Benchmarking pipeline: {pipeline_config.name}")

    cer_scores = []
    wer_scores = []
    details = []

    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    for audio_file in audio_files:
        name = audio_file.stem
        if name not in references:
            logger.warning(f"No reference for {name}, skipping")
            continue

        ref_text = references[name]

        try:
            hyp_text = transcribe_with_pipeline(audio_file, pipeline_config)
            cer = calculate_cer(hyp_text, ref_text)
            wer = calculate_wer(hyp_text, ref_text)

            cer_scores.append(cer)
            wer_scores.append(wer)

            details.append({
                "file": name,
                "reference": ref_text,
                "hypothesis": hyp_text,
                "cer": cer,
                "wer": wer,
            })

            logger.info(f"  {name}: CER={cer:.4f}, WER={wer:.4f}")

        except Exception as e:
            logger.error(f"  {name}: Failed - {e}")

    import numpy as np
    avg_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    avg_wer = float(np.mean(wer_scores)) if wer_scores else 0.0

    return BenchmarkResult(
        pipeline_name=pipeline_config.name,
        avg_cer=avg_cer,
        avg_wer=avg_wer,
        total_samples=len(cer_scores),
        details=details,
    )


def run_benchmark(
    audio_dir: Path,
    ref_path: Path,
    output_path: Optional[Path] = None,
) -> List[BenchmarkResult]:
    """运行完整评估"""
    # 定义要对比的管线配置
    pipelines = [
        PipelineConfig(
            name="baseline",
            correction_pipeline="post_process",
        ),
        PipelineConfig(
            name="hotword_only",
            correction_pipeline="hotword,post_process",
        ),
        PipelineConfig(
            name="hotword+rules",
            correction_pipeline="hotword,rules,post_process",
        ),
        PipelineConfig(
            name="hotword+pycorrector",
            correction_pipeline="hotword,rules,pycorrector,post_process",
            text_correct_enable=True,
            text_correct_backend="kenlm",
        ),
        PipelineConfig(
            name="full_pipeline",
            correction_pipeline="hotword,rules,pycorrector,post_process",
            text_correct_enable=True,
            text_correct_backend="kenlm",
            punc_restore_enable=True,
            punc_merge_enable=True,
        ),
    ]

    references = load_reference(ref_path)
    logger.info(f"Loaded {len(references)} reference texts")

    results = []
    for pipeline_config in pipelines:
        result = benchmark_pipeline(audio_dir, references, pipeline_config)
        results.append(result)

    # 输出对比表格
    print("\n" + "=" * 60)
    print("Pipeline Comparison Results")
    print("=" * 60)
    print(f"{'Pipeline':<25} {'CER':<10} {'WER':<10} {'Samples':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.pipeline_name:<25} {r.avg_cer:<10.4f} {r.avg_wer:<10.4f} {r.total_samples:<10}")
    print("=" * 60)

    # 保存详细结果
    if output_path:
        output_data = {
            "summary": [
                {
                    "pipeline": r.pipeline_name,
                    "avg_cer": r.avg_cer,
                    "avg_wer": r.avg_wer,
                    "total_samples": r.total_samples,
                }
                for r in results
            ],
            "details": {r.pipeline_name: r.details for r in results},
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="纠错效果 A/B 评估")
    parser.add_argument("--audio-dir", type=Path, required=True, help="音频文件目录")
    parser.add_argument("--ref-path", type=Path, required=True, help="参考文本路径")
    parser.add_argument("--output", type=Path, help="结果输出路径 (JSON)")
    args = parser.parse_args()

    if not args.audio_dir.exists():
        logger.error(f"Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    if not args.ref_path.exists():
        logger.error(f"Reference path not found: {args.ref_path}")
        sys.exit(1)

    run_benchmark(args.audio_dir, args.ref_path, args.output)


if __name__ == "__main__":
    main()
