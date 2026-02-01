#!/usr/bin/env python3
"""
ASR 模型性能对比 Benchmark 脚本

测试模型:
1. paraformer-zh (当前使用)
2. paraformer-zh + ONNX Runtime
3. Fun-ASR-Nano-2512
4. SenseVoice

后端测试 (通过新的后端抽象层):
- pytorch: PyTorch 后端
- onnx: ONNX Runtime 后端
- sensevoice: SenseVoice 后端

测量指标:
- RTF (Real-Time Factor): 处理时间 / 音频时长
- 内存占用 (峰值)
- CER (Character Error Rate): 需要提供参考文本
- 首次推理延迟

使用方法:
    python scripts/benchmark_asr.py --audio data/benchmark/test.wav
    python scripts/benchmark_asr.py --audio data/benchmark/ --ref data/benchmark/ref.txt
    python scripts/benchmark_asr.py --audio data/benchmark/ --models backend-pytorch backend-onnx
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import calculate_cer, calculate_wer


@dataclass
class BenchmarkResult:
    """单次测试结果"""
    model_name: str
    audio_file: str
    audio_duration: float  # 秒
    inference_time: float  # 秒
    rtf: float  # Real-Time Factor
    memory_mb: float  # 峰值内存 MB
    text: str  # 识别结果
    cer: Optional[float] = None  # 字符错误率
    error: Optional[str] = None  # 错误信息


@dataclass
class ModelBenchmark:
    """模型测试汇总"""
    model_name: str
    total_audio_duration: float = 0.0
    total_inference_time: float = 0.0
    avg_rtf: float = 0.0
    avg_memory_mb: float = 0.0
    avg_cer: Optional[float] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    supported: bool = True
    error: Optional[str] = None


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception:
        # 备用方案：使用 soundfile
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0


def get_memory_usage() -> float:
    """获取当前进程内存使用（MB）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class ASRBenchmarker:
    """ASR 模型基准测试器"""

    def __init__(self, device: str = "cpu", warmup: bool = True):
        self.device = device
        self.warmup = warmup
        self.models: Dict[str, Any] = {}

    def _warmup_model(self, model, audio_path: str):
        """预热模型（第一次推理通常较慢）"""
        try:
            if hasattr(model, 'generate'):
                model.generate(input=audio_path)
            elif hasattr(model, '__call__'):
                model(audio_path)
        except Exception:
            pass

    def benchmark_paraformer(self, audio_path: str, ref_text: str = None) -> BenchmarkResult:
        """测试 paraformer-zh 模型"""
        model_name = "paraformer-zh"
        duration = get_audio_duration(audio_path)

        try:
            from funasr import AutoModel

            if model_name not in self.models:
                print(f"  加载 {model_name}...")
                self.models[model_name] = AutoModel(
                    model="paraformer-zh",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc-c",
                    device=self.device,
                    disable_pbar=True,
                    disable_log=True,
                )
                if self.warmup:
                    print(f"  预热 {model_name}...")
                    self._warmup_model(self.models[model_name], audio_path)

            model = self.models[model_name]
            gc.collect()
            mem_before = get_memory_usage()

            start_time = time.perf_counter()
            result = model.generate(
                input=audio_path,
                batch_size_s=300,
            )
            end_time = time.perf_counter()

            mem_after = get_memory_usage()
            inference_time = end_time - start_time
            text = result[0].get("text", "") if result else ""

            cer = calculate_cer(text, ref_text) if ref_text else None

            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=inference_time,
                rtf=inference_time / duration if duration > 0 else 0,
                memory_mb=max(mem_after - mem_before, 0),
                text=text,
                cer=cer,
            )
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=str(e),
            )

    def benchmark_paraformer_onnx(self, audio_path: str, ref_text: str = None) -> BenchmarkResult:
        """测试 paraformer-zh ONNX 版本"""
        model_name = "paraformer-zh-onnx"
        duration = get_audio_duration(audio_path)

        try:
            from funasr_onnx import Paraformer

            if model_name not in self.models:
                print(f"  加载 {model_name}...")
                # ONNX 模型需要预先下载
                self.models[model_name] = Paraformer(
                    model_dir="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
                    quantize=True,  # INT8 量化
                )
                if self.warmup:
                    print(f"  预热 {model_name}...")
                    self._warmup_model(self.models[model_name], audio_path)

            model = self.models[model_name]
            gc.collect()
            mem_before = get_memory_usage()

            start_time = time.perf_counter()
            result = model(audio_path)
            end_time = time.perf_counter()

            mem_after = get_memory_usage()
            inference_time = end_time - start_time
            text = result[0].get("text", "") if result else ""

            cer = calculate_cer(text, ref_text) if ref_text else None

            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=inference_time,
                rtf=inference_time / duration if duration > 0 else 0,
                memory_mb=max(mem_after - mem_before, 0),
                text=text,
                cer=cer,
            )
        except ImportError:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error="funasr_onnx 未安装，请运行: pip install funasr-onnx",
            )
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=str(e),
            )

    def benchmark_fun_asr_nano(self, audio_path: str, ref_text: str = None) -> BenchmarkResult:
        """测试 Fun-ASR-Nano-2512 模型"""
        model_name = "Fun-ASR-Nano-2512"
        duration = get_audio_duration(audio_path)

        try:
            from funasr import AutoModel

            if model_name not in self.models:
                print(f"  加载 {model_name}...")
                self.models[model_name] = AutoModel(
                    model="FunAudioLLM/Fun-ASR-Nano-2512",
                    trust_remote_code=True,
                    device=self.device,
                )
                if self.warmup:
                    print(f"  预热 {model_name}...")
                    self._warmup_model(self.models[model_name], audio_path)

            model = self.models[model_name]
            gc.collect()
            mem_before = get_memory_usage()

            start_time = time.perf_counter()
            result = model.generate(
                input=audio_path,
                language="中文",
                itn=True,
            )
            end_time = time.perf_counter()

            mem_after = get_memory_usage()
            inference_time = end_time - start_time
            text = result[0].get("text", "") if result else ""

            cer = calculate_cer(text, ref_text) if ref_text else None

            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=inference_time,
                rtf=inference_time / duration if duration > 0 else 0,
                memory_mb=max(mem_after - mem_before, 0),
                text=text,
                cer=cer,
            )
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=str(e),
            )

    def benchmark_sensevoice(self, audio_path: str, ref_text: str = None) -> BenchmarkResult:
        """测试 SenseVoice 模型"""
        model_name = "SenseVoice-Small"
        duration = get_audio_duration(audio_path)

        try:
            from funasr import AutoModel

            if model_name not in self.models:
                print(f"  加载 {model_name}...")
                self.models[model_name] = AutoModel(
                    model="iic/SenseVoiceSmall",
                    trust_remote_code=True,
                    device=self.device,
                )
                if self.warmup:
                    print(f"  预热 {model_name}...")
                    self._warmup_model(self.models[model_name], audio_path)

            model = self.models[model_name]
            gc.collect()
            mem_before = get_memory_usage()

            start_time = time.perf_counter()
            result = model.generate(
                input=audio_path,
                language="zh",
            )
            end_time = time.perf_counter()

            mem_after = get_memory_usage()
            inference_time = end_time - start_time
            text = result[0].get("text", "") if result else ""

            cer = calculate_cer(text, ref_text) if ref_text else None

            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=inference_time,
                rtf=inference_time / duration if duration > 0 else 0,
                memory_mb=max(mem_after - mem_before, 0),
                text=text,
                cer=cer,
            )
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=str(e),
            )

    def benchmark_backend(self, backend_type: str, audio_path: str, ref_text: str = None) -> BenchmarkResult:
        """通过新的后端抽象层测试

        Args:
            backend_type: 后端类型 (pytorch, onnx, sensevoice)
            audio_path: 音频路径
            ref_text: 参考文本
        """
        model_name = f"backend-{backend_type}"
        duration = get_audio_duration(audio_path)

        try:
            from src.models.backends import get_backend

            if model_name not in self.models:
                print(f"  加载后端 {backend_type}...")
                backend = get_backend(
                    backend_type=backend_type,
                    device=self.device,
                    ncpu=4,
                )
                backend.load()
                self.models[model_name] = backend

                if self.warmup:
                    print(f"  预热后端 {backend_type}...")
                    try:
                        backend.transcribe(audio_path)
                    except Exception:
                        pass

            backend = self.models[model_name]
            gc.collect()
            mem_before = get_memory_usage()

            start_time = time.perf_counter()
            result = backend.transcribe(audio_path)
            end_time = time.perf_counter()

            mem_after = get_memory_usage()
            inference_time = end_time - start_time
            text = result.get("text", "") if result else ""

            cer = calculate_cer(text, ref_text) if ref_text else None

            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=inference_time,
                rtf=inference_time / duration if duration > 0 else 0,
                memory_mb=max(mem_after - mem_before, 0),
                text=text,
                cer=cer,
            )
        except ImportError as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=f"后端 {backend_type} 依赖未安装: {e}",
            )
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                audio_file=audio_path,
                audio_duration=duration,
                inference_time=0,
                rtf=0,
                memory_mb=0,
                text="",
                error=str(e),
            )


def collect_audio_files(path: str) -> List[str]:
    """收集音频文件"""
    p = Path(path)
    if p.is_file():
        return [str(p)]

    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma'}
    files = []
    for f in p.rglob('*'):
        if f.suffix.lower() in audio_extensions:
            files.append(str(f))
    return sorted(files)


def load_references(ref_path: str, audio_files: List[str] = None) -> Dict[str, str]:
    """加载参考文本

    支持格式:
    1. 单个文件: TSV 格式 (文件名<tab>文本) 或 JSON 格式
    2. 目录: 每个音频文件对应同名 .txt 文件
    3. 自动检测: 在音频目录中查找同名 .txt 文件

    Args:
        ref_path: 参考文本文件或目录路径
        audio_files: 音频文件列表 (用于目录模式匹配)

    Returns:
        {文件名: 参考文本} 字典
    """
    refs = {}

    if not ref_path:
        # 自动从音频文件旁边查找 .txt 文件
        if audio_files:
            for audio_file in audio_files:
                txt_path = Path(audio_file).with_suffix('.txt')
                if txt_path.exists():
                    try:
                        refs[Path(audio_file).name] = txt_path.read_text(encoding='utf-8').strip()
                    except Exception:
                        pass
        return refs

    ref_p = Path(ref_path)
    if not ref_p.exists():
        return refs

    # 目录模式: 查找所有 .txt 文件
    if ref_p.is_dir():
        for txt_file in ref_p.glob('*.txt'):
            try:
                text = txt_file.read_text(encoding='utf-8').strip()
                # 支持匹配 audio.wav 和 audio.txt
                refs[txt_file.stem] = text
                # 也存储带 .txt 后缀的版本
                refs[txt_file.name] = text
            except Exception:
                pass
        return refs

    # 文件模式
    with open(ref_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 尝试 JSON 格式
    try:
        refs = json.loads(content)
        return refs
    except json.JSONDecodeError:
        pass

    # 尝试 TSV 格式
    for line in content.split('\n'):
        if '\t' in line:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                filename, text = parts
                refs[filename] = text
                # 也存储不带扩展名的版本
                refs[Path(filename).stem] = text

    return refs


def print_results(benchmarks: List[ModelBenchmark]):
    """打印测试结果"""
    print("\n" + "=" * 80)
    print("ASR 模型性能对比结果")
    print("=" * 80)

    # 表头
    print(f"\n{'模型名称':<25} {'平均RTF':>10} {'总时长(s)':>12} {'推理时间(s)':>12} {'内存(MB)':>10} {'CER':>8}")
    print("-" * 80)

    for bm in benchmarks:
        if not bm.supported:
            print(f"{bm.model_name:<25} {'不支持':<10} {bm.error or ''}")
            continue

        cer_str = f"{bm.avg_cer:.2%}" if bm.avg_cer is not None else "N/A"
        print(f"{bm.model_name:<25} {bm.avg_rtf:>10.4f} {bm.total_audio_duration:>12.2f} "
              f"{bm.total_inference_time:>12.2f} {bm.avg_memory_mb:>10.1f} {cer_str:>8}")

    print("-" * 80)

    # 详细结果
    print("\n详细结果:")
    for bm in benchmarks:
        if not bm.supported or not bm.results:
            continue

        print(f"\n### {bm.model_name}")
        for r in bm.results:
            if r.error:
                print(f"  {Path(r.audio_file).name}: 错误 - {r.error}")
            else:
                cer_str = f", CER={r.cer:.2%}" if r.cer is not None else ""
                print(f"  {Path(r.audio_file).name}: RTF={r.rtf:.4f}, 时长={r.audio_duration:.2f}s{cer_str}")
                if r.text:
                    # 截断显示
                    text_preview = r.text[:100] + "..." if len(r.text) > 100 else r.text
                    print(f"    识别: {text_preview}")


def main():
    parser = argparse.ArgumentParser(description="ASR 模型性能对比 Benchmark")
    parser.add_argument("--audio", "-a", required=True, help="音频文件或目录路径")
    parser.add_argument("--ref", "-r", help="参考文本: 文件(TSV/JSON)、目录(含.txt)、或自动检测音频旁.txt")
    parser.add_argument("--device", "-d", default="cpu", help="设备 (cpu/cuda:0)")
    parser.add_argument("--models", "-m", nargs="+",
                       default=["paraformer", "onnx", "nano", "sensevoice"],
                       choices=["paraformer", "onnx", "nano", "sensevoice", "all",
                                "backend-pytorch", "backend-onnx", "backend-sensevoice"],
                       help="要测试的模型 (backend-* 使用新后端抽象层)")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")

    args = parser.parse_args()

    # 收集音频文件
    audio_files = collect_audio_files(args.audio)
    if not audio_files:
        print(f"错误: 未找到音频文件: {args.audio}")
        sys.exit(1)

    print(f"找到 {len(audio_files)} 个音频文件")

    # 加载参考文本 (支持文件/目录/自动检测)
    references = load_references(args.ref, audio_files)
    if references:
        print(f"加载了 {len(references)} 条参考文本")

    # 初始化测试器
    benchmarker = ASRBenchmarker(device=args.device, warmup=not args.no_warmup)

    # 确定要测试的模型
    models_to_test = args.models
    if "all" in models_to_test:
        models_to_test = ["paraformer", "onnx", "nano", "sensevoice"]

    model_benchmarks: List[ModelBenchmark] = []

    # 测试每个模型
    for model_key in models_to_test:
        if model_key == "paraformer":
            benchmark_func = benchmarker.benchmark_paraformer
            model_name = "paraformer-zh"
        elif model_key == "onnx":
            benchmark_func = benchmarker.benchmark_paraformer_onnx
            model_name = "paraformer-zh-onnx"
        elif model_key == "nano":
            benchmark_func = benchmarker.benchmark_fun_asr_nano
            model_name = "Fun-ASR-Nano-2512"
        elif model_key == "sensevoice":
            benchmark_func = benchmarker.benchmark_sensevoice
            model_name = "SenseVoice-Small"
        elif model_key.startswith("backend-"):
            backend_type = model_key.replace("backend-", "")
            benchmark_func = lambda ap, rt=None, bt=backend_type: benchmarker.benchmark_backend(bt, ap, rt)
            model_name = f"backend-{backend_type}"
        else:
            continue

        print(f"\n测试 {model_name}...")
        mb = ModelBenchmark(model_name=model_name)

        for audio_file in audio_files:
            filename = Path(audio_file).name
            ref_text = references.get(filename) or references.get(Path(audio_file).stem)

            print(f"  处理: {filename}")
            result = benchmark_func(audio_file, ref_text)

            if result.error:
                print(f"    错误: {result.error}")
                if "未安装" in result.error or "not found" in result.error.lower():
                    mb.supported = False
                    mb.error = result.error
                    break
            else:
                print(f"    RTF: {result.rtf:.4f}, 时间: {result.inference_time:.2f}s")

            mb.results.append(result)
            mb.total_audio_duration += result.audio_duration
            mb.total_inference_time += result.inference_time

        # 计算平均值
        if mb.results and mb.supported:
            valid_results = [r for r in mb.results if not r.error]
            if valid_results:
                mb.avg_rtf = mb.total_inference_time / mb.total_audio_duration if mb.total_audio_duration > 0 else 0
                mb.avg_memory_mb = sum(r.memory_mb for r in valid_results) / len(valid_results)
                cer_results = [r.cer for r in valid_results if r.cer is not None]
                if cer_results:
                    mb.avg_cer = sum(cer_results) / len(cer_results)

        model_benchmarks.append(mb)

    # 打印结果
    print_results(model_benchmarks)

    # 保存结果
    if args.output:
        output_data = {
            "device": args.device,
            "audio_files": audio_files,
            "benchmarks": [
                {
                    "model_name": mb.model_name,
                    "supported": mb.supported,
                    "error": mb.error,
                    "total_audio_duration": mb.total_audio_duration,
                    "total_inference_time": mb.total_inference_time,
                    "avg_rtf": mb.avg_rtf,
                    "avg_memory_mb": mb.avg_memory_mb,
                    "avg_cer": mb.avg_cer,
                    "results": [
                        {
                            "audio_file": r.audio_file,
                            "audio_duration": r.audio_duration,
                            "inference_time": r.inference_time,
                            "rtf": r.rtf,
                            "memory_mb": r.memory_mb,
                            "text": r.text,
                            "cer": r.cer,
                            "error": r.error,
                        }
                        for r in mb.results
                    ]
                }
                for mb in model_benchmarks
            ]
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
