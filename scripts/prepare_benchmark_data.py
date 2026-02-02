#!/usr/bin/env python3
"""
下载示例音频文件用于 benchmark 测试
这个脚本会下载一些公开的中文语音样本用于测试。
你也可以使用自己的音频文件。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_sample_audio():
    """下载示例音频"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        from modelscope.hub.file_download import model_file_download

        output_dir = Path(__file__).parent.parent / "data" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("下载示例音频文件...")

        # 从 ModelScope 下载示例
        # 使用 FunASR 官方示例音频
        try:
            # 下载 paraformer 模型附带的示例音频
            model_dir = snapshot_download(
                "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                cache_dir=str(output_dir / ".cache"),
            )
            example_dir = Path(model_dir) / "example"
            if example_dir.exists():
                import shutil
                for f in example_dir.glob("*.wav"):
                    dest = output_dir / f.name
                    if not dest.exists():
                        shutil.copy(f, dest)
                        print(f"  复制: {f.name}")
        except Exception as e:
            print(f"  从 ModelScope 下载失败: {e}")

        # 创建示例参考文本
        ref_file = output_dir / "ref.txt"
        if not ref_file.exists():
            # 常见的测试句子
            ref_content = """# 参考文本文件
# 格式: 文件名<tab>参考文本
# 请根据你的测试音频修改

# 示例:
# test.wav	这是一段测试音频的正确转写文本
"""
            with open(ref_file, 'w', encoding='utf-8') as f:
                f.write(ref_content)
            print(f"  创建参考文本模板: {ref_file}")

        print(f"\n音频文件已保存到: {output_dir}")
        print("请将你的测试音频文件放入该目录")

    except ImportError:
        print("提示: 安装 modelscope 可自动下载示例音频")
        print("  pip install modelscope")
        print("\n你也可以手动将测试音频放入 data/benchmark/ 目录")


def create_test_audio_from_tts():
    """使用 TTS 生成测试音频（备用方案）"""
    try:
        import edge_tts
        import asyncio

        output_dir = Path(__file__).parent.parent / "data" / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        test_sentences = [
            ("test_short.mp3", "今天天气真好，适合出去散步。"),
            ("test_medium.mp3", "人工智能技术正在深刻改变我们的生活方式，从智能手机到自动驾驶汽车，无处不在。"),
            ("test_hotword.mp3", "请问贵公司的客服电话是多少？我想咨询一下产品售后服务。"),
        ]

        async def generate():
            for filename, text in test_sentences:
                output_path = output_dir / filename
                if output_path.exists():
                    print(f"  跳过已存在: {filename}")
                    continue

                print(f"  生成: {filename}")
                communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                await communicate.save(str(output_path))

        asyncio.run(generate())
        print(f"\n测试音频已生成到: {output_dir}")

        # 创建对应的参考文本
        ref_file = output_dir / "ref.txt"
        ref_content = "\n".join([f"{name}\t{text}" for name, text in test_sentences])
        with open(ref_file, 'w', encoding='utf-8') as f:
            f.write(ref_content)
        print(f"参考文本已保存到: {ref_file}")

    except ImportError:
        print("提示: 安装 edge-tts 可自动生成测试音频")
        print("  pip install edge-tts")


def main():
    print("=" * 60)
    print("ASR Benchmark 测试数据准备")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已有音频文件
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    existing_audio = [f for f in output_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in audio_extensions]

    if existing_audio:
        print(f"\n已存在 {len(existing_audio)} 个音频文件:")
        for f in existing_audio[:5]:
            print(f"  - {f.name}")
        if len(existing_audio) > 5:
            print(f"  ... 还有 {len(existing_audio) - 5} 个文件")
        print("\n可以直接运行 benchmark:")
        print(f"  python scripts/benchmark_asr.py --audio {output_dir}")
        return

    print("\n选择数据准备方式:")
    print("1. 使用 TTS 生成测试音频 (需要 edge-tts)")
    print("2. 从 ModelScope 下载示例 (需要 modelscope)")
    print("3. 手动准备 (将音频放入 data/benchmark/)")

    try:
        choice = input("\n请选择 (1/2/3): ").strip()
    except EOFError:
        choice = "3"

    if choice == "1":
        create_test_audio_from_tts()
    elif choice == "2":
        download_sample_audio()
    else:
        print(f"\n请将测试音频文件放入: {output_dir}")
        print("支持格式: .wav, .mp3, .flac, .m4a")


if __name__ == "__main__":
    main()
