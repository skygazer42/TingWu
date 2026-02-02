"""核心转写引擎"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from pathlib import Path

import numpy as np

from src.config import settings
from src.models.model_manager import model_manager
from src.core.hotword import PhonemeCorrector
from src.core.hotword.rule_corrector import RuleCorrector
from src.core.hotword.rectification import RectificationRAG
from src.core.speaker import SpeakerLabeler
from src.core.llm import LLMClient, LLMMessage, PromptBuilder
from src.core.llm.roles import get_role
from src.core.text_processor import TextPostProcessor
from src.core.text_processor.text_corrector import TextCorrector
from src.core.audio.chunker import AudioChunker

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """转写引擎 - 整合 ASR + 热词纠错 + 说话人识别 + LLM润色"""

    def __init__(self):
        self.corrector = PhonemeCorrector(
            threshold=settings.hotwords_threshold,
            similar_threshold=settings.hotwords_threshold - 0.2,
            use_faiss=settings.hotword_use_faiss,
            faiss_index_type=settings.hotword_faiss_index_type,
        )
        self.rule_corrector = RuleCorrector()
        self.rectification_rag = RectificationRAG()
        self.speaker_labeler = SpeakerLabeler()

        # 文本后处理器
        self.post_processor = TextPostProcessor.from_config(settings)

        # 长音频分块器
        self.audio_chunker = AudioChunker(
            max_chunk_duration=settings.vad_max_segment_ms / 1000.0,
            min_chunk_duration=5.0,
            overlap_duration=0.5,
        )

        # 通用文本纠错器 (pycorrector)
        self._text_corrector: Optional[TextCorrector] = None
        self._text_correct_enabled = settings.text_correct_enable

        # LLM 组件
        self._llm_client: Optional[LLMClient] = None
        self._prompt_builder: Optional[PromptBuilder] = None
        self._hotwords_list: List[str] = []

        self._hotwords_loaded = False
        self._rules_loaded = False
        self._rectify_loaded = False

    def warmup(self, duration: float = 1.0) -> Dict[str, Any]:
        """预热模型，消除首次推理延迟

        Args:
            duration: 预热音频时长(秒)

        Returns:
            预热结果统计
        """
        import numpy as np

        results = {
            "backend": model_manager.backend.get_info()["name"],
            "warmup_duration": duration,
            "timings": {}
        }

        # 生成静默音频
        sample_rate = 16000
        samples = int(sample_rate * duration)
        silent_audio = np.zeros(samples, dtype=np.float32)

        logger.info(f"Warming up transcription engine ({duration}s audio)...")

        # 预热 ASR 后端
        backend = model_manager.backend
        start_time = time.time()

        try:
            # 尝试调用后端的 warmup 方法
            if hasattr(backend, 'warmup'):
                backend.warmup(duration)
            else:
                # 直接进行一次推理
                _ = backend.transcribe(silent_audio)

            results["timings"]["asr"] = time.time() - start_time
            logger.info(f"ASR warmup completed in {results['timings']['asr']:.2f}s")
        except Exception as e:
            logger.warning(f"ASR warmup failed: {e}")
            results["timings"]["asr"] = -1

        # 预热热词纠错器
        start_time = time.time()
        try:
            if self._hotwords_loaded:
                _ = self.corrector.correct("测试预热文本")
                results["timings"]["hotword"] = time.time() - start_time
        except Exception as e:
            logger.warning(f"Hotword warmup failed: {e}")

        logger.info("Engine warmup completed")
        return results

    @property
    def llm_client(self) -> LLMClient:
        """懒加载 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                backend=settings.llm_backend,
                max_tokens=settings.llm_max_tokens,
                cache_enable=settings.llm_cache_enable,
                cache_size=settings.llm_cache_size,
                cache_ttl=settings.llm_cache_ttl,
            )
        return self._llm_client

    @property
    def text_corrector(self) -> Optional[TextCorrector]:
        """懒加载文本纠错器"""
        if self._text_corrector is None and self._text_correct_enabled:
            try:
                self._text_corrector = TextCorrector(
                    backend=settings.text_correct_backend,
                    device=settings.text_correct_device,
                )
            except Exception as e:
                logger.error(f"Failed to initialize TextCorrector: {e}")
                self._text_correct_enabled = False
        return self._text_corrector

    def load_hotwords(self, path: Optional[str] = None):
        """加载热词"""
        if path is None:
            path = str(settings.hotwords_dir / settings.hotwords_file)

        if Path(path).exists():
            count = self.corrector.load_hotwords_file(path)
            # 缓存热词列表供 LLM 使用
            with open(path, 'r', encoding='utf-8') as f:
                self._hotwords_list = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith('#')
                ]
            logger.info(f"Loaded {count} hotwords from {path}")
            self._hotwords_loaded = True
        else:
            logger.warning(f"Hotwords file not found: {path}")

    def load_rules(self, path: Optional[str] = None):
        """加载规则"""
        if path is None:
            path = str(settings.hotwords_dir / "hot-rules.txt")

        if Path(path).exists():
            count = self.rule_corrector.load_rules_file(path)
            logger.info(f"Loaded {count} rules from {path}")
            self._rules_loaded = True
        else:
            logger.warning(f"Rules file not found: {path}")

    def load_rectify_history(self, path: Optional[str] = None):
        """加载纠错历史"""
        if path is None:
            path = str(settings.hotwords_dir / "hot-rectify.txt")

        if Path(path).exists():
            self.rectification_rag = RectificationRAG(rectify_file=path)
            count = self.rectification_rag.load_history()
            logger.info(f"Loaded {count} rectify records from {path}")
            self._rectify_loaded = True
        else:
            logger.warning(f"Rectify file not found: {path}")

    def load_all(self):
        """加载所有热词相关文件"""
        self.load_hotwords()
        self.load_rules()
        self.load_rectify_history()

    def update_hotwords(self, hotwords: Union[str, List[str]]):
        """更新热词"""
        if isinstance(hotwords, list):
            self._hotwords_list = hotwords
            hotwords = "\n".join(hotwords)
        else:
            self._hotwords_list = [
                line.strip() for line in hotwords.split('\n')
                if line.strip() and not line.startswith('#')
            ]

        count = self.corrector.update_hotwords(hotwords)
        logger.info(f"Updated {count} hotwords")
        self._hotwords_loaded = True

    def _get_injection_hotwords(self, custom_hotwords: Optional[str] = None) -> Optional[str]:
        """获取用于前向注入的热词字符串

        Args:
            custom_hotwords: 自定义热词（优先使用）

        Returns:
            热词字符串（换行分隔）或 None
        """
        # 如果提供了自定义热词，直接使用
        if custom_hotwords:
            return custom_hotwords

        # 检查是否启用前向注入
        if not settings.hotword_injection_enable:
            return None

        # 检查是否有已加载的热词
        if not self._hotwords_list:
            return None

        # 截取最大数量并拼接
        max_count = settings.hotword_injection_max
        hotwords_to_inject = self._hotwords_list[:max_count]
        return "\n".join(hotwords_to_inject)

    def _apply_corrections(self, text: str) -> Tuple[str, List[Tuple[str, str, float]]]:
        """应用纠错管线

        按 correction_pipeline 配置的顺序执行各纠错步骤。
        默认顺序: hotword → rules → pycorrector → post_process

        Returns:
            (纠错后文本, 相似词候选列表 [(原词, 热词, 分数), ...])
        """
        original = text
        all_similars: List[Tuple[str, str, float]] = []
        pipeline = [s.strip() for s in settings.correction_pipeline.split(',') if s.strip()]

        for step in pipeline:
            if step == "hotword" and self._hotwords_loaded and text:
                prev = text
                correction = self.corrector.correct(text)
                text = correction.text
                all_similars.extend(correction.similars)
                if text != prev:
                    logger.debug(f"Hotword correction: {prev!r} -> {text!r}")

            elif step == "rules" and self._rules_loaded and text:
                prev = text
                text = self.rule_corrector.substitute(text)
                if text != prev:
                    logger.debug(f"Rule correction: {prev!r} -> {text!r}")

            elif step == "pycorrector" and self._text_correct_enabled and text and self.text_corrector:
                prev = text
                text, errors = self.text_corrector.correct(text)
                if text != prev:
                    logger.debug(f"Text correction: {prev!r} -> {text!r}, errors={errors}")

            elif step == "post_process":
                text = self.post_processor.process(text)

        if text != original:
            logger.debug(f"Total correction: {original!r} -> {text!r}")

        return text, all_similars

    def _filter_low_confidence(
        self,
        sentence_info: List[Dict[str, Any]],
        threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """标记并处理低置信度片段

        对低置信度的句子进行额外纠错处理。
        支持 pycorrector 和 LLM 两种回退策略。

        Args:
            sentence_info: 句子信息列表
            threshold: 置信度阈值

        Returns:
            处理后的句子信息列表
        """
        if threshold <= 0:
            return sentence_info

        fallback = settings.confidence_fallback

        for sent in sentence_info:
            confidence = sent.get('confidence', 1.0)
            if confidence < threshold:
                sent['low_confidence'] = True
                text = sent.get('text', '')

                if not text:
                    continue

                if fallback == "pycorrector" and self.text_corrector:
                    corrected, _ = self.text_corrector.correct(text)
                    if corrected != text:
                        logger.debug(f"Low confidence ({confidence:.2f}) pycorrector: {text!r} -> {corrected!r}")
                        sent['text'] = corrected
                        sent['correction_method'] = 'pycorrector'

                elif fallback == "llm" and settings.llm_enable:
                    # 使用 LLM 进行纠错 (异步转同步)
                    try:
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            corrected = loop.run_until_complete(
                                self._apply_llm_polish(text, role="corrector")
                            )
                        except RuntimeError:
                            corrected = asyncio.run(
                                self._apply_llm_polish(text, role="corrector")
                            )
                        if corrected and corrected != text:
                            logger.debug(f"Low confidence ({confidence:.2f}) LLM: {text!r} -> {corrected!r}")
                            sent['text'] = corrected
                            sent['correction_method'] = 'llm'
                    except Exception as e:
                        logger.warning(f"LLM correction failed for low confidence segment: {e}")

        return sentence_info

    @staticmethod
    def _dedupe_similars(similars: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """去重相似词候选，同一 (原词, 热词) 对只保留最高分"""
        seen: Dict[tuple, float] = {}
        for orig, hw, score in similars:
            key = (orig, hw)
            if key not in seen or score > seen[key]:
                seen[key] = score
        return [(k[0], k[1], v) for k, v in sorted(seen.items(), key=lambda x: -x[1])]

    async def _apply_llm_polish(
        self,
        text: str,
        role: str = "default",
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None,
        similarity_candidates: Optional[List[Tuple[str, str, float]]] = None,
    ) -> str:
        """应用 LLM 润色

        Args:
            text: 待润色文本
            role: LLM 角色
            prev_context: 前文上下文
            next_context: 后文上下文
            similarity_candidates: 相似词候选 [(原词, 热词, 分数), ...]
        """
        if not text:
            return text

        # 获取角色
        role_obj = get_role(role)

        # 构建提示词
        prompt_builder = PromptBuilder(system_prompt=role_obj.system_prompt)

        # 获取纠错历史上下文
        rectify_context = None
        if self._rectify_loaded:
            results = self.rectification_rag.search(text, top_k=3)
            if results:
                rectify_context = self.rectification_rag.format_prompt(results)

        # 构建消息
        messages = prompt_builder.build(
            user_content=text,
            hotwords=self._hotwords_list[:50] if self._hotwords_list else None,
            similarity_candidates=similarity_candidates,
            rectify_context=rectify_context,
            prev_context=prev_context,
            next_context=next_context,
            include_history=False
        )

        # 转换为 LLMMessage
        llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]

        # 调用 LLM
        result_parts = []
        async for chunk in self.llm_client.chat(llm_messages, stream=False):
            result_parts.append(chunk)

        polished = "".join(result_parts).strip()
        return polished if polished else text

    async def _apply_llm_fulltext_polish(
        self,
        text: str,
        max_chars: int = 2000,
        similarity_candidates: Optional[List[Tuple[str, str, float]]] = None,
    ) -> str:
        """应用 LLM 全文纠错

        使用专门的 corrector 角色对全文进行一次性纠错，
        利用完整上下文提升一致性。

        Args:
            text: 待纠错全文
            max_chars: 最大字符数限制
            similarity_candidates: 相似词候选 [(原词, 热词, 分数), ...]

        Returns:
            纠错后的文本
        """
        if not text:
            return text

        # 超长文本截断
        if len(text) > max_chars:
            logger.warning(f"Text too long for fulltext polish ({len(text)} > {max_chars}), truncating")
            text = text[:max_chars]

        # 使用 corrector 角色
        role_obj = get_role("corrector")
        prompt_builder = PromptBuilder(system_prompt=role_obj.system_prompt)

        # 获取纠错历史上下文
        rectify_context = None
        if self._rectify_loaded:
            results = self.rectification_rag.search(text[:200], top_k=5)
            if results:
                rectify_context = self.rectification_rag.format_prompt(results)

        # 构建消息
        messages = prompt_builder.build(
            user_content=role_obj.format_user_input(text),
            hotwords=self._hotwords_list[:50] if self._hotwords_list else None,
            similarity_candidates=similarity_candidates,
            rectify_context=rectify_context,
            include_history=False
        )

        # 转换为 LLMMessage
        llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]

        # 调用 LLM
        result_parts = []
        async for chunk in self.llm_client.chat(llm_messages, stream=False):
            result_parts.append(chunk)

        polished = "".join(result_parts).strip()
        if polished:
            logger.debug(f"Fulltext LLM correction applied ({len(text)} -> {len(polished)} chars)")
        return polished if polished else text

    async def _apply_llm_polish_with_context(
        self,
        sentences: List[Dict[str, Any]],
        role: str = "default",
        context_sentences: int = 1,
        similarity_candidates: Optional[List[Tuple[str, str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """对句子列表应用带上下文的 LLM 润色

        Args:
            sentences: 句子列表 [{"text": "...", ...}, ...]
            role: LLM 角色
            context_sentences: 上下文句子数
            similarity_candidates: 相似词候选 [(原词, 热词, 分数), ...]

        Returns:
            润色后的句子列表
        """
        if not sentences or context_sentences <= 0:
            # 不使用上下文，逐句处理
            for sent in sentences:
                if sent.get("text"):
                    sent["text"] = await self._apply_llm_polish(
                        sent["text"], role=role, similarity_candidates=similarity_candidates
                    )
            return sentences

        # 使用上下文处理
        for i, sent in enumerate(sentences):
            if not sent.get("text"):
                continue

            # 构建上下文
            prev_texts = []
            for j in range(max(0, i - context_sentences), i):
                if sentences[j].get("text"):
                    prev_texts.append(sentences[j]["text"])

            next_texts = []
            for j in range(i + 1, min(len(sentences), i + 1 + context_sentences)):
                if sentences[j].get("text"):
                    next_texts.append(sentences[j]["text"])

            prev_context = " ".join(prev_texts) if prev_texts else None
            next_context = " ".join(next_texts) if next_texts else None

            sent["text"] = await self._apply_llm_polish(
                sent["text"],
                role=role,
                prev_context=prev_context,
                next_context=next_context,
                similarity_candidates=similarity_candidates,
            )

        return sentences

    async def _apply_llm_batch_polish(
        self,
        sentences: List[Dict[str, Any]],
        role: str = "default",
        batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """批量 LLM 润色 - 将多个句子合并为一个请求

        将多个句子合并发送给 LLM，减少 API 调用次数，提高效率。

        Args:
            sentences: 句子列表 [{"text": "...", ...}, ...]
            role: LLM 角色
            batch_size: 每批处理的句子数

        Returns:
            润色后的句子列表
        """
        import re as re_module

        if not sentences:
            return sentences

        role_obj = get_role(role)

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            texts = [s.get('text', '') for s in batch if s.get('text')]

            if not texts:
                continue

            # 合并为编号列表
            combined = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))

            prompt_builder = PromptBuilder(system_prompt=role_obj.system_prompt)
            messages = prompt_builder.build(
                user_content=f"请润色以下语音识别结果，按编号返回：\n{combined}",
                hotwords=self._hotwords_list[:50] if self._hotwords_list else None,
                include_history=False
            )

            llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]

            result_parts = []
            async for chunk in self.llm_client.chat(llm_messages, stream=False):
                result_parts.append(chunk)

            polished = "".join(result_parts).strip()

            # 解析返回结果
            if polished:
                pattern = re_module.compile(r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)', re_module.DOTALL)
                matches = pattern.findall(polished)

                for num_str, text in matches:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(texts):
                        # 找到对应的原始句子并更新
                        for j, s in enumerate(batch):
                            if s.get('text') == texts[idx]:
                                sentences[i + j]['text'] = text.strip()
                                break

        return sentences

    def transcribe(
        self,
        audio_input: Union[bytes, str, Path],
        with_speaker: bool = False,
        apply_hotword: bool = True,
        apply_llm: bool = False,
        llm_role: str = "default",
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行转写

        Args:
            audio_input: 音频输入（文件路径或字节）
            with_speaker: 是否进行说话人识别
            apply_hotword: 是否应用热词纠错
            apply_llm: 是否应用 LLM 润色
            llm_role: LLM 角色（default/translator/code）
            hotwords: 自定义热词（覆盖已加载的热词）
            **kwargs: 其他参数传递给 ASR 模型

        Returns:
            转写结果字典
        """
        # 获取后端
        backend = model_manager.backend

        # 获取注入热词
        injection_hotwords = self._get_injection_hotwords(hotwords)

        # 检查说话人识别支持
        if with_speaker and not backend.supports_speaker:
            logger.warning(
                f"Backend {backend.get_info()['name']} does not support speaker diarization, "
                "falling back to PyTorch backend"
            )
            # 回退到 loader (PyTorch) 以支持说话人识别
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=injection_hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        else:
            # 使用配置的后端
            try:
                raw_result = backend.transcribe(
                    audio_input,
                    hotwords=injection_hotwords,
                    with_speaker=with_speaker,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"ASR transcription failed: {e}")
                raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 置信度过滤
        if settings.confidence_threshold > 0:
            sentence_info = self._filter_low_confidence(
                sentence_info, threshold=settings.confidence_threshold
            )

        # 热词纠错 - 收集相似词候选
        all_similars: List[Tuple[str, str, float]] = []
        if apply_hotword:
            text, similars = self._apply_corrections(text)
            all_similars.extend(similars)
            # 同时纠错每个句子的文本
            for sent in sentence_info:
                sent["text"], sent_similars = self._apply_corrections(sent.get("text", ""))
                all_similars.extend(sent_similars)

        # 去重相似词候选
        all_similars = self._dedupe_similars(all_similars)

        # LLM 润色 - 传入相似词候选
        if apply_llm:
            try:
                if settings.llm_fulltext_enable:
                    text = asyncio.get_event_loop().run_until_complete(
                        self._apply_llm_fulltext_polish(
                            text,
                            max_chars=settings.llm_fulltext_max_chars,
                            similarity_candidates=all_similars
                        )
                    )
                else:
                    text = asyncio.get_event_loop().run_until_complete(
                        self._apply_llm_polish(text, role=llm_role, similarity_candidates=all_similars)
                    )
            except RuntimeError:
                if settings.llm_fulltext_enable:
                    text = asyncio.run(
                        self._apply_llm_fulltext_polish(
                            text,
                            max_chars=settings.llm_fulltext_max_chars,
                            similarity_candidates=all_similars
                        )
                    )
                else:
                    text = asyncio.run(
                        self._apply_llm_polish(text, role=llm_role, similarity_candidates=all_similars)
                    )

        # 说话人标注
        if with_speaker and sentence_info:
            sentence_info = self.speaker_labeler.label_speakers(sentence_info)

        # 构建返回结果
        result = {
            "text": text,
            "sentences": [
                {
                    "text": s.get("text", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    **({"speaker": s.get("speaker"), "speaker_id": s.get("speaker_id")}
                       if with_speaker else {})
                }
                for s in sentence_info
            ],
            "raw_text": raw_result.get("text", ""),
        }

        # 生成格式化转写稿
        if with_speaker:
            result["transcript"] = self.speaker_labeler.format_transcript(
                result["sentences"],
                include_timestamp=True
            )

        return result

    async def transcribe_async(
        self,
        audio_input: Union[bytes, str, Path],
        with_speaker: bool = False,
        apply_hotword: bool = True,
        apply_llm: bool = False,
        llm_role: str = "default",
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步执行转写（适用于 FastAPI 异步端点）

        Args:
            同 transcribe()

        Returns:
            转写结果字典
        """
        # 获取后端
        backend = model_manager.backend

        # 获取注入热词
        injection_hotwords = self._get_injection_hotwords(hotwords)

        # 检查说话人识别支持
        if with_speaker and not backend.supports_speaker:
            logger.warning(
                f"Backend {backend.get_info()['name']} does not support speaker diarization, "
                "falling back to PyTorch backend"
            )
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=injection_hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        else:
            # 使用配置的后端
            try:
                raw_result = backend.transcribe(
                    audio_input,
                    hotwords=injection_hotwords,
                    with_speaker=with_speaker,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"ASR transcription failed: {e}")
                raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 置信度过滤
        if settings.confidence_threshold > 0:
            sentence_info = self._filter_low_confidence(
                sentence_info, threshold=settings.confidence_threshold
            )

        # 热词纠错 - 收集相似词候选
        all_similars: List[Tuple[str, str, float]] = []
        if apply_hotword:
            text, similars = self._apply_corrections(text)
            all_similars.extend(similars)
            for sent in sentence_info:
                sent["text"], sent_similars = self._apply_corrections(sent.get("text", ""))
                all_similars.extend(sent_similars)

        # 去重相似词候选
        all_similars = self._dedupe_similars(all_similars)

        # LLM 润色（异步）- 传入相似词候选
        if apply_llm:
            if settings.llm_fulltext_enable:
                text = await self._apply_llm_fulltext_polish(
                    text,
                    max_chars=settings.llm_fulltext_max_chars,
                    similarity_candidates=all_similars
                )
            else:
                text = await self._apply_llm_polish(
                    text, role=llm_role, similarity_candidates=all_similars
                )

        # 说话人标注
        if with_speaker and sentence_info:
            sentence_info = self.speaker_labeler.label_speakers(sentence_info)

        # 构建返回结果
        result = {
            "text": text,
            "sentences": [
                {
                    "text": s.get("text", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    **({"speaker": s.get("speaker"), "speaker_id": s.get("speaker_id")}
                       if with_speaker else {})
                }
                for s in sentence_info
            ],
            "raw_text": raw_result.get("text", ""),
        }

        if with_speaker:
            result["transcript"] = self.speaker_labeler.format_transcript(
                result["sentences"],
                include_timestamp=True
            )

        return result

    def transcribe_long_audio(
        self,
        audio_input: Union[bytes, str, Path, np.ndarray],
        with_speaker: bool = False,
        apply_hotword: bool = True,
        apply_llm: bool = False,
        llm_role: str = "default",
        hotwords: Optional[str] = None,
        max_workers: int = 2,
        sample_rate: int = 16000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        长音频智能分块转写

        使用 VAD 检测智能分割长音频，并行处理多个分块，
        最后合并结果并去除重叠。

        Args:
            audio_input: 音频输入（文件路径、字节或 numpy 数组）
            with_speaker: 是否进行说话人识别
            apply_hotword: 是否应用热词纠错
            apply_llm: 是否应用 LLM 润色
            llm_role: LLM 角色
            hotwords: 自定义热词
            max_workers: 并行处理线程数
            sample_rate: 采样率
            **kwargs: 其他参数

        Returns:
            转写结果字典
        """
        from src.core.audio.preprocessor import AudioPreprocessor

        # 加载音频为 numpy 数组
        if isinstance(audio_input, np.ndarray):
            audio = audio_input
        elif isinstance(audio_input, (str, Path)):
            preprocessor = AudioPreprocessor()
            audio, sr = preprocessor.load(str(audio_input))
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        elif isinstance(audio_input, bytes):
            import soundfile as sf
            import io
            audio, sr = sf.read(io.BytesIO(audio_input), dtype='float32')
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # 检查音频长度，短音频直接转写
        duration = len(audio) / sample_rate
        if duration <= self.audio_chunker.max_chunk_duration:
            logger.info(f"Audio is short ({duration:.1f}s), using direct transcription")
            return self.transcribe(
                audio_input,
                with_speaker=with_speaker,
                apply_hotword=apply_hotword,
                apply_llm=apply_llm,
                llm_role=llm_role,
                hotwords=hotwords,
                **kwargs
            )

        logger.info(f"Long audio detected ({duration:.1f}s), using chunked transcription")

        # 分割音频
        chunks = self.audio_chunker.split(audio, sample_rate)

        # 定义单块转写函数
        def transcribe_chunk(chunk_audio: np.ndarray) -> Dict[str, Any]:
            return self.transcribe(
                chunk_audio,
                with_speaker=with_speaker,
                apply_hotword=apply_hotword,
                apply_llm=False,  # LLM 在合并后统一处理
                hotwords=hotwords,
                **kwargs
            )

        # 并行处理分块
        chunk_results = self.audio_chunker.process_parallel(
            chunks, transcribe_chunk, max_workers=max_workers
        )

        # 合并结果
        merged = self.audio_chunker.merge_results(chunk_results, sample_rate)

        text = merged.get("text", "")
        sentences = merged.get("sentences", [])

        # 对合并后的文本进行热词检索，收集相似词候选
        all_similars: List[Tuple[str, str, float]] = []
        if apply_hotword and text and self._hotwords_loaded:
            # 对合并后的全文进行热词匹配，收集相似词候选
            correction = self.corrector.correct(text)
            all_similars.extend(correction.similars)
            all_similars = self._dedupe_similars(all_similars)

        # 全文 LLM 润色 - 传入相似词候选
        if apply_llm and text:
            try:
                if settings.llm_fulltext_enable:
                    text = asyncio.get_event_loop().run_until_complete(
                        self._apply_llm_fulltext_polish(
                            text,
                            max_chars=settings.llm_fulltext_max_chars,
                            similarity_candidates=all_similars
                        )
                    )
                else:
                    text = asyncio.get_event_loop().run_until_complete(
                        self._apply_llm_polish(text, role=llm_role, similarity_candidates=all_similars)
                    )
            except RuntimeError:
                if settings.llm_fulltext_enable:
                    text = asyncio.run(
                        self._apply_llm_fulltext_polish(
                            text,
                            max_chars=settings.llm_fulltext_max_chars,
                            similarity_candidates=all_similars
                        )
                    )
                else:
                    text = asyncio.run(
                        self._apply_llm_polish(text, role=llm_role, similarity_candidates=all_similars)
                    )

        # 说话人标注
        if with_speaker and sentences:
            sentences = self.speaker_labeler.label_speakers(sentences)

        result = {
            "text": text,
            "sentences": sentences,
            "duration": duration,
            "chunks": len(chunks),
        }

        if with_speaker:
            result["transcript"] = self.speaker_labeler.format_transcript(
                sentences,
                include_timestamp=True
            )

        logger.info(f"Long audio transcription completed: {len(chunks)} chunks, {duration:.1f}s")
        return result

    def transcribe_streaming(
        self,
        audio_chunk: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """流式转写 (单个音频块)

        注意: 流式转写仅支持 PyTorch 后端。
        其他后端会自动回退到 PyTorch。
        """
        backend = model_manager.backend

        # 获取注入热词
        injection_hotwords = self._get_injection_hotwords(hotwords)

        # 检查流式支持
        if not backend.supports_streaming:
            logger.debug(
                f"Backend {backend.get_info()['name']} does not support streaming, "
                "using PyTorch backend for streaming"
            )
            # 使用 PyTorch 后端的流式功能
            return model_manager.loader._backend.transcribe_streaming(
                audio_chunk,
                cache,
                is_final=is_final,
                hotwords=injection_hotwords,
                **kwargs
            )

        # 使用后端的流式转写
        result = backend.transcribe_streaming(
            audio_chunk,
            cache,
            is_final=is_final,
            hotwords=injection_hotwords,
            **kwargs
        )

        # 应用纠错
        if result.get("text"):
            result["text"], _ = self._apply_corrections(result["text"])

        return result


# 全局引擎实例
transcription_engine = TranscriptionEngine()
