"""核心转写引擎"""
import asyncio
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.config import settings
from src.models.model_manager import model_manager
from src.core.hotword import PhonemeCorrector
from src.core.hotword.rule_corrector import RuleCorrector
from src.core.hotword.rectification import RectificationRAG
from src.core.speaker import SpeakerLabeler
from src.core.llm import LLMClient, LLMMessage, PromptBuilder
from src.core.llm.roles import get_role
from src.core.text_processor import TextPostProcessor

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """转写引擎 - 整合 ASR + 热词纠错 + 说话人识别 + LLM润色"""

    def __init__(self):
        self.corrector = PhonemeCorrector(
            threshold=settings.hotwords_threshold,
            similar_threshold=settings.hotwords_threshold - 0.2
        )
        self.rule_corrector = RuleCorrector()
        self.rectification_rag = RectificationRAG()
        self.speaker_labeler = SpeakerLabeler()

        # 文本后处理器
        self.post_processor = TextPostProcessor.from_config(settings)

        # LLM 组件
        self._llm_client: Optional[LLMClient] = None
        self._prompt_builder: Optional[PromptBuilder] = None
        self._hotwords_list: List[str] = []

        self._hotwords_loaded = False
        self._rules_loaded = False
        self._rectify_loaded = False

    @property
    def llm_client(self) -> LLMClient:
        """懒加载 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient(
                base_url=settings.llm_base_url,
                model=settings.llm_model
            )
        return self._llm_client

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

    def _apply_corrections(self, text: str) -> str:
        """应用所有纠错（热词 + 规则 + 文本后处理）"""
        # 热词纠错
        if self._hotwords_loaded and text:
            correction = self.corrector.correct(text)
            text = correction.text

        # 规则纠错
        if self._rules_loaded and text:
            text = self.rule_corrector.substitute(text)

        # 文本后处理 (ITN、繁简转换、标点转换)
        text = self.post_processor.process(text)

        return text

    async def _apply_llm_polish(
        self,
        text: str,
        role: str = "default"
    ) -> str:
        """应用 LLM 润色"""
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
        return polished if polished else text

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
        # 执行 ASR
        try:
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 热词纠错
        if apply_hotword:
            text = self._apply_corrections(text)
            # 同时纠错每个句子的文本
            for sent in sentence_info:
                sent["text"] = self._apply_corrections(sent.get("text", ""))

        # LLM 润色
        if apply_llm:
            try:
                text = asyncio.get_event_loop().run_until_complete(
                    self._apply_llm_polish(text, role=llm_role)
                )
            except RuntimeError:
                # 没有运行中的事件循环，创建新的
                text = asyncio.run(self._apply_llm_polish(text, role=llm_role))

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
        # 执行 ASR（同步，因为 FunASR 不支持异步）
        try:
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 热词纠错
        if apply_hotword:
            text = self._apply_corrections(text)
            for sent in sentence_info:
                sent["text"] = self._apply_corrections(sent.get("text", ""))

        # LLM 润色（异步）
        if apply_llm:
            text = await self._apply_llm_polish(text, role=llm_role)

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

    def transcribe_streaming(
        self,
        audio_chunk: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """流式转写 (单个音频块)"""
        online_model = model_manager.loader.asr_model_online

        result = online_model.generate(
            input=audio_chunk,
            cache=cache.get("asr_cache", {}),
            is_final=is_final,
            **kwargs
        )

        if result:
            cache["asr_cache"] = result[0].get("cache", {})
            text = result[0].get("text", "")

            # 应用纠错
            text = self._apply_corrections(text)

            return {"text": text, "is_final": is_final}

        return {"text": "", "is_final": is_final}


# 全局引擎实例
transcription_engine = TranscriptionEngine()
