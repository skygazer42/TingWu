from unittest.mock import MagicMock, patch


def test_model_loader_initialization():
    """ASRModelLoader is a thin compatibility wrapper around PyTorchBackend."""
    with patch("src.models.asr_loader.PyTorchBackend") as mock_backend_cls:
        mock_backend = MagicMock()
        mock_backend_cls.return_value = mock_backend

        from src.models.asr_loader import ASRModelLoader

        loader = ASRModelLoader(device="cpu", ngpu=0)
        assert loader is not None

        # Verify key args are forwarded to backend ctor.
        called_kwargs = mock_backend_cls.call_args.kwargs
        assert called_kwargs["device"] == "cpu"
        assert called_kwargs["ngpu"] == 0


def test_transcribe_forwards_to_backend():
    """ASRModelLoader.transcribe should forward to backend.transcribe."""
    with patch("src.models.asr_loader.PyTorchBackend") as mock_backend_cls:
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = {"text": "你好", "sentence_info": []}
        mock_backend_cls.return_value = mock_backend

        from src.models.asr_loader import ASRModelLoader

        loader = ASRModelLoader(device="cpu", ngpu=0)
        result = loader.transcribe(b"fake_audio", hotwords="Claude", with_speaker=True)

        assert result["text"] == "你好"
        mock_backend.transcribe.assert_called_once_with(
            b"fake_audio",
            hotwords="Claude",
            with_speaker=True,
        )


def test_model_manager_singleton():
    """测试 ModelManager 单例模式"""
    from src.models.model_manager import ModelManager

    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2

