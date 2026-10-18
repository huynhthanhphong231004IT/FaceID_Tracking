class ModelLoadError(RuntimeError):
    """Ném ra khi không thể tải/khởi tạo một model (thiếu file, thiếu mạng, thiếu thư viện...)."""
