"""HTTP 客户端示例"""
import requests
import sys

API_BASE = "http://localhost:8000"


def transcribe_file(file_path: str, with_speaker: bool = False):
    """上传音频文件进行转写"""
    url = f"{API_BASE}/api/v1/transcribe"

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        data = {
            "with_speaker": with_speaker,
            "apply_hotword": True,
        }

        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"转写结果: {result['text']}")
        print()

        if with_speaker and result.get("transcript"):
            print("带说话人的转写稿:")
            print(result["transcript"])

        return result
    else:
        print(f"转写失败: {response.text}")
        return None


def update_hotwords(hotwords: list):
    """更新热词"""
    url = f"{API_BASE}/api/v1/hotwords"
    response = requests.post(url, json={"hotwords": hotwords})

    if response.status_code == 200:
        print(f"热词更新成功: {response.json()}")
    else:
        print(f"热词更新失败: {response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python client_http.py <音频文件路径> [--speaker]")
        sys.exit(1)

    file_path = sys.argv[1]
    with_speaker = "--speaker" in sys.argv

    transcribe_file(file_path, with_speaker)
