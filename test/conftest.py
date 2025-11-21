import os
from pathlib import Path
import pytest
import tempfile
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch, MagicMock
import wave


MINIMAL_FILE_CONTENT = {
    'pdf': b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>
endobj
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000120 00000 n
trailer
<< /Root 1 0 R /Size 4 >>
startxref
180
%%EOF
""",
    'txt': b'lorem ipsum'
}

DUMMY_BACKUP_DATA = [
        {
            "voice_id": "voice_id_1",
            "text": "Tiny is a ninjacat",
            "filepath": "my_fake_filepath1",
            "request_id": "123"
        },
        {
            "voice_id": "voice_id_2",
            "text": "Abby is a chungus",
            "filepath": "my_fake_filepath2",
            "request_id": "456"
        }
    ]

class DummyTextResponse:
	def __init__(self, text):
		self.text = text

class DummyAudioResponse:
	def __init__(self, audio_data):
		self.candidates = [
			SimpleNamespace(
				content=SimpleNamespace(
					parts=[
						SimpleNamespace(
							inline_data=SimpleNamespace(
								data=audio_data
							)
						)
					]
				)
			)
		]

@pytest.fixture
def dummy_file():
    """
    Fixture that returns a function to generate a temporary dummy file of a given size.
    The file is automatically deleted after the test finishes.

    Usage:
        def test_something(dummy_file):
            path = dummy_file('pdf', 2 * 1024 * 1024)  # 2 MB
            assert os.path.getsize(path) >= 2 * 1024 * 1024
    """
    temp_files = []

    def _make_dummy_file(
            filetype: Literal['pdf', 'txt'],
            min_bytes: int = 0
        ) -> str:
        fd, path = tempfile.mkstemp(suffix=f".{filetype}")
        os.close(fd)
        temp_files.append(path)

        # Minimal content
        content = MINIMAL_FILE_CONTENT.get(filetype, 'pdf')

        # Pad with null bytes if necessary
        if len(content) < min_bytes:
            content += b"\0" * (min_bytes - len(content))

        with open(path, "wb") as f:
            f.write(content)

        return Path(path)

    yield _make_dummy_file

    # Cleanup after test
    for path in temp_files:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

@pytest.fixture
def output_dir():
    tmpdir_obj = tempfile.TemporaryDirectory()
    output_filepath = tmpdir_obj.name
    yield Path(output_filepath)
    tmpdir_obj.cleanup()

@pytest.fixture
def genai_client_mock():
    mock_client = MagicMock()

    def count_tokens_mock(text: str) -> int:
        mock_resp = MagicMock()
        mock_resp.total_tokens = max(1, len(text) // 4) # 1 token minimum, 4 chars per token
        return mock_resp

    def set_count_tokens():
        mock_client.models.count_tokens.side_effect = lambda model, contents: count_tokens_mock(contents)

    def set_expected_response(expected_response, has_sys_instrux = False, is_audio_generation = False):
        if not has_sys_instrux and not is_audio_generation:
            mock_client.models.generate_content.side_effect = (
                lambda model, contents: DummyTextResponse(expected_response)
            )
        elif has_sys_instrux and not is_audio_generation:
            mock_client.models.generate_content.side_effect = (
                lambda model, config, contents: DummyTextResponse(expected_response)
            )
        elif is_audio_generation:
            mock_client.models.generate_content.side_effect = (
                lambda model, config, contents: DummyAudioResponse(expected_response)
            )

    with patch("main.genai.Client", return_value=mock_client):
        yield mock_client, set_count_tokens, set_expected_response
        
@pytest.fixture
def wav_file_data():
    def _get_wav_file_data(input_filepath: str):
        with wave.open(input_filepath, "rb") as wf:
            n_frames = wf.getnframes()       # number of frames
            n_channels = wf.getnchannels()   # number of channels
            framerate = wf.getframerate() # sample rate (samples per second)
            sampwidth = wf.getsampwidth()    # bytes per sample
            audio_data_size = n_frames * n_channels * sampwidth # audio data size in bytes
            audio_data = wf.readframes(n_frames)  # read all audio frames
        return {
            "num_frames": n_frames,
            "num_channels": n_channels,
            "framerate": framerate,
            "sample_width": sampwidth,
            "audio_data_size_in_bytes": audio_data_size,
            "audio_data": audio_data
        }
    yield _get_wav_file_data 

class ElevenLabs:
    # A simplified stand-in for the ElevenLabs client
    pass

class VoiceSettings:
     pass

@pytest.fixture
def mock_elevenlabs_client():
    """Fixture to provide a mocked ElevenLabs client."""
    return MagicMock()

@pytest.fixture
def mock_voice_settings():
     return MagicMock()