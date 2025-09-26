import pytest
from unittest.mock import MagicMock, patch

from conftest import DummyAudioResponse, DummyTextResponse
from main import DEFAULT_SPEAKER_ONE_LABEL, DEFAULT_SPEAKER_TWO_LABEL, generate_audio_summary

class DummyTokenCountResponse:
    def __init__(self, total_tokens):
	    self.total_tokens = total_tokens

@pytest.fixture
def input_pdf(dummy_file):
    yield dummy_file('pdf', 15 * 1024 * 1024)

@pytest.fixture
def check_api_mock():
    with patch('main.check_api_key', return_value="my_api_key") as check_api_mock:
        yield check_api_mock

@pytest.fixture
def typer_confirm_mock():
    with patch('main.typer.confirm', return_value=True) as typer_confirm_mock:
        yield typer_confirm_mock

def test_generate_audio_summary(
    input_pdf,
    output_dir,
    check_api_mock,
    typer_confirm_mock,
    wav_file_data	
):
    """
    E2E starting from PDF, with defaults for all
    options
    """
    mock_client = MagicMock()
    mock_client.models.generate_content = MagicMock()
    mock_client.models.count_tokens = MagicMock()

    test_input_wav_filepath = 'test/test_data/hello.wav'
    text_summary = "This is my text summary"
    transcript_text = (
        f"{DEFAULT_SPEAKER_ONE_LABEL}: That's a great text right there.\n"
        f"{DEFAULT_SPEAKER_TWO_LABEL}: It sure is!"
    )
    input_audio_data = wav_file_data(test_input_wav_filepath).get('audio_data')

    mock_client.models.count_tokens.return_value = DummyTokenCountResponse(25)

    # side effect in sequence of call #
    mock_client.models.generate_content.side_effect = [
        DummyTextResponse(text_summary),
        DummyTextResponse(transcript_text),
        DummyAudioResponse(input_audio_data)
    ]

    with patch("main.genai.Client", return_value=mock_client):
        generate_audio_summary(input_pdf, output_dir)
        summary_files = list(output_dir.glob("text_summary_*.txt"))
        transcript_files = list(output_dir.glob("transcript_*.txt"))
        audio_files = list(output_dir.glob("combined_audio_*.wav"))

        assert len(summary_files) == 1
        assert len(transcript_files) == 1
        assert len(audio_files) == 1

        with open(str(summary_files[0]), 'r') as f:
            assert f.read() == text_summary
        with open(str(transcript_files[0]), 'r') as f:
            assert f.read() == transcript_text
        
        assert wav_file_data(str(audio_files[0])).get('audio_data') == input_audio_data
