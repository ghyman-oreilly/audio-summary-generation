import pytest
from unittest.mock import MagicMock, patch, ANY, call
from pathlib import Path
import re

# Assuming the functions and constants are imported from a module named 'main'
# in your project structure, as suggested by your patches.
# You must ensure 'from main import _generate_audio_summary' works in your environment.
from main import _generate_audio_summary
from prompts import TEXT_SUMMARY_PROMPT, TRANSCRIPT_SYS_INSTRUCTIONS


# --- Mock Data and Constants ---
MOCK_API_KEY = "mock_api_key"
MOCK_TEXT_SUMMARY = "This is a mock summary of the PDF content."
MOCK_TRANSCRIPT = "SPEAKER 1: First part. \nSPEAKER 2: Second part."
MOCK_REQUEST_ID = "mock_request_id" # This is returned by the mock wrapper
MOCK_TIMESTAMP = 1234567890

# --- Pytest Fixtures ---

@pytest.fixture
def mock_path_to_pdf(tmp_path):
    """Fixture to create a mock PDF file."""
    pdf_file = tmp_path / "test_document.pdf"
    pdf_file.write_text("Mock PDF content for size > 0")
    return pdf_file

@pytest.fixture
def mock_output_dir(tmp_path):
    """Fixture to create a mock output directory."""
    return tmp_path / "output"

def assert_call_with_regex_path(mock_func, text_content, path_regex):
    """
    Asserts that the mock function was called with the exact text_content
    and a Path object that matches the path_regex. This replaces the complex
    unittest.mock assertion logic for arguments containing comparison objects.
    """
    for call_args, call_kwargs in mock_func.call_args_list:
        # Check if the text content matches and the path is the second positional argument
        if call_args and len(call_args) == 2 and call_args[0] == text_content:
            path_arg = call_args[1]
            if isinstance(path_arg, Path) and re.match(path_regex, str(path_arg)):
                return True
    
    raise AssertionError(
        f"Call not found for text: '{text_content}' and path regex: '{path_regex}'"
    )

@patch("main.combine_wav_files")
@patch("main.delete_files")
@patch("main.write_backup_to_json_file")
@patch("main.write_audio_data_to_wav_file")
@patch("main.generate_audio_with_timeout", return_value=MOCK_REQUEST_ID)
@patch("main.create_speaker_text_chunks", return_value=[['Chunk 1'], ['Chunk 2']])
@patch("main.write_text_to_file")
@patch("main.generate_text", return_value=MOCK_TRANSCRIPT)
@patch("main.infer_with_pdf_document_understanding", return_value=MOCK_TEXT_SUMMARY)
@patch("main.file_is_valid", return_value=True)
@patch("main.dir_is_valid", return_value=True)
@patch("main.check_api_key", return_value=MOCK_API_KEY)
@patch("main.validate_voices")
@patch("main.typer.confirm")
@patch("main.Path.cwd")
@patch("main.time.time", return_value=MOCK_TIMESTAMP)
@patch("main.ElevenLabs")
@patch("main.generate_voice_settings")
def test_generate_audio_summary_pdf_only_e2e(
    mock_generate_voice_settings,
    mock_elevenlabs_constructor,
    mock_time,
    mock_cwd,
    mock_confirm,
    mock_validate_voices,
    mock_check_api_key,
    mock_dir_is_valid,
    mock_file_is_valid,
    mock_infer_with_pdf_document_understanding,
    mock_generate_text,
    mock_write_text_to_file,
    mock_create_speaker_text_chunks,
    mock_generate_audio_with_timeout,
    mock_write_audio_data_to_wav_file,
    mock_write_backup_to_json_file,
    mock_delete_files,
    mock_combine_wav_files,
    mock_output_dir,
    mock_path_to_pdf,
    mock_elevenlabs_client,
    mock_voice_settings
):
    """
    E2E test for _generate_audio_summary when only path_to_pdf and output_dir are provided.
    This simulates the full workflow from PDF -> Summary -> Transcript -> Audio Chunks -> Combined Audio.
    """
    
    EXPECTED_PATH_000 = Path(mock_output_dir / f"audio_chunk_000_{MOCK_TIMESTAMP}.wav")
    EXPECTED_PATH_001 = Path(mock_output_dir / f"audio_chunk_001_{MOCK_TIMESTAMP}.wav")

    # 1. Setup Mock User Interaction:
    # [0] Summary to transcript? (True)
    # [1] Transcript to audio? (True)
    # [2] Save partial audio chunks? (False, to ensure delete_files is called)
    mock_confirm.side_effect = [True, True, False]
    mock_cwd.return_value = mock_output_dir.parent

    mock_elevenlabs_constructor.return_value = mock_elevenlabs_client
    mock_generate_voice_settings.return_value = mock_voice_settings

    # 2. Execute the function under test:
    _generate_audio_summary(
        path_to_pdf=mock_path_to_pdf,
        output_dir=mock_output_dir,
        text_summary_file=None,
        transcript_file=None,
        backup_file_for_regen=None,
        speaker_one_voice="VOICE_ONE",
        speaker_two_voice="VOICE_TWO"
    )
   
    # 3. Assertions (Verifying the Execution Flow):
    
    ## API Key and Voice Validation
    mock_validate_voices.assert_called_once_with(
        ANY,
        "VOICE_ONE",
        "VOICE_TWO"
    )

    ## PDF Summarization Workflow
    mock_infer_with_pdf_document_understanding.assert_called_once()
    
    # Check that the summary was written to a file
    assert_call_with_regex_path(
        mock_write_text_to_file,
        MOCK_TEXT_SUMMARY,
        str(mock_output_dir) + r'/text_summary_\d+\.txt'
    )

    ## Transcript Generation Workflow
    mock_generate_text.assert_called_once_with(
        MOCK_TEXT_SUMMARY,
        MOCK_API_KEY,
        TRANSCRIPT_SYS_INSTRUCTIONS,
        'gemini-2.5-pro'
    )
    
    # Check that the transcript was written to a file
    assert_call_with_regex_path(
        mock_write_text_to_file,
        MOCK_TRANSCRIPT,
        str(mock_output_dir) + r'/transcript_\d+\.txt'
    )

    ## Audio Generation Workflow
    mock_create_speaker_text_chunks.assert_called_once_with(MOCK_TRANSCRIPT)
    
    # Should generate audio for 2 chunks (from MOCK_TRANSCRIPT)
    assert mock_generate_audio_with_timeout.call_count == 2
    
    # Check the first chunk generation call
    mock_generate_audio_with_timeout.assert_any_call(
        text='Chunk 1', 
        voice_id='VOICE_ONE', 
        output_file=EXPECTED_PATH_000,
        tts_client=mock_elevenlabs_client, 
        voice_settings=mock_voice_settings,
        model_id='eleven_multilingual_v2', 
        previous_request_ids=[] # note: no previous request ID
    )
    
    # Check the second chunk generation call
    mock_generate_audio_with_timeout.assert_any_call(
        text='Chunk 2', 
        voice_id='VOICE_TWO', 
        output_file=EXPECTED_PATH_001,
        tts_client=mock_elevenlabs_client, 
        voice_settings=mock_voice_settings,
        model_id='eleven_multilingual_v2', 
        previous_request_ids=[MOCK_REQUEST_ID]
    )

    # Check that the backup file was written (2 items, 2 requests)
    mock_write_backup_to_json_file.assert_called_once()
    
    ## Final Steps
    mock_combine_wav_files.assert_called_once()
    
    # Check that delete_files was called because mock_confirm[2] was False
    mock_delete_files.assert_called_once()


def test_generate_audio_summary_provide_summary():
    # TODO: write test
    pass

def test_generate_audio_summary_provide_transcript():
    # TODO: write test
    pass

def test_generate_audio_summary_select_voices():
    # TODO: write test
    pass

def test_generate_audio_summary_provide_backup_file():
    # TODO: write test
    pass

def test_combine_audio_files():
    # TODO: write test
    pass

def test_add_elevenlabs_voice():
    # TODO: write test
    pass