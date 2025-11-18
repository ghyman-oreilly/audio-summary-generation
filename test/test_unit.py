import os
from pathlib import Path 
import pytest
import random
import tempfile
import typer
from unittest.mock import MagicMock, patch
import wave

from conftest import MINIMAL_FILE_CONTENT
from main import (
    check_api_key,
    create_speaker_text_chunks,
    combine_wav_files,
    delete_files,
    dir_is_valid,
    execute_pdf_workflow, 
    execute_transcript_generation_workflow,
    file_is_valid,
    generate_audio_with_timeout,
    generate_audio_segments,
    generate_text,
    infer_with_pdf_document_understanding,
    read_text_from_file,
    validate_voices,
    write_audio_data_to_wav_file,
    write_text_to_file
)
from prompts import TRANSCRIPT_SYS_INSTRUCTIONS


@pytest.fixture
def audio_output_filepath():
    fd, output_filepath = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    yield output_filepath
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

@pytest.fixture
def api_key():
    yield "my_api_key"

class ElevenLabs:
    # A simplified stand-in for the ElevenLabs client
    def __init__(self):
        self.voices = MagicMock()

@pytest.fixture
def mock_voice_client():
    """Fixture to provide a mocked ElevenLabs client."""
    client = MagicMock(spec=ElevenLabs)
    # Ensure client.voices.get is a mock object
    client.voices.get = MagicMock()
    return client

@pytest.mark.parametrize(
    "path_to_dir, expected",
    [
        pytest.param(Path.cwd(), True, id="path-is-valid"),
        pytest.param(Path('This is a made-up path'), False, id="path-is-invalid")
    ],
)
def test_dir_is_valid(path_to_dir, expected):
    assert dir_is_valid(path_to_dir) == expected

@pytest.mark.parametrize(
    "filetype, size, expected_suffix, max_size_mb, expected",
    [
        pytest.param('pdf', 24 * 1024 * 1024, ".pdf", 25, True, id="pdf-size-is-okay"),
        pytest.param('pdf', 30 * 1024 * 1024, ".pdf", 25, False, id="pdf-size-too-big"),
        pytest.param('txt', 2 * 1024 * 1024, ".pdf", 25, False, id="txt-wrong-filetype"),
    ],
)
def test_file_is_valid(dummy_file, filetype, size, expected_suffix, max_size_mb, expected):
    path_to_file = dummy_file(filetype, size)
    assert file_is_valid(path_to_file, expected_suffix, max_size_mb) == expected

def test_infer_with_pdf_document_understanding(dummy_file, genai_client_mock, api_key):
    """
    `infer_with_pdf_document_understanding` is a thin wrapper
    around API calls, so we're just spot-checking the signature
    """
    _, _, set_expected_response = genai_client_mock
    
    pdf_filepath = dummy_file('pdf')
    prompt = "Please summarize this document."
    expected_response = "That's a great document!"
    set_expected_response(expected_response)

    response = infer_with_pdf_document_understanding(pdf_filepath, prompt, api_key)

    assert response == expected_response

@pytest.mark.parametrize(
    "sys_instrux, expected_response",
    [
        pytest.param("You are a helpful AI", "I'm a helpful AI!", id="includes-sys-instrux"),
        pytest.param(None, "I'm a loser baby!", id="no-sys-instrux"),
    ],
)
def test_generate_text(sys_instrux, expected_response, genai_client_mock, api_key):
    """
    Another test for a thin wrapper around 
    genai service calls
    """
    _, _, set_expected_response = genai_client_mock
    user_prompt = "Tell me about yourself."
    set_expected_response(expected_response, has_sys_instrux=bool(sys_instrux))
    response = generate_text(user_prompt, api_key, sys_instrux)
    assert response == expected_response

def test_write_text_to_file(dummy_file):
    path_to_file = str(dummy_file('txt'))
    file_content = "This is my file content"
    write_text_to_file(file_content, path_to_file)
    with open(path_to_file, 'r') as f:
        assert f.read() == file_content

def test_read_text_from_file(dummy_file):
    path_to_file = str(dummy_file('txt'))
    assert read_text_from_file(path_to_file) == MINIMAL_FILE_CONTENT.get('txt').decode()

# Mock implementation of the helper function (must be defined in the same scope for patching)
# NOTE: In a real environment, you'd patch where the function is imported (e.g., 'your_module.chunk_segment_by_sentences')
def chunk_segment_by_sentences(line, char_limit):
    # This mock always splits a long line into three smaller chunks
    return [
        f"Chunk 1 of {len(line)}",
        f"Chunk 2 of {len(line)}",
        f"Chunk 3 of {len(line)}",
    ]

@patch('main.chunk_segment_by_sentences', side_effect=chunk_segment_by_sentences)
@pytest.mark.parametrize(
    "text_input, char_limit, expected_speaker_turns, expected_total_chunks",
    [
        pytest.param(
            # Input: 2 short lines, 1 long line (will be split by mock)
            "Short line 1.\nShort line 2.\nThis is a very long line that will definitely exceed the limit.",
            50, # A small limit to force the third line to split
            3, # Expected speaker turns (3 lines in input)
            5, # Expected total chunks: 1 (line 1) + 1 (line 2) + 3 (mock split line 3) = 5
            id="mixed_split_small_limit"
        ),
        pytest.param(
            # Input: Same 3 lines
            "Short line 1.\nShort line 2.\nThis is a very long line that will definitely exceed the limit.",
            5000, # A very large limit, nothing should split
            3, # Expected speaker turns (3 lines)
            3, # Expected total chunks: 1 (line 1) + 1 (line 2) + 1 (line 3, not split) = 3
            id="no_split_large_limit"
        ),
        pytest.param(
            # Input: 4 short lines, one is empty (should be ignored)
            "Line A\nLine B\n\nLine C\n",
            100,
            3, # Expected speaker turns (A, B, C)
            3, # Expected total chunks (1 for each line)
            id="filter_empty_lines"
        ),
    ],
)
def test_create_speaker_text_chunks(
    mock_split_func, # The mocked function (must be the first argument after patching)
    text_input,
    char_limit,
    expected_speaker_turns,
    expected_total_chunks
):
    """Tests the new structure and splitting logic of create_speaker_text_chunks."""

    chunked_strings = create_speaker_text_chunks(text_input, char_limit)
  
    # Assert the number of speaker turns (outer list length)
    assert len(chunked_strings) == expected_speaker_turns, \
        "The number of speaker turns (lines) does not match expected count."

    # Assert the total number of audio chunks (sum of inner list lengths)
    total_chunks = sum(len(sublist) for sublist in chunked_strings)
    assert total_chunks == expected_total_chunks, \
        "The total number of resulting audio chunks does not match expected count."

    # Assert the structure and content quality
    for speaker_turn in chunked_strings:
        # Each speaker turn must be a list
        assert isinstance(speaker_turn, list), \
            "Each speaker turn must be a list (the sub-list)."
        
        for chunk in speaker_turn:
            # Each chunk must be a non-empty string
            assert isinstance(chunk, str)
            assert len(chunk) > 0
            
    # Check if the mock was called the correct number of times (only in "mixed_split_small_limit")
    if expected_total_chunks > expected_speaker_turns:
        mock_split_func.assert_called_once()
    else:
        # For tests where no splitting occurs, the mock shouldn't be called
        mock_split_func.assert_not_called()

def test_generate_audio_segments(output_dir, api_key):
    text_chunks = ["hello", "world"]
    timestamp = 123456
    expected_filepath_one = Path(output_dir / f'audio_chunk_000_{timestamp}.wav')
    expected_filepath_two = Path(output_dir / f'audio_chunk_001_{timestamp}.wav')
    with patch('main.generate_audio_with_timeout', return_value=None):
        audio_chunk_filepaths = generate_audio_segments(text_chunks, timestamp, output_dir, api_key)
        assert len(audio_chunk_filepaths) == 2
        assert audio_chunk_filepaths[0] == expected_filepath_one
        assert audio_chunk_filepaths[1] == expected_filepath_two

def test_generate_audio_with_timeout(audio_output_filepath, wav_file_data, genai_client_mock, api_key):
    _, _, set_expected_response = genai_client_mock
    text = "Hello"
    output_filepath = audio_output_filepath
    input_filepath = 'test/test_data/chunk_audio_00.wav'
    expected_response = wav_file_data(input_filepath).get('audio_data')
    set_expected_response(expected_response, is_audio_generation=True)
    generate_audio_with_timeout(text, output_filepath, api_key)
    output_wav_data = wav_file_data(output_filepath).get('audio_data')
    assert expected_response == output_wav_data

def test_write_audio_data_to_wav_file(wav_file_data, audio_output_filepath):  
    input_filepath = 'test/test_data/chunk_audio_00.wav'
    output_filepath = audio_output_filepath

    input_wav_data = wav_file_data(input_filepath)

    write_audio_data_to_wav_file(
        Path(output_filepath), 
        input_wav_data.get("audio_data"), 
        input_wav_data.get("num_channels"), 
        input_wav_data.get("framerate"), 
        input_wav_data.get("sample_width")
    )
    
    output_wav_data = wav_file_data(output_filepath)

    input_audio_data_size_in_bytes = input_wav_data.get("audio_data_size_in_bytes")
    output_audio_data_size_in_bytes = output_wav_data.get("audio_data_size_in_bytes")

    assert input_audio_data_size_in_bytes > 0 and output_audio_data_size_in_bytes > 0
    assert input_audio_data_size_in_bytes == output_audio_data_size_in_bytes
    
    os.remove(output_filepath)  

@pytest.mark.parametrize(
    "passed_silence_duration",
    [
        pytest.param(None, id="silence-duration-not-passed"),
        pytest.param(0.5, id="silence-duration-passed"),
    ],
)
def test_combine_wav_files_with_silence(passed_silence_duration, wav_file_data, audio_output_filepath):
    """
    Test combining WAV files, including the calculation for added silence.
    """
    output_filepath = audio_output_filepath
    
    input_filepaths = [f'test/test_data/chunk_audio_0{d}.wav' for d in range(0, 3)]
    num_input_files = len(input_filepaths)
    
    if not passed_silence_duration:
        silence_duration_sec = 0.4 # Matches the function's default
    else:
        silence_duration_sec = passed_silence_duration

    # 2. Calculate the total size of input audio data
    input_audio_data_size_in_bytes = sum(
        wav_file_data(f).get("audio_data_size_in_bytes") for f in input_filepaths
    )

    # Get parameters from the first file to calculate the expected silence size
    first_filepath = input_filepaths[0]
    
    # Use standard wave library to peek at file parameters
    with wave.open(first_filepath, 'rb') as w:
        n_channels, sample_width, frame_rate = w.getparams()[:3]
        frame_size = n_channels * sample_width

    # Calculate the expected total size of silence data
    # Silence is added (N - 1) times, where N is the number of input files.
    num_silence_sections = num_input_files - 1 
    silence_frames = int(frame_rate * silence_duration_sec)
    
    expected_silence_data_size_in_bytes = num_silence_sections * silence_frames * frame_size

    expected_output_data_size_in_bytes = (
        input_audio_data_size_in_bytes + expected_silence_data_size_in_bytes
    )

    if not passed_silence_duration:
        combine_wav_files([Path(f) for f in input_filepaths], output_filepath)
    else:
        combine_wav_files([Path(f) for f in input_filepaths], output_filepath, passed_silence_duration)

    output_wav_data = wav_file_data(output_filepath)
    output_audio_data_size_in_bytes = output_wav_data.get("audio_data_size_in_bytes")

    assert input_audio_data_size_in_bytes > 0
    assert output_audio_data_size_in_bytes > 0
    
    # The output size must equal the sum of inputs plus the expected silence size.
    assert output_audio_data_size_in_bytes == expected_output_data_size_in_bytes

def test_delete_files(dummy_file):
    paths = [dummy_file('txt'), dummy_file('txt')]
    for path in paths:
        assert path.is_file() == True
    delete_files(paths)
    for path in paths:
        assert path.is_file() == False

@pytest.mark.parametrize(
    "key_value, force_prompt",
    [
        pytest.param("my_api_key", False, id="api-key-found"),
        pytest.param(None, False, id="api-key-not-found"),
        pytest.param("my_api_key", True, id="force-prompt-issued"),
    ],
)
def test_check_api_key(key_value, force_prompt):
    service_name = "my_service"
    username = "my_username"
    with (
        patch("main.keyring.get_password") as mock_get,
        patch("main.keyring.set_password") as mock_set,
        patch("main.typer.prompt") as mock_prompt,
    ):  
        mock_get.return_value = key_value
        mock_prompt.return_value = "new_api_key"

        result = check_api_key(service_name, username, force_prompt=force_prompt)

        if key_value and not force_prompt:
            # Key is found and no prompt forced
            assert result == key_value
            mock_get.assert_called_once_with(service_name, username)
            mock_set.assert_not_called()
            mock_prompt.assert_not_called()
        else:
            # Prompt should be called
            assert result == "new_api_key"
            mock_prompt.assert_called_once()
            mock_set.assert_called_once_with(service_name, username, "new_api_key")

def test_valid_voices_success(mock_voice_client):
    """
    Unit test for validate_voices
    Test case where both voices are valid.
    """
    
    # Arrange: Set return values for the mocked API calls
    mock_voice_client.voices.get.return_value = "VoiceObject" 
    
    # Act: Call the function with distinct valid voice IDs
    validate_voices(mock_voice_client, "voice_A_id", "voice_B_id")
    
    # Assert: Check that the API was called for both voices and no exception was raised
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_A_id")
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_B_id")
    assert mock_voice_client.voices.get.call_count == 2


def test_same_voice_ids_exit(mock_voice_client):
    """
    Unit test for validate_voices
    Test case where speaker_one_voice and speaker_two_voice are the same.
    """
    
    # Arrange/Act/Assert: Expect a typer.Exit with status code 1
    with pytest.raises(typer.Exit) as excinfo:
        validate_voices(mock_voice_client, "same_id", "same_id")
    
    assert excinfo.value.exit_code == 1
    # Assert: Ensure no API calls were made (it exits before the try block)
    mock_voice_client.voices.get.assert_not_called()


def test_speaker_one_invalid_exit(mock_voice_client):
    """
    Unit test for validate_voices
    Test case where speaker_one_voice is invalid (raises exception).
    """
    
    # Arrange: Make the first call (voice_A_id) raise an exception, 
    # and the second call (voice_B_id) succeed.
    mock_voice_client.voices.get.side_effect = [
        Exception, # for voice_A_id
        "VoiceObject" # for voice_B_id
    ]
    
    # Act/Assert: Expect a typer.Exit with status code 1
    with pytest.raises(typer.Exit) as excinfo:
        validate_voices(mock_voice_client, "voice_A_id", "voice_B_id")
    
    assert excinfo.value.exit_code == 1
    # Assert: Ensure both API calls were attempted
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_A_id")
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_B_id")
    assert mock_voice_client.voices.get.call_count == 2


def test_speaker_two_invalid_exit(mock_voice_client):
    """
    Unit test for validate_voices
    Test case where speaker_two_voice is invalid (raises exception).
    """
    
    # Arrange: Make the first call (voice_A_id) succeed, 
    # and the second call (voice_B_id) raise an exception.
    mock_voice_client.voices.get.side_effect = [
        "VoiceObject", # for voice_A_id
        Exception # for voice_B_id
    ]
    
    # Act/Assert: Expect a typer.Exit with status code 1
    with pytest.raises(typer.Exit) as excinfo:
        validate_voices(mock_voice_client, "voice_A_id", "voice_B_id")
    
    assert excinfo.value.exit_code == 1
    # Assert: Ensure both API calls were attempted
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_A_id")
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_B_id")
    assert mock_voice_client.voices.get.call_count == 2


def test_both_invalid_exit(mock_voice_client):
    """
    Unit test for validate_voices
    Test case where both voices are invalid (both raise exceptions).
    """
    
    # Arrange: Make both calls raise an exception.
    mock_voice_client.voices.get.side_effect = [
        Exception, # for voice_A_id
        Exception # for voice_B_id
    ]
    
    # Act/Assert: Expect a typer.Exit with status code 1
    with pytest.raises(typer.Exit) as excinfo:
        validate_voices(mock_voice_client, "voice_A_id", "voice_B_id")
    
    assert excinfo.value.exit_code == 1
    # Assert: Ensure both API calls were attempted
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_A_id")
    mock_voice_client.voices.get.assert_any_call(voice_id="voice_B_id")
    assert mock_voice_client.voices.get.call_count == 2

@pytest.mark.parametrize(
    "input_file_is_valid",
    [
        pytest.param(True, id='valid-input-file'),
        pytest.param(False, id='invalid-input-file')
    ],
)
def test_execute_pdf_workflow(dummy_file, input_file_is_valid, output_dir, api_key):
    timestamp = 123456
    expected_text_summary = "What a great text!"
    expected_output_file = output_dir / f'text_summary_{timestamp}.txt'
    if input_file_is_valid:
        input_filepath = dummy_file('pdf', 2 * 1024 * 1024)
        with (
            patch('main.infer_with_pdf_document_understanding', return_value=expected_text_summary),
            patch('main.typer.confirm', return_value=True),
        ):
            text_summary = execute_pdf_workflow(input_filepath, output_dir, api_key, timestamp)
            assert text_summary == expected_text_summary
            with open(expected_output_file, 'r') as f:
                assert f.read() == expected_text_summary
    else:
        input_filepath = dummy_file('txt', 2 * 1024 * 1024)
        with pytest.raises(typer.Exit) as exc_info:
            text_summary = execute_pdf_workflow(input_filepath, output_dir, api_key, timestamp)
            assert exc_info.value.exit_code == 1
    
def test_execute_transcript_generation_workflow(output_dir, api_key):
    text_summary = "This is a great text!"
    unformatted_sys_instrux = TRANSCRIPT_SYS_INSTRUCTIONS
    timestamp = 123456
    label_one = "Abby"
    label_two = "Tiny"
    expected_transcript = "Abby: This is a great text.\nTiny: I'm hungry!"
    expected_output_filepath = Path(output_dir / f'transcript_{timestamp}.txt')
    with (
        patch('main.generate_text', return_value=expected_transcript),
        patch('main.typer.confirm', return_value=True)
    ):
        actual_transcript = execute_transcript_generation_workflow(
            text_summary, 
            output_dir, 
            api_key,
            unformatted_sys_instrux,
            timestamp,
            label_one,
            label_two
        )
        assert actual_transcript == expected_transcript
        with open(expected_output_filepath, 'r') as f:
            assert f.read() == expected_transcript
