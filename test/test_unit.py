import json
import os
from pathlib import Path, PosixPath 
import pytest
import random
import tempfile
import typer
from unittest.mock import call, MagicMock, patch
import wave

from conftest import MINIMAL_FILE_CONTENT, DUMMY_BACKUP_DATA
from main import (
    check_api_key,
    check_tokenizer_data_availability,
    create_generation_data,
    create_speaker_text_chunks,
    combine_wav_files,
    delete_files,
    dir_is_valid,
    execute_pdf_workflow, 
    execute_transcript_generation_workflow,
    file_is_valid,
    generate_audio_with_timeout,
    generate_audio_segments,
    generate_menu,
    generate_text,
    get_voice_owner_id_from_community_library,
    infer_with_pdf_document_understanding,
    read_backup_from_json_file,
    read_text_from_file,
    validate_backup_data,
    validate_backup_data_shape,
    validate_backup_data_voice_ids,
    validate_backup_data_segment_filepaths,
    validate_voices,
    voice_exists_in_account_library,
    write_audio_data_to_wav_file,
    write_backup_to_json_file,
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
    pass

@pytest.fixture
def mock_elevenlabs_client():
    """Fixture to provide a mocked ElevenLabs client."""
    client = MagicMock(spec=ElevenLabs)
    client.voices = MagicMock()
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
    text_input,
    char_limit,
    expected_speaker_turns,
    expected_total_chunks
):
    """Tests the new structure and splitting logic of create_speaker_text_chunks."""

    def chunk_segment_by_sentences(line, char_limit):
        """
        Mock implementation of the helper function chunk_segment_by_sentences
        """
        # This mock always splits a long line into three smaller chunks
        return [
            f"Chunk 1 of {len(line)}",
            f"Chunk 2 of {len(line)}",
            f"Chunk 3 of {len(line)}",
        ]

    with patch(
        'main.chunk_segment_by_sentences', 
        side_effect=chunk_segment_by_sentences
        ) as mock_split_func:

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

def test_generate_audio_segments(mock_elevenlabs_client):
    """
    Unit test against generate_audio_segments 
    """
    generation_data = [
        {'voice_id': 'my_vid_1', 'text': 'hello', 'filepath': 'my_filepath1'}, 
        {'voice_id': 'my_vid_2', 'text': 'world', 'filepath': 'my_filepath2'}
    ]
    mock_return_values = [
        'request_id_0', 
        'request_id_1'
    ]
    expected_data_for_backup = [
        {'voice_id': 'my_vid_1', 'text': 'hello', 'filepath': 'my_filepath1', 'request_id': 'request_id_0'}, 
        {'voice_id': 'my_vid_2', 'text': 'world', 'filepath': 'my_filepath2', 'request_id': 'request_id_1'}
    ]
    fake_backup_filepath = Path('fake_path')
    model_id = 'some_model'

    with (
        patch('main.generate_audio_with_timeout', side_effect=mock_return_values) as mock_gen,
        patch('main.write_backup_to_json_file') as mock_backup
    ):
        generate_audio_segments(
            generation_data, 
            mock_elevenlabs_client, 
            model_id=model_id,
            backup_filepath=fake_backup_filepath
        )
        
        # generate_audio_with_timeout should be called once for each item in the data (2 times)
        mock_gen.call_count == 2
        
        # write_backup_to_json_file should be called exactly once
        mock_backup.assert_called_once()
                
        # The expected calls for generate_audio_with_timeout, including previous_request_ids
        expected_calls = [
            call(
                text='hello',
                voice_id='my_vid_1',
                output_file=Path('my_filepath1'),
                tts_client=mock_elevenlabs_client,
                model_id=model_id,
                previous_request_ids=[]
            ),
            call(
                text='world',
                voice_id='my_vid_2',
                output_file=Path('my_filepath2'),
                tts_client=mock_elevenlabs_client,
                model_id=model_id,
                previous_request_ids=['request_id_0']
            )
        ]
        
        mock_gen.assert_has_calls(expected_calls, any_order=False)

        # Check that the mock_backup was called with the modified data
        mock_backup.assert_called_once_with(
            expected_data_for_backup, 
            fake_backup_filepath
        )

def test_generate_audio_with_timeout_succeeds(mock_elevenlabs_client):
    """
    Unit test against generate_audio_with_timeout
    """
    expected_request_id = "successful_request_id_123"

    with (
        patch(
            'main.generate_audio_chunk_from_chunk', 
            return_value=expected_request_id
        ) as mock_chunk_gen
    ):

        request_id = generate_audio_with_timeout(
                    text="Test text",
                    voice_id="test_voice",
                    output_file=Path("test_output.mp3"),
                    tts_client=mock_elevenlabs_client,
                    model_id="test_model",
                    previous_request_ids=["prev_id"]
                )
        
        assert request_id == expected_request_id

        mock_chunk_gen.assert_called_once_with(
                text="Test text",
                voice_id="test_voice",
                output_file=Path("test_output.mp3"),
                tts_client=mock_elevenlabs_client,
                model_id="test_model",
                previous_request_ids=["prev_id"]
            )

def test_generate_audio_with_timeout_timeout_logic_works(mock_elevenlabs_client):
    """
    Tests that the correct timeout value is passed and the TimeoutError is handled.
    """
    TEST_TIMEOUT = 5.0

    #  Mock the Future object (where the exception is forced)
    mock_future = MagicMock()
    mock_future.result.side_effect = TimeoutError 

    # Mock the Executor instance (The object bound to 'as executor')
    # This mock represents the *active* executor inside the 'with' block.
    mock_executor_instance = MagicMock()
    mock_executor_instance.submit.return_value = mock_future

    # Mock the ThreadPoolExecutor class
    mock_executor_class = MagicMock()

    # This links the class mock to the instance mock correctly.
    # When the code runs TPE(), it returns a mock object.
    # When the code runs with ... as executor:, it calls __enter__ on that mock object.
    # We tell __enter__ to return our mock_executor_instance.
    mock_executor_class.return_value.__enter__.return_value = mock_executor_instance

    # Patch the function being executed and the executor class
    with (
        patch('main.generate_audio_chunk_from_chunk'),
        # Ensure this patch path is correct!
        patch('main.ThreadPoolExecutor', new=mock_executor_class)
    ):
        
        with pytest.raises(typer.Exit) as excinfo:
            generate_audio_with_timeout(
                text="Test timeout",
                voice_id="test_voice",
                output_file=Path("test_output.mp3"),
                tts_client=mock_elevenlabs_client,
                model_id="test_model",
                timeout=TEST_TIMEOUT, 
                previous_request_ids=[]
            )
        
        assert excinfo.value.exit_code == 1
        mock_future.result.assert_called_once_with(timeout=TEST_TIMEOUT)
        mock_executor_instance.submit.assert_called_once()

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

class TestValidateVoices():
    """
    Unit tests for validate_voices
    """
    def test_validate_voices_success(self, mock_elevenlabs_client):
        """
        Unit test for validate_voices
        Test case where both voices are valid.
        """
        # Set return values for the mocked API calls
        mock_elevenlabs_client.voices.get.return_value = "VoiceObject" 
        
        # Call the function with distinct valid voice IDs
        validate_voices(mock_elevenlabs_client, "voice_A_id", "voice_B_id")
        
        # Check that the API was called for both voices and no exception was raised
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_A_id")
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_B_id")
        assert mock_elevenlabs_client.voices.get.call_count == 2


    def test_validate_voices_same_voice_ids_exit(self, mock_elevenlabs_client):
        """
        Unit test for validate_voices
        Test case where speaker_one_voice and speaker_two_voice are the same.
        """
        
        # Arrange/Act/Assert: Expect a typer.Exit with status code 1
        with pytest.raises(typer.Exit) as excinfo:
            validate_voices(mock_elevenlabs_client, "same_id", "same_id")
        
        assert excinfo.value.exit_code == 1
        # Assert: Ensure no API calls were made (it exits before the try block)
        mock_elevenlabs_client.voices.get.assert_not_called()


    def test_validate_voices_speaker_one_invalid_exit(self, mock_elevenlabs_client):
        """
        Unit test for validate_voices
        Test case where speaker_one_voice is invalid (raises exception).
        """
        
        # Arrange: Make the first call (voice_A_id) raise an exception, 
        # and the second call (voice_B_id) succeed.
        mock_elevenlabs_client.voices.get.side_effect = [
            Exception, # for voice_A_id
            "VoiceObject" # for voice_B_id
        ]
        
        # Act/Assert: Expect a typer.Exit with status code 1
        with pytest.raises(typer.Exit) as excinfo:
            validate_voices(mock_elevenlabs_client, "voice_A_id", "voice_B_id")
        
        assert excinfo.value.exit_code == 1
        # Assert: Ensure both API calls were attempted
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_A_id")
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_B_id")
        assert mock_elevenlabs_client.voices.get.call_count == 2


    def test_validate_voices_speaker_two_invalid_exit(self, mock_elevenlabs_client):
        """
        Unit test for validate_voices
        Test case where speaker_two_voice is invalid (raises exception).
        """
        
        # Arrange: Make the first call (voice_A_id) succeed, 
        # and the second call (voice_B_id) raise an exception.
        mock_elevenlabs_client.voices.get.side_effect = [
            "VoiceObject", # for voice_A_id
            Exception # for voice_B_id
        ]
        
        # Act/Assert: Expect a typer.Exit with status code 1
        with pytest.raises(typer.Exit) as excinfo:
            validate_voices(mock_elevenlabs_client, "voice_A_id", "voice_B_id")
        
        assert excinfo.value.exit_code == 1
        # Assert: Ensure both API calls were attempted
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_A_id")
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_B_id")
        assert mock_elevenlabs_client.voices.get.call_count == 2


    def test_validate_voices_both_invalid_exit(self, mock_elevenlabs_client):
        """
        Unit test for validate_voices
        Test case where both voices are invalid (both raise exceptions).
        """
        
        # Arrange: Make both calls raise an exception.
        mock_elevenlabs_client.voices.get.side_effect = [
            Exception, # for voice_A_id
            Exception # for voice_B_id
        ]
        
        # Act/Assert: Expect a typer.Exit with status code 1
        with pytest.raises(typer.Exit) as excinfo:
            validate_voices(mock_elevenlabs_client, "voice_A_id", "voice_B_id")
        
        assert excinfo.value.exit_code == 1
        # Assert: Ensure both API calls were attempted
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_A_id")
        mock_elevenlabs_client.voices.get.assert_any_call(voice_id="voice_B_id")
        assert mock_elevenlabs_client.voices.get.call_count == 2

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
    expected_transcript = "This is a great text.\nI'm hungry!"
    with (
        patch('main.generate_text', return_value=expected_transcript),
        patch('main.write_text_to_file') as mock_write,
        patch('main.typer.confirm', return_value=True)
    ):
        actual_transcript = execute_transcript_generation_workflow(
            text_summary, 
            output_dir, 
            api_key,
            unformatted_sys_instrux,
            timestamp,
            text_model='my_text_model'
        )
        assert actual_transcript == expected_transcript
        mock_write.assert_called_once()

# TODO: organize unit tests for separate commands. Same with E2E tests.

def test_write_backup_to_json_file(output_dir):
    """
    Unit test against write_backup_to_json_file
    """
    input_data = DUMMY_BACKUP_DATA
    output_filepath = Path(output_dir / 'my_backup_file.json')
    write_backup_to_json_file(input_data, output_filepath)
    with open(str(output_filepath), "r") as f:
        output_data = json.load(f)

    assert output_filepath.exists()
    assert output_data == input_data

def test_read_backup_from_json_file():
    """
    Unit test against read_backup_from_json_file
    """
    input_filepath = 'test/test_data/my_backup_file.json'
    expected_data = DUMMY_BACKUP_DATA
    
    with patch('main.validate_backup_data') as p:
        actual_data = read_backup_from_json_file(input_filepath)
    
    p.assert_called_once_with(
        expected_data,
        None,
        True,
        ['voice_id', 'text', 'filepath', 'request_id']
    )
    
    assert actual_data == expected_data 

@pytest.mark.parametrize(
    "validate_voices",
    [
        pytest.param(True, id='should-validate-voices'),
        pytest.param(False, id='should-not-validate-voices')
    ],
)
def test_validate_backup_data(validate_voices):
    """
    Unit test against validate_backup_data
    """
    input_data = DUMMY_BACKUP_DATA
    expected_fields = ['voice_id', 'text', 'filepath', 'request_id']
    with (
        patch('main.validate_backup_data_shape') as p_shape,
        patch('main.validate_backup_data_voice_ids') as p_voice,
        patch('main.validate_backup_data_segment_filepaths') as p_filepaths
    ):
        validate_backup_data(input_data, validate_voices=validate_voices)

    if validate_voices:
        p_voice.assert_called_once_with(
            input_data,
            None
        )
    else:
        p_voice.assert_not_called()
    
    p_shape.assert_called_once_with(
        input_data,
        expected_fields
    )
    
    p_filepaths.assert_called_once_with(
        input_data
    )

@pytest.mark.parametrize(
    "input_data, has_two_voice_ids",
    [
        pytest.param(
            [
                {'voice_id': 'voice_id_1'},
                {'voice_id': 'voice_id_2'},
                {'voice_id': 'voice_id_2'}
            ], 
            True, id='has-two-voice-ids'
        ),
        pytest.param(
            [
                {'voice_id': 'voice_id_1'},
                {'voice_id': 'voice_id_1'},
                {'voice_id': 'voice_id_1'}
            ], False, id='has-one-voice-id'
        ),
        pytest.param(            
            [
                {'voice_id': 'voice_id_1'},
                {'voice_id': 'voice_id_2'},
                {'voice_id': 'voice_id_3'}
            ], False, id='has-three-voice-ids'
        )
    ],
)
def test_validate_backup_data_voice_ids(mock_elevenlabs_client, input_data, has_two_voice_ids):
    """
    Unit test against validate_backup_data_voice_ids
    """
    with patch('main.validate_voices') as p:
        if has_two_voice_ids:
            validate_backup_data_voice_ids(input_data, mock_elevenlabs_client)
            # can't use p.assert_called_once_with() b/c two arguments come from an unordered set
            # so we'll inspect the args individually
            actual_call = p.call_args[0]
            assert len(actual_call) == 3 # number of expected args
            assert actual_call[0] is mock_elevenlabs_client
            voice_id_args = actual_call[1:]
            assert len(voice_id_args) == 2
            assert 'voice_id_1' in voice_id_args
            assert 'voice_id_2' in voice_id_args
        else:
            with pytest.raises(typer.Exit) as excinfo:
                validate_backup_data_voice_ids(input_data, mock_elevenlabs_client)
            p.assert_not_called()
            assert excinfo.value.exit_code == 1

@pytest.mark.parametrize(
    "filepaths_are_valid",
    [
        pytest.param(True, id='filepaths-are-valid'),
        pytest.param(False, id='filepaths-are-invalid')
    ],
)    
def test_validate_backup_data_segment_filepaths(filepaths_are_valid):
    """
    Unit test against validate_backup_data_segment_filepaths
    """
    input_data = [{'filepath': 'my_fake_filepath1.wav'}, {'filepath': 'my_fake_filepath2.wav'}]

    expected_calls = [
        call(PosixPath('my_fake_filepath1.wav'), '.wav'),
        call(PosixPath('my_fake_filepath2.wav'), '.wav')
    ]

    with patch('main.file_is_valid', return_value=filepaths_are_valid) as p_file_valid:
        if filepaths_are_valid:
            validate_backup_data_segment_filepaths(input_data)
            p_file_valid.assert_has_calls(
                expected_calls,
                any_order=False
            )
        else:
            with pytest.raises(typer.Exit) as excinfo:
                validate_backup_data_segment_filepaths(input_data)
            p_file_valid.assert_has_calls(
                expected_calls,
                any_order=False
            )
            assert excinfo.value.exit_code == 1

@pytest.mark.parametrize(
    "input_data, is_valid",
    [
        pytest.param(DUMMY_BACKUP_DATA, True, id='data-is-valid'),
        pytest.param(DUMMY_BACKUP_DATA[0], False, id='data-is-not-list'),
        pytest.param(
            [[DUMMY_BACKUP_DATA[0]], DUMMY_BACKUP_DATA[1]], 
            False, 
            id='data-subitem-isnt-a-dict'
        ),
        pytest.param(
            [{'voice_id': 1, 'text': 'str', 'filepath': 'str', 'request_id': 'str'}], 
            False, 
            id='data-field-isnt-a-string'
        ),
        pytest.param(
            [{'text': 'str', 'filepath': 'str', 'request_id': 'str'}],
            False, 
            id='expected-field-not-found')
    ]
)    
def test_validate_backup_data_shape(input_data, is_valid):
    """
    Unit test against validate_backup_data_shape
    """
    if is_valid:
        assert validate_backup_data_shape(input_data) == True
    else:
        with pytest.raises(typer.Exit) as excinfo:
            validate_backup_data_shape(input_data)
        assert excinfo.value.exit_code == 1

@patch('main.TerminalMenu')
@pytest.mark.parametrize(
    "options, title_no_multi, multiselect",
    [
        pytest.param([0, 1, 2], False, False, id='simple-invocation'),
        pytest.param([0, 1, 2], True, False, id='invocation-with-title'),
        pytest.param([0, 1, 2], False, True, id='invocation-with-multiselect')
    ],
)    
def test_generate_menu(term_menu_mock, options, title_no_multi, multiselect):
    """
    Unit test against generate_menu
    """
    title = 'my_title'
    term_menu_mock.return_value.show.return_value = 0
    if title_no_multi:
        assert generate_menu(options, title) == 0
        term_menu_mock.assert_called_once_with(
            options,
            title=title
        )
    elif multiselect:
        # kwargs can't be passed without explicitly passing a title as well 
        # (even if setting title to None)
        assert generate_menu(options, title=title, multiselect=True) == 0
        term_menu_mock.assert_called_once_with(
            options,
            title=title,
            multiselect=True
        )
    else:
        assert generate_menu(options) == 0
        term_menu_mock.assert_called_once_with(options, title=None)

@patch('main.nltk')        
@pytest.mark.parametrize(
    "models_found",
    [
        pytest.param(True, id='punkt-models-found'),
        pytest.param(False, id='punkt-models-missing')
    ],
)   
def test_check_tokenizer_data_availability(nltk_mock, models_found):
    """
    Unit test against check_tokenizer_data_availability
    """
    if models_found:
        check_tokenizer_data_availability()
        nltk_mock.data.find.assert_called_once()
        nltk_mock.download.assert_not_called()
    else:
        nltk_mock.data.find.side_effect = LookupError
        check_tokenizer_data_availability()
        nltk_mock.data.find.assert_called_once()
        nltk_mock.download.assert_called_once()
     
@pytest.mark.parametrize(
    "voice_exists, other_exception_raised",
    [
        pytest.param(True, False, id='voice-exists'),
        pytest.param(False, False, id='voice-doesnt-exist'),
        pytest.param(False, True, id='other-exception-raised')
    ],
)   
def test_voice_exists_in_account_library(voice_exists, other_exception_raised, mock_elevenlabs_client):
    """
    Unit test against voice_exists_in_account_library
    """
    voice_id = 'my_voice_id'
    if voice_exists:
        assert voice_exists_in_account_library(voice_id, mock_elevenlabs_client) == True
        mock_elevenlabs_client.voices.get.assert_called_once()
    elif not other_exception_raised:
        mock_elevenlabs_client.voices.get.side_effect = Exception('... voice_not_found ...')
        assert voice_exists_in_account_library(voice_id, mock_elevenlabs_client) == False
    else:
        mock_elevenlabs_client.voices.get.side_effect = Exception('... other exception message ...')
        with pytest.raises(Exception):
            voice_exists_in_account_library(voice_id, mock_elevenlabs_client)

class TestGetVoiceOwnerIdFromCommunityLibrary():
    """
    Unit tests against get_voice_owner_id_from_community_library
    """

    # Helper to create fake voice objects quickly
    def make_mock_voice(self, v_id, owner_id="owner_123"):
        m = MagicMock()
        m.voice_id = v_id
        m.public_owner_id = owner_id
        return m

    @patch('main.time.sleep')
    def test_get_voice_owner_scenarios(self, mock_sleep):
        """
        Unit test against get_voice_owner_id_from_community_library

        voice_id is found
        """
        target_voice_id = "target_id"
        expected_owner_id = "found_owner_abc"
        mock_client = MagicMock()

        # Page 1: Wrong voices, has_more = True
        page_1 = MagicMock()
        page_1.voices = [self.make_mock_voice("wrong_id_1"), self.make_mock_voice("wrong_id_2")]
        page_1.has_more = True

        # Page 2: Target voice exists, has_more = True
        # (we should stop paginating here)
        page_2 = MagicMock()
        page_2.voices = [self.make_mock_voice(target_voice_id, expected_owner_id)]
        page_2.has_more = True

        # Page 3: Wrong voices, has_more = False
        page_3 = MagicMock()
        page_3.voices = [self.make_mock_voice("wrong_id_3"), self.make_mock_voice("wrong_id_4")]
        page_3.has_more = False

        # Configure the client to return the pages
        mock_client.voices.get_shared.side_effect = [page_1, page_2, page_3]

        result = get_voice_owner_id_from_community_library(target_voice_id, mock_client)

        assert result == expected_owner_id
        
        assert mock_client.voices.get_shared.call_count == 2
        mock_client.voices.get_shared.assert_has_calls([
            call(page_size=100, page=0),
            call(page_size=100, page=1)
        ])
        
        mock_sleep.assert_called_once()

    @patch('main.time.sleep')
    def test_voice_not_found(self, mock_sleep):
        """
        Unit test against get_voice_owner_id_from_community_library

        voice_id is not found
        """
        mock_client = MagicMock()
        target_voice_id = "missing_id"

        page_1 = MagicMock()
        page_1.voices = [self.make_mock_voice("other_id")]
        page_1.has_more = False
        
        mock_client.voices.get_shared.return_value = page_1

        result = get_voice_owner_id_from_community_library(target_voice_id, mock_client)
        
        assert result is None
        mock_client.voices.get_shared.assert_called_once()

@pytest.mark.parametrize(
    "has_multistring_sublist",
    [
        pytest.param(True, id='simple-chunk-list'),
        pytest.param(False, id='list-with-multistring-sublist')
    ],
)   
def test_create_generation_data(has_multistring_sublist):
    """
    Unit test against create_generation_data

    Text strings within a single sublist should be
    associated with the same voice ID 
    """
    speaker_one = 'abc'
    speaker_two = 'xyz'
    timestamp = '123'
    output_filepath = 'my_fake_filepath'
    if has_multistring_sublist:
        transcript_chunks = [
            ["Hey there."],
            ["Hello!"],
            ["I'm trying to write some tests."],
            ["That's great to hear! Good luck!"]
        ]
        expected_output = [
            {
                'voice_id': 'abc', 
                'text': 'Hey there.', 
                'filepath': 'my_fake_filepath/audio_chunk_000_123.wav'
            },
            {
                'voice_id': 'xyz', 
                'text': 'Hello!', 
                'filepath': 'my_fake_filepath/audio_chunk_001_123.wav'
            },
            {
                'voice_id': 'abc', 
                'text': 'I\'m trying to write some tests.', 
                'filepath': 'my_fake_filepath/audio_chunk_002_123.wav'
            },
            {
                'voice_id': 'xyz', 
                'text': 'That\'s great to hear! Good luck!', 
                'filepath': 'my_fake_filepath/audio_chunk_003_123.wav'
            }
        ]
        assert create_generation_data(
            transcript_chunks,
            Path(output_filepath),
            timestamp,
            speaker_one,
            speaker_two
        ) == expected_output
    else:
        transcript_chunks = [
            ["Hey there."],
            ["Hello!", "I'm trying to write some tests."], # two-string sublist
            ["That's great to hear! Good luck!"]
        ]
        expected_output = [
            {
                'voice_id': 'abc', 
                'text': 'Hey there.', 
                'filepath': 'my_fake_filepath/audio_chunk_000_123.wav'
            },
            {
                'voice_id': 'xyz', # two consequtive dicts with this id
                'text': 'Hello!',
                'filepath': 'my_fake_filepath/audio_chunk_001_123.wav'
            },
            {
                'voice_id': 'xyz', 
                'text': 'I\'m trying to write some tests.',
                'filepath': 'my_fake_filepath/audio_chunk_002_123.wav'
            },
            {
                'voice_id': 'abc', 
                'text': 'That\'s great to hear! Good luck!', 
                'filepath': 'my_fake_filepath/audio_chunk_003_123.wav'
            }
        ]
        assert create_generation_data(
            transcript_chunks,
            Path(output_filepath),
            timestamp,
            speaker_one,
            speaker_two
        ) == expected_output


def test_get_previous_request_id():
    # TODO: write test
    pass

def test_regenerate_audio_segments():
    # TODO: write test
    pass

def test_chunk_segment_by_sentences():
    # TODO: write test
    pass

def test_generate_audio_chunk_from_chunk():
    # TODO: write test
    pass

def test_jitter_wait():
    # TODO: write test
    pass

def test_execute_audio_generation_workflow():
    # TODO: write test
    pass

def test_execute_audio_regeneration_workflow():
    # TODO: write test
    pass