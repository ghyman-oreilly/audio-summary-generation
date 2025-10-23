import os
from pathlib import Path 
import pytest
import random
import tempfile
import typer
from unittest.mock import patch

from conftest import MINIMAL_FILE_CONTENT
from main import (
    check_api_key,
    chunk_string,
    combine_wav_files,
    DEFAULT_SPEAKER_ONE_LABEL,
    DEFAULT_SPEAKER_TWO_LABEL,
    delete_files,
    dir_is_valid,
    execute_pdf_workflow, 
    execute_transcript_generation_workflow,
    file_is_valid,
    format_sys_instrux,
    generate_audio_chunk_from_text_chunk,
    generate_audio_chunks,
    generate_text,
    infer_with_pdf_document_understanding,
    read_text_from_file,
    select_voices_google,
    SERVICE_NAME,
    select_speaker_labels,
    transcript_validates,
    USERNAME,
    validate_sys_instrux_format,
    VOICES_GOOGLE,
    write_audio_data_to_wav_file,
    write_text_to_file
)
from prompts import TRANSCRIPT_SYS_INSTRUCTIONS


VOICE_ONE, VOICE_TWO = random.sample(VOICES_GOOGLE, 2)


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
    "token_limit, expected_chunks",
    [
        pytest.param(3000, 11),
        pytest.param(None, 11),
        pytest.param(5000, 7),
    ],
)
def test_chunk_string(
    token_limit, 
    expected_chunks, 
    api_key,
    genai_client_mock # mock genai.Client and its count_tokens method
):
    base_string = (
        "I'm baby schlitz health goth pok pok next level brunch shaman butcher hell of aesthetic. "
        "Blog food truck jean shorts street art bespoke raw denim yes plz fixie yuccie, subway tile "
        "everyday carry flexitarian tote bag. Hammock poke irony photo booth, meh tumeric whatever "
        "same 8-bit vinyl ascot cliche kinfolk tote bag unicorn. Ramps lomo trust fund, bespoke irony "
        "vape lyft blog unicorn hell of biodiesel yr."
    )

    _, set_count_tokens, _ = genai_client_mock

    set_count_tokens()

    my_string = (base_string + '\n') * 300 

    chunked_strings = chunk_string(my_string, api_key, token_limit=token_limit)

    assert len(chunked_strings) == expected_chunks

    for chunk in chunked_strings:
        assert isinstance(chunk, str) == True
        assert chunk != ""

def test_generate_audio_chunks(output_dir, api_key):
    text_chunks = ["hello", "world"]
    timestamp = 123456
    expected_filepath_one = Path(output_dir / f'audio_chunk_000_{timestamp}.wav')
    expected_filepath_two = Path(output_dir / f'audio_chunk_001_{timestamp}.wav')
    with patch('main.generate_audio_chunk_from_text_chunk', return_value=None):
        audio_chunk_filepaths = generate_audio_chunks(text_chunks, timestamp, output_dir, api_key)
        assert len(audio_chunk_filepaths) == 2
        assert audio_chunk_filepaths[0] == expected_filepath_one
        assert audio_chunk_filepaths[1] == expected_filepath_two

def test_generate_audio_chunk_from_text_chunk(audio_output_filepath, wav_file_data, genai_client_mock, api_key):
    _, _, set_expected_response = genai_client_mock
    text = "Hello"
    output_filepath = audio_output_filepath
    input_filepath = 'test/test_data/chunk_audio_00.wav'
    expected_response = wav_file_data(input_filepath).get('audio_data')
    set_expected_response(expected_response, is_audio_generation=True)
    generate_audio_chunk_from_text_chunk(text, output_filepath, api_key)
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

def test_combine_wav_files(wav_file_data, audio_output_filepath):
    output_filepath = audio_output_filepath
    input_filepaths = [f'test/test_data/chunk_audio_0{d}.wav' for d in range(0, 3)]
    input_audio_data_size_in_bytes = sum(wav_file_data(f).get("audio_data_size_in_bytes") for f in input_filepaths)

    combine_wav_files([Path(f) for f in input_filepaths], output_filepath)

    output_wav_data = wav_file_data(output_filepath)
    output_audio_data_size_in_bytes = output_wav_data.get("audio_data_size_in_bytes")

    assert input_audio_data_size_in_bytes > 0 and output_audio_data_size_in_bytes > 0
    assert input_audio_data_size_in_bytes == output_audio_data_size_in_bytes

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
    with (
        patch("main.keyring.get_password") as mock_get,
        patch("main.keyring.set_password") as mock_set,
        patch("main.typer.prompt") as mock_prompt,
    ):  
        mock_get.return_value = key_value
        mock_prompt.return_value = "new_api_key"

        result = check_api_key(force_prompt=force_prompt)

        if key_value and not force_prompt:
            # Key is found and no prompt forced
            assert result == key_value
            mock_get.assert_called_once_with(SERVICE_NAME, USERNAME)
            mock_set.assert_not_called()
            mock_prompt.assert_not_called()
        else:
            # Prompt should be called
            assert result == "new_api_key"
            mock_prompt.assert_called_once()
            mock_set.assert_called_once_with(SERVICE_NAME, USERNAME, "new_api_key")

@pytest.mark.parametrize(
    "voice_one, voice_two, is_invalid",
    [
        pytest.param(None, None, False, id="user-selects-no-voices"),
        pytest.param(VOICE_ONE, None, False, id="user-selects-one-voice"),
        pytest.param(VOICE_ONE, VOICE_TWO, False, id="user-selects-two-voices"),
        pytest.param("Huckleberry", None, True, id="user-selects-an-invalid-voice"),
        pytest.param(VOICE_ONE, VOICE_ONE, True, id="user-selects-identical-voices"),
    ],
)
def test_select_voices(voice_one, voice_two, is_invalid):
    if is_invalid:
        with pytest.raises(typer.Exit) as exc_info:
            speaker_one_voice, speaker_two_voice = select_voices_google(voice_one, voice_two)
        assert exc_info.value.exit_code == 1
    else:
        speaker_one_voice, speaker_two_voice = select_voices_google(voice_one, voice_two)
        assert speaker_one_voice != speaker_two_voice
        assert speaker_one_voice in VOICES_GOOGLE
        assert speaker_two_voice in VOICES_GOOGLE

@pytest.mark.parametrize(
    "label_one, label_two, is_invalid",
    [
        pytest.param(None, None, False, id="user-selects-no-labels"),
        pytest.param("Abby", None, False, id="user-selects-one-label"),
        pytest.param("Abby:", None, False, id="user-selects-one-label-with-colon"),
        pytest.param("Abby", "Tiny", False, id="user-selects-two-labels"),
        pytest.param(DEFAULT_SPEAKER_TWO_LABEL, None, True, id="user-selects-default-label-in-wrong-location"),
        pytest.param("Tiny", "Tiny", True, id="user-selects-identical-labels"),
    ],
)
def test_select_speaker_labels(label_one, label_two, is_invalid):
    if is_invalid:
        with pytest.raises(typer.Exit) as exc_info:
            speaker_label_one, speaker_label_two = select_speaker_labels(label_one, label_two)
            assert exc_info.value.exit_code == 1
    else:
        speaker_label_one, speaker_label_two = select_speaker_labels(label_one, label_two)
        assert not ":" in [l[-1] for l in [speaker_label_one, speaker_label_two]] # no colon at end of strings
        assert speaker_label_one != speaker_label_two
        if not label_one and not label_two:
            assert speaker_label_one == DEFAULT_SPEAKER_ONE_LABEL
            assert speaker_label_two == DEFAULT_SPEAKER_TWO_LABEL
        else:
            assert speaker_label_one == label_one.replace(':', '') if label_one else True
            assert speaker_label_two == label_two.replace(':', '') if label_two else True

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
    
def test_format_sys_instrux():
    template = "Label the speakers with {speaker_1} and {speaker_2}"
    label_one = "Abby"
    label_two = "Tiny"
    expected = "Label the speakers with Abby and Tiny"
    actual = format_sys_instrux(template, label_one, label_two)
    assert actual == expected

@pytest.mark.parametrize(
    "template, expected",
    [
        pytest.param(
            "Label the speakers with {speaker_1} and {speaker_2}", 
            True, 
            id='instrux-template-is-valid'
        ),
        pytest.param(
            "You don't really need to label the speakers.",
            False, 
            id='instrux-template-is-invalid'
        )
    ],
)
def test_validate_sys_instrux_format(template, expected):
    assert validate_sys_instrux_format(template) == expected

def test_execute_transcript_generation_workflow(output_dir, api_key):
    text_summary = "This is a great text!"
    unformatted_sys_instrux = TRANSCRIPT_SYS_INSTRUCTIONS
    timestamp = 123456
    label_one = "Abby"
    label_two = "Tiny"
    expected_transcript = "Abby: This is a great text.\nTiny: I'm hungry!"
    expected_output_filepath = Path(output_dir / f'transcript_{timestamp}.txt')
    with (
        patch('main.format_sys_instrux', return_value=unformatted_sys_instrux),
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

@pytest.mark.parametrize(
    "transcript, speaker1, speaker2, expected",
    [
        pytest.param("Abby: hi\nTiny: hello", "Abby", "Tiny", True, id='speaker-labels-both-found'),
        pytest.param("Abby: hi\nJoannie: hello", "Abby", "Tiny", False, id='one-speaker-label-missing'),
        pytest.param("Nyla: hi\nJoannie: hello", "Abby", "Tiny", False, id='both-speaker-labels-missing'),
    ],
)
def test_transcript_validates(transcript, speaker1, speaker2, expected):
    assert transcript_validates(transcript, speaker1, speaker2) == expected
