import os
from pathlib import Path 
import pytest
import random
import tempfile
import typer
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch, MagicMock
import wave

from main import (
    check_api_key,
    chunk_string,
    combine_wav_files,
    DEFAULT_SPEAKER_ONE_LABEL,
    DEFAULT_SPEAKER_TWO_LABEL,
    delete_files,
    dir_is_valid, 
    file_is_valid,
    generate_audio_chunk_from_text_chunk,
    generate_text,
    infer_with_pdf_document_understanding,
    read_text_from_file,
    select_voices,
    SERVICE_NAME,
    select_speaker_labels,
    transcript_validates,
    USERNAME,
    VOICES,
    write_audio_data_to_wav_file,
    write_text_to_file
)


VOICE_ONE, VOICE_TWO = random.sample(VOICES, 2)

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

@pytest.fixture
def audio_output_filepath():
    fd, output_filepath = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    yield output_filepath
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

@pytest.fixture
def genai_client_mock():
    mock_client = MagicMock()

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

def test_infer_with_pdf_document_understanding(dummy_file, genai_client_mock):
    """
    `infer_with_pdf_document_understanding` is a thin wrapper
    around API calls, so we're just spot-checking the signature
    """
    _, _, set_expected_response = genai_client_mock
    
    pdf_filepath = dummy_file('pdf')
    prompt = "Please summarize this document."
    api_key = "my_api_key"
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
def test_generate_text(sys_instrux, expected_response, genai_client_mock):
    """
    Another test for a thin wrapper around 
    genai service calls
    """
    _, _, set_expected_response = genai_client_mock
    user_prompt = "Tell me about yourself."
    api_key = "my_api_key"
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
    genai_client_mock # mock genai.Client and its count_tokens method
):
    base_string = "I'm baby schlitz health goth pok pok next level brunch shaman butcher hell of aesthetic. Blog food truck jean shorts street art bespoke raw denim yes plz fixie yuccie, subway tile everyday carry flexitarian tote bag. Hammock poke irony photo booth, meh tumeric whatever same 8-bit vinyl ascot cliche kinfolk tote bag unicorn. Ramps lomo trust fund, bespoke irony vape lyft blog unicorn hell of biodiesel yr."

    _, set_count_tokens, _ = genai_client_mock

    set_count_tokens()

    my_string = (base_string + '\n') * 300 

    chunked_strings = chunk_string(my_string, "my_api_key", token_limit=token_limit)

    assert len(chunked_strings) == expected_chunks

    for chunk in chunked_strings:
        assert isinstance(chunk, str) == True
        assert chunk != ""

def test_generate_audio_chunks():
    # TODO: write test
    pass


def test_generate_audio_chunk_from_text_chunk(audio_output_filepath, wav_file_data, genai_client_mock):
    _, _, set_expected_response = genai_client_mock
    text = "Hello"
    output_filepath = audio_output_filepath
    api_key = "my_api_key"
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
            speaker_one_voice, speaker_two_voice = select_voices(voice_one, voice_two)
        assert exc_info.value.exit_code == 1
    else:
        speaker_one_voice, speaker_two_voice = select_voices(voice_one, voice_two)
        assert speaker_one_voice != speaker_two_voice
        assert speaker_one_voice in VOICES
        assert speaker_two_voice in VOICES

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

def test_execute_pdf_workflow():
    # TODO: write test
    pass

def test_execute_transcript_generation_workflow():
    # TODO: write test
    pass

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
