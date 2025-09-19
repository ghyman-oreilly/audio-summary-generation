

## Workflow

* Accept PDF input
* Verify PDF input
  * Is PDF
  * Is under 20MB (in future can increase limit by using GCP File API)
* Use document understanding to summarize document
* Output summary and prompt user: wish to continue? (most likely it will be yes)
* Convert summary to transcript
* Clean up transcript
* Output transcript and prompt user: wish to continue? (usually it will be no)
* Chunk transcript and generate audio files from chunks
* Combine audio files and delete chunk audio
* Output combined audio file

## Options
* Accept summary text and start workflow from there
* Accept transcript text and start workflow from there

## Notes

* Combining audio files may require a tool like ffmpeg. How can we facilitate this without requiring users to install?
* Best way to handle API authentication with large base of users who have varying levels of comfort with tech
* Maybe write as a package with script CLI, but in a way that we can make it a backend for a web app later? 