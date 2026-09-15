from pathlib import Path

TEXT_EXTENSIONS = {".txt"}


def get_files_to_process(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Provided path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError(
                f"File '{input_path.name}' is not a valid text file. Expected one of the following extensions: {TEXT_EXTENSIONS}"
            )
        return [input_path]

    elif input_path.is_dir():
        files = list(input_path.rglob("*.txt"))
        if not files:
            print(f"Warning: No '*.txt' files found in directory: {input_path}")
        return files

    return []
