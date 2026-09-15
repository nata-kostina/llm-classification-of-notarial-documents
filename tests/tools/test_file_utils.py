from pathlib import Path

import pytest

from src.tools.file_utils import get_files_to_process


def test_get_files_to_process_recursive_directory(tmp_path: Path):
    # tmp_path/
    # ├── root.txt
    # ├── ignore.pdf
    # └── sub_folder/
    #     ├── nested.txt
    #     └── deep_folder/
    #         └── deep.txt
    file1 = tmp_path / "root.txt"
    file1.touch()

    (tmp_path / "ignore.pdf").touch()

    sub_folder = tmp_path / "sub_folder"
    sub_folder.mkdir()
    file2 = sub_folder / "nested.txt"
    file2.touch()

    deep_folder = sub_folder / "deep_folder"
    deep_folder.mkdir()
    file3 = deep_folder / "deep.txt"
    file3.touch()

    result = get_files_to_process(tmp_path)

    assert set(result) == {file1, file2, file3}


def test_get_files_to_process_single_valid_file(tmp_path: Path):
    txt_file = tmp_path / "single.txt"
    txt_file.touch()

    result = get_files_to_process(txt_file)

    assert result == [txt_file]


def test_get_files_to_process_single_invalid_file(tmp_path: Path):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.touch()

    with pytest.raises(ValueError, match="is not a valid text file"):
        get_files_to_process(pdf_file)


def test_get_files_to_process_non_existent_path(tmp_path: Path):
    non_existent_path = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Provided path does not exist"):
        get_files_to_process(non_existent_path)
