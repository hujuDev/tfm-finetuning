from utils.paths import get_paths, get_repo_root


def test_configured_relative_results_are_repository_relative(tmp_path, monkeypatch):
    monkeypatch.delenv('RESULTS_ROOT', raising=False)
    assert get_paths({'paths': {'results_root': 'runs'}}, repo_root=tmp_path).results_root == tmp_path / 'runs'
    monkeypatch.setenv('RESULTS_ROOT', 'other')
    assert get_paths(repo_root=tmp_path).results_root == tmp_path / 'other'


def test_root_found_from_notebook_subdirectory(tmp_path, monkeypatch):
    monkeypatch.setenv('REPO_ROOT', 'placeholder')  # get_repo_root() exports REPO_ROOT; restore on teardown
    monkeypatch.delenv('REPO_ROOT')
    (tmp_path / 'pyproject.toml').touch()
    notebook_dir = tmp_path / 'notebooks'
    notebook_dir.mkdir()
    assert get_repo_root(notebook_dir) == tmp_path
