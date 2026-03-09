from homie_app.init import run_init


def test_run_init_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMIE_STORAGE_PATH", str(tmp_path / ".homie"))
    monkeypatch.chdir(tmp_path)
    cfg = run_init(auto=True)
    assert cfg is not None
    assert cfg.storage.path is not None
