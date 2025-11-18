import yaml

from replicate_runner.profiles import ProfileManager


def test_profile_merge_precedence(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    workspace_config = workspace_dir / "config"
    workspace_config.mkdir(parents=True)
    workspace_file = workspace_config / "profiles.yaml"
    with workspace_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "profiles": {
                    "merge-test": {
                        "model": "workspace/model",
                        "defaults": {"params": {"guidance": 1.0}, "subject": "base"},
                    }
                }
            },
            fh,
        )

    config_home = tmp_path / "config_home"
    user_dir = config_home / "replicate-runner"
    user_dir.mkdir(parents=True)
    user_file = user_dir / "profiles.yaml"
    with user_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "profiles": {
                    "merge-test": {
                        "model": "user/model",
                        "defaults": {"params": {"guidance": 2.0}},
                    }
                }
            },
            fh,
        )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    manager = ProfileManager(cwd=workspace_dir)

    resolved = manager.resolve_profile("merge-test")
    assert resolved.data["model"] == "user/model"
    assert resolved.data["defaults"]["params"]["guidance"] == 2.0
    assert resolved.data["defaults"]["subject"] == "base"
    assert resolved.sources[-1].scope == "user"


def test_profile_unset(tmp_path, monkeypatch):
    config_home = tmp_path / "config_home"
    user_dir = config_home / "replicate-runner"
    user_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    manager = ProfileManager(cwd=tmp_path)
    manager.save_profile(
        "clear-me",
        {
            "model": "one",
            "defaults": {"params": {"guidance": 6.5, "num_outputs": 2}},
        },
    )
    manager.save_profile("clear-me", {}, unset_paths=["defaults.params.guidance"])

    resolved = manager.resolve_profile("clear-me")
    params = resolved.data["defaults"]["params"]
    assert "guidance" not in params
    assert params["num_outputs"] == 2
