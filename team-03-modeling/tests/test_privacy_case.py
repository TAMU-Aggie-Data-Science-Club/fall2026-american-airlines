from aaforecast.privacy import check_file


def test_windows_case_variants_do_not_bypass_artifact_checks(tmp_path):
    artifact = tmp_path / "Outputs" / "summary.txt"
    artifact.parent.mkdir()
    artifact.write_text("private result", encoding="utf-8")
    assert "restricted artifact path/type" in check_file(artifact, tmp_path)
    secret = tmp_path / ".ENV.local"
    secret.write_text("example", encoding="utf-8")
    assert "environment secrets file" in check_file(secret, tmp_path)
