from pipeline import train_sft as T


def test_replay_flags_parse():
    a = T.parse_args(["--name", "v4r", "--rows_from", "v4", "--resume_state", "tinker://x/weights/v4-step1500",
                      "--start_step", "1501", "--lr", "4.7297908091376354e-4", "--no_publish"])
    assert a.rows_from == "v4" and a.start_step == 1501 and a.resume_state.endswith("v4-step1500")
    assert a.final_state_ttl == "none"


def test_step_range_and_lr_match_v4_schedule():
    assert T.step_range(1501, 1688) == range(1501, 1688)
    assert len(T.step_range(1501, 1688)) == 187
    assert T.lr_at(1501, 1688, 4.7297908091376354e-4) == 4.7297908091376354e-4 * (1 - 1501 / 1688)
