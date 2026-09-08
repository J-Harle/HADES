from MODULES.UPPUMPING.omax import calculate_omega_max


def test_omega_max_handles_spectrum_without_frequencies_above_cutoff():
    assert calculate_omega_max([10.0, 50.0, 100.0]) == (None, 0)


def test_omega_max_selects_highest_frequency_in_window():
    assert calculate_omega_max([90.0, 120.0, 180.0, 240.0]) == (180.0, 2)
