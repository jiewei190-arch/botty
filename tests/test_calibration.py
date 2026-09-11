"""Score calibration: whether the ranking predicts outcomes.

The scanner's score is the share of a strategy's evidence that was present. It
is not a probability, and this module exists to measure the gap. These tests
are mostly about the honesty of the reporting: a win rate without its sample
size and its error is a number that invites being misread.
"""

from __future__ import annotations

import pytest

from trading_bot.backtesting import MIN_BAND_SAMPLE, Calibration, ScoreBand, calibrate


def trade(confidence, pnl, r=None):
    return {"confidence": confidence, "pnl": pnl, "r_multiple": r}


def sample(confidence, wins, losses, r=1.0):
    return [trade(confidence, 100.0, r) for _ in range(wins)] + [
        trade(confidence, -50.0, -r) for _ in range(losses)
    ]


class TestBucketing:
    def test_trades_land_in_the_right_band(self):
        result = calibrate([trade(65, 10), trade(95, 10)])
        by_label = {band.label: band for band in result.bands}
        assert by_label["60-70"].trades == 1
        assert by_label["90-100"].trades == 1

    def test_a_trade_without_a_score_is_ignored(self):
        """Rather than silently landing in the lowest band."""
        result = calibrate([trade(None, 10), trade(65, 10)])
        assert result.total_trades == 1

    def test_scores_below_every_band_are_ignored(self):
        assert calibrate([trade(10, 5)], bands=(60.0, 100.01)).total_trades == 0

    def test_a_perfect_score_is_included(self):
        """The top edge is exclusive, so 100 must still be counted."""
        assert calibrate([trade(100, 5)]).total_trades == 1

    def test_bands_need_two_edges(self):
        with pytest.raises(ValueError, match="lower and an upper"):
            calibrate([], bands=(50.0,))


class TestOutcomes:
    def test_a_breakeven_trade_counts_as_a_loss(self):
        """Zero after costs is not a win; counting it as one flatters the rate."""
        result = calibrate([trade(85, 0.0)])
        assert result.total_wins == 0

    def test_win_rate_and_expectancy(self):
        result = calibrate(sample(85, wins=3, losses=1, r=2.0))
        band = next(b for b in result.bands if b.label == "80-90")
        assert band.win_rate == pytest.approx(75.0)
        assert result.base_rate == pytest.approx(75.0)

    def test_r_multiples_are_averaged_only_over_trades_that_have_one(self):
        trades = [trade(85, 100, 2.0), trade(85, 100, None)]
        band = next(b for b in calibrate(trades).bands if b.label == "80-90")
        assert band.measured == 1
        assert band.average_r == pytest.approx(2.0)


class TestHonestyOfReporting:
    def test_a_small_band_is_marked_unusable(self):
        result = calibrate(sample(85, wins=2, losses=2))
        band = next(b for b in result.bands if b.label == "80-90")
        assert not band.is_meaningful
        assert "too few" in "\n".join(result.summary_lines())

    def test_a_large_band_is_usable(self):
        result = calibrate(sample(85, wins=MIN_BAND_SAMPLE, losses=MIN_BAND_SAMPLE))
        assert next(b for b in result.bands if b.label == "80-90").is_meaningful

    def test_the_standard_error_shrinks_with_sample_size(self):
        small = ScoreBand(80, 90, trades=40, wins=18, total_r=0.0)
        large = ScoreBand(80, 90, trades=4000, wins=1800, total_r=0.0)
        assert small.standard_error() > large.standard_error() * 5

    def test_too_few_trades_overall_says_so_plainly(self):
        assert "too few to say anything" in calibrate(sample(85, 2, 2)).verdict

    def test_one_readable_band_cannot_be_compared(self):
        result = calibrate(sample(85, wins=40, losses=40))
        assert "cannot be compared" in result.verdict

    def test_a_difference_inside_the_error_is_called_unproven(self):
        """The result that must not be dressed up as an edge."""
        trades = sample(65, wins=40, losses=40) + sample(95, wins=41, losses=39)
        verdict = calibrate(trades).verdict
        assert "does not separate" in verdict
        assert "unproven" in verdict

    def test_a_real_separation_is_reported_as_one(self):
        trades = sample(65, wins=20, losses=80) + sample(95, wins=80, losses=20)
        verdict = calibrate(trades).verdict
        assert "rises with score" in verdict

    def test_an_inverted_ranking_is_not_described_as_rising(self):
        """Higher score, worse outcome — the finding that condemns a ranking."""
        trades = sample(65, wins=80, losses=20) + sample(95, wins=20, losses=80)
        result = calibrate(trades)
        assert not result.is_monotonic
        assert "does not rise cleanly" in result.verdict


class TestLiftOverBaseRate:
    """A band only justifies the ranking if it beats taking everything."""

    def test_lift_is_measured_against_the_base_rate(self):
        trades = sample(65, wins=20, losses=80) + sample(95, wins=80, losses=20)
        result = calibrate(trades)
        assert result.base_rate == pytest.approx(50.0)
        top = next(b for b in result.bands if b.label == "90-100")
        assert result.lift(top) == pytest.approx(30.0)

    def test_no_lift_when_every_band_performs_alike(self):
        trades = sample(65, wins=40, losses=40) + sample(95, wins=40, losses=40)
        result = calibrate(trades)
        for band in result.usable_bands:
            assert result.lift(band) == pytest.approx(0.0, abs=0.01)


class TestEmpty:
    def test_no_trades_yields_a_zeroed_calibration(self):
        result = calibrate([])
        assert result.total_trades == 0
        assert result.base_rate == 0.0
        assert result.expectancy_r == 0.0

    def test_an_empty_calibration_still_summarises(self):
        assert Calibration().summary_lines()
