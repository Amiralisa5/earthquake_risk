"""Monte Carlo entry point (hazard / vulnerability / loss analysis)."""

from risk.runner import run_monte_carlo

if __name__ == "__main__":
    analysis_type = "loss"  # 'hazard', 'vul', or 'loss'
    investigation_time = 10
    block_size = 10_000

    run_monte_carlo(
        analysis_type=analysis_type,
        investigation_time=investigation_time,
        block_size=block_size,
    )
