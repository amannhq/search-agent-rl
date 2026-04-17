"""Environment contract tests."""

from __future__ import annotations

from searcharena import SearchAction, SearchEnvironment


def test_seeded_reset_returns_same_task(corpus, tasks) -> None:
    """Equal seeds should choose equal tasks."""
    env_one = SearchEnvironment(corpus=corpus, tasks=tasks)
    env_two = SearchEnvironment(corpus=corpus, tasks=tasks)

    obs_one = env_one.reset(seed=11)
    obs_two = env_two.reset(seed=11)

    assert obs_one.question == obs_two.question


def test_step_after_done_returns_terminal_error(env: SearchEnvironment) -> None:
    """Stepping after a terminal answer should stay terminal."""
    env.reset()
    env.step(SearchAction.make_answer("done"))

    obs = env.step(SearchAction.make_search("after done"))

    assert obs.done is True
    assert obs.action_result is not None
    assert "error" in obs.action_result
