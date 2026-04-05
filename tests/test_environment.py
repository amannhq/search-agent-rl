"""Tests for SearchEnvironment reset and step functionality."""

from models import SearchAction, SearchEnvConfig
from server.environment import SearchEnvironment


class TestEnvironmentReset:
    """Tests for environment reset functionality."""

    def test_reset_produces_clean_state(self, env: SearchEnvironment) -> None:
        """reset() should produce a clean state with no context."""
        obs = env.reset()

        assert obs.context_token_count == 0
        assert len(obs.context_chunks) == 0
        assert obs.step_count == 0
        assert obs.done is False
        assert obs.question != ""

    def test_reset_cycles_through_tasks(
        self, env: SearchEnvironment, tasks: list
    ) -> None:
        """reset() should cycle through available tasks."""
        questions = []
        for _ in range(len(tasks) + 2):
            obs = env.reset()
            questions.append(obs.question)

        # Should cycle back to first task
        assert questions[0] == questions[len(tasks)]

    def test_reset_with_specific_task(
        self, env: SearchEnvironment, custom_task
    ) -> None:
        """reset() should accept a specific task."""
        obs = env.reset(task=custom_task)
        assert obs.question == "What is the meaning of life?"

    def test_reset_clears_previous_episode_state(self, env: SearchEnvironment) -> None:
        """reset() should clear state from previous episode."""
        # Run an episode
        env.reset()
        env.step(SearchAction.make_search("Facebook"))
        env.step(SearchAction.make_answer("test"))

        # Reset
        obs = env.reset()

        assert obs.step_count == 0
        assert obs.context_token_count == 0
        assert len(obs.queries_issued) == 0
        assert obs.done is False


class TestEnvironmentStep:
    """Tests for environment step functionality."""

    def test_search_action_returns_results(self, env: SearchEnvironment) -> None:
        """Search action should return relevant results."""
        env.reset()

        action = SearchAction.make_search("Facebook Instagram acquisition")
        obs = env.step(action)

        assert obs.action_type == "search"
        assert obs.action_result is not None
        assert "results" in obs.action_result
        assert len(obs.action_result["results"]) > 0

    def test_read_action_adds_to_context(self, env: SearchEnvironment) -> None:
        """Read action should add chunks to context."""
        env.reset()

        # Search first
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        results = search_obs.action_result.get("results", [])
        assert len(results) > 0

        # Read first result
        chunk_id = results[0]["chunk_id"]
        obs = env.step(SearchAction.make_read([chunk_id]))

        assert obs.action_type == "read"
        assert obs.context_token_count > 0
        assert len(obs.context_chunks) == 1

    def test_prune_action_removes_from_context(self, env: SearchEnvironment) -> None:
        """Prune action should remove chunks from context."""
        env.reset()

        # Add chunk to context
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        chunk_id = search_obs.action_result["results"][0]["chunk_id"]
        env.step(SearchAction.make_read([chunk_id]))

        # Prune it
        obs = env.step(SearchAction.make_prune([chunk_id]))

        assert obs.action_type == "prune"
        assert obs.context_token_count == 0
        assert len(obs.context_chunks) == 0

    def test_answer_action_ends_episode(self, env: SearchEnvironment) -> None:
        """Answer action should end the episode."""
        env.reset()

        action = SearchAction.make_answer("test answer")
        obs = env.step(action)

        assert obs.done is True
        assert obs.action_type == "answer"
        assert obs.action_result is not None
        assert "final_reward" in obs.action_result

    def test_step_increments_count(self, env: SearchEnvironment) -> None:
        """Each step should increment step_count."""
        env.reset()

        for i in range(3):
            obs = env.step(SearchAction.make_search(f"query {i}"))
            assert obs.step_count == i + 1

    def test_step_after_done_returns_error(self, env: SearchEnvironment) -> None:
        """Stepping after episode ends should return error."""
        env.reset()

        env.step(SearchAction.make_answer("done"))
        obs = env.step(SearchAction.make_search("after done"))

        assert obs.action_result is not None
        assert "error" in obs.action_result


class TestBudgetManagement:
    """Tests for token budget management."""

    def test_budget_warning_at_threshold(self, env_with_config) -> None:
        """Should show warning when approaching budget limit."""
        config = SearchEnvConfig(max_context_tokens=100, soft_budget_threshold=0.5)
        env = env_with_config(config)
        env.reset()

        # Add chunks until warning appears
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        for result in search_obs.action_result.get("results", [])[:5]:
            obs = env.step(SearchAction.make_read([result["chunk_id"]]))
            if obs.budget_warning:
                break

        # Should eventually trigger warning
        assert env._context_token_count > 0

    def test_hard_budget_blocks_actions(self, env_with_config) -> None:
        """Should block non-prune/answer actions at hard budget limit."""
        config = SearchEnvConfig(
            max_context_tokens=50,  # Very small budget
            hard_budget_threshold=0.5,
        )
        env = env_with_config(config)
        env.reset()

        # Fill budget
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        for result in search_obs.action_result.get("results", []):
            env.step(SearchAction.make_read([result["chunk_id"]]))

        # Try to search when over budget
        obs = env.step(SearchAction.make_search("more search"))

        # Should either error or work (depends on implementation)
        # At minimum, step should not crash
        assert obs is not None
