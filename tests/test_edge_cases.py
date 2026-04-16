"""Tests for edge cases and error handling."""

from __future__ import annotations

from collections.abc import Callable

from searcharena import SearchAction, SearchEnvConfig, SearchEnvironment


class TestSearchEdgeCases:
    """Tests for search action edge cases."""

    def test_search_empty_query(self, env: SearchEnvironment) -> None:
        """Search with empty query should handle gracefully."""
        env.reset()

        obs = env.step(SearchAction.make_search(""))

        # Should not crash, may return empty results
        assert obs is not None
        assert obs.action_type == "search"

    def test_search_very_long_query(self, env: SearchEnvironment) -> None:
        """Search with very long query should handle gracefully."""
        env.reset()

        long_query = "Facebook " * 100
        obs = env.step(SearchAction.make_search(long_query))

        assert obs is not None
        assert obs.action_type == "search"

    def test_search_special_characters(self, env: SearchEnvironment) -> None:
        """Search with special characters should handle gracefully."""
        env.reset()

        obs = env.step(SearchAction.make_search("test @#$%^&*() query"))

        assert obs is not None
        assert obs.action_type == "search"


class TestReadEdgeCases:
    """Tests for read action edge cases."""

    def test_read_nonexistent_chunk(self, env: SearchEnvironment) -> None:
        """Reading non-existent chunk should not crash."""
        env.reset()

        obs = env.step(SearchAction.make_read(["nonexistent_chunk_id_12345"]))

        assert obs is not None
        result = obs.action_result or {}
        assert result.get("tokens_added", 0) == 0

    def test_read_empty_list(self, env: SearchEnvironment) -> None:
        """Reading empty chunk list should handle gracefully."""
        env.reset()

        obs = env.step(SearchAction.make_read([]))

        assert obs is not None
        assert obs.action_type == "read"

    def test_multiple_read_same_chunk(self, env: SearchEnvironment) -> None:
        """Reading the same chunk twice should not duplicate it."""
        env.reset()

        # Search to get a chunk
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        results = search_obs.action_result.get("results", [])

        if results:
            chunk_id = results[0]["chunk_id"]

            # Read it twice
            obs1 = env.step(SearchAction.make_read([chunk_id]))
            initial_count = obs1.context_token_count
            initial_chunks = len(obs1.context_chunks)

            obs2 = env.step(SearchAction.make_read([chunk_id]))

            # Should not increase (chunk already in context)
            assert obs2.context_token_count == initial_count
            assert len(obs2.context_chunks) == initial_chunks


class TestPruneEdgeCases:
    """Tests for prune action edge cases."""

    def test_prune_nonexistent_chunk(self, env: SearchEnvironment) -> None:
        """Pruning non-existent chunk should report it as invalid."""
        env.reset()

        obs = env.step(SearchAction.make_prune(["nonexistent_chunk_12345"]))

        assert obs is not None
        result = obs.action_result or {}
        # Should be in invalid_ids
        assert "nonexistent_chunk_12345" in result.get("invalid_ids", [])

    def test_prune_empty_list(self, env: SearchEnvironment) -> None:
        """Pruning empty list should handle gracefully."""
        env.reset()

        obs = env.step(SearchAction.make_prune([]))

        assert obs is not None
        assert obs.action_type == "prune"

    def test_prune_not_in_context(self, env: SearchEnvironment) -> None:
        """Pruning chunk not in context should report as invalid."""
        env.reset()

        # Search to get a chunk but don't read it
        search_obs = env.step(SearchAction.make_search("Facebook"))
        assert search_obs.action_result is not None
        results = search_obs.action_result.get("results", [])

        if results:
            chunk_id = results[0]["chunk_id"]
            # Try to prune without reading first
            obs = env.step(SearchAction.make_prune([chunk_id]))

            result = obs.action_result or {}
            assert chunk_id in result.get("invalid_ids", [])


class TestAnswerEdgeCases:
    """Tests for answer action edge cases."""

    def test_answer_empty_string(self, env: SearchEnvironment) -> None:
        """Answering with empty string should handle gracefully."""
        env.reset()

        obs = env.step(SearchAction.make_answer(""))

        assert obs is not None
        assert obs.done

    def test_answer_very_long(self, env: SearchEnvironment) -> None:
        """Answering with very long string should handle gracefully."""
        env.reset()

        long_answer = "test answer " * 100
        obs = env.step(SearchAction.make_answer(long_answer))

        assert obs is not None
        assert obs.done


class TestEpisodeEdgeCases:
    """Tests for episode lifecycle edge cases."""

    def test_final_allowed_step_executes_before_termination(
        self, env_with_config: Callable[[SearchEnvConfig], SearchEnvironment]
    ) -> None:
        """The last allowed step should run, then terminate the episode."""
        config = SearchEnvConfig(max_steps=3)
        env = env_with_config(config)
        env.reset()

        env.step(SearchAction.make_search("Instagram"))
        env.step(SearchAction.make_search("WhatsApp"))
        obs = env.step(SearchAction.make_search("Facebook"))

        result = obs.action_result or {}
        assert obs.step_count == 3
        assert obs.action_type == "search"
        assert obs.done is True
        assert "results" in result
        assert result.get("step_limit_reached") is True
        assert result.get("termination_reason") == "max_steps"
        assert "final_reward" in result

    def test_step_limit_ends_episode(
        self, env_with_config: Callable[[SearchEnvConfig], SearchEnvironment]
    ) -> None:
        """Hitting step limit should end episode."""
        config = SearchEnvConfig(max_steps=3)
        env = env_with_config(config)
        env.reset()

        # Take 3 steps
        for _ in range(3):
            if not env._done:
                env.step(SearchAction.make_search("test"))

        # Episode should be done
        assert env._done

    def test_multiple_answers(self, env: SearchEnvironment) -> None:
        """Multiple answer attempts should be handled."""
        env.reset()

        # First answer ends episode
        obs1 = env.step(SearchAction.make_answer("first"))
        assert obs1.done

        # Second answer should error
        obs2 = env.step(SearchAction.make_answer("second"))
        assert obs2.action_result is not None
        assert "error" in obs2.action_result

    def test_reset_after_done(self, env: SearchEnvironment) -> None:
        """Reset after episode ends should work correctly."""
        env.reset()
        env.step(SearchAction.make_answer("done"))

        # Reset should work
        obs = env.reset()

        assert not obs.done
        assert obs.step_count == 0
        assert obs.context_token_count == 0
