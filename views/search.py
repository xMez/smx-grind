from pathlib import Path

import orjson
import streamlit as st


def get_changelog_content() -> str:
    """Read changelog content from CHANGELOG.md file."""
    changelog_path = Path("CHANGELOG.md")
    if changelog_path.exists():
        try:
            return changelog_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return "Changelog not available."
    return "Changelog not found."


def get_recent_players_with_latest_play() -> list[dict[str, str]]:
    """Get recent players with their latest play times from cache state."""
    cache_dir = Path("cache")
    state_file = cache_dir / "state.json"

    if not state_file.exists():
        return []

    try:
        with state_file.open("r") as f:
            state = orjson.loads(f.read())
            players_data = state.get("players", {})

        recent_players = []
        for player, player_state in players_data.items():
            latest_timestamp = player_state.get("latest_timestamp")
            if latest_timestamp:
                recent_players.append(
                    {"player": player, "latest_play": latest_timestamp},
                )
        recent_players.sort(key=lambda x: x["latest_play"], reverse=True)
    except (orjson.JSONDecodeError, KeyError):
        return []
    else:
        return recent_players


def search() -> None:
    # Center the search interface
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.subheader("Search for a Player")

        search_player = st.text_input(
            "Enter SMX username:",
            placeholder="e.g., xMez, ARMN4LFE...",
            help="Enter a player's SMX username to view their scores",
            key="search_input",
        )

        def search_for_player() -> None:
            if search_player and search_player.strip():
                st.query_params.player = search_player.strip()
                st.query_params.page = "overview"
                st.rerun()

        if st.button("🔍 Search Player", use_container_width=True, type="primary"):
            search_for_player()

        # Also search when Enter is pressed (using session state to detect changes)
        if "last_search" not in st.session_state:
            st.session_state.last_search = ""

        if (
            search_player != st.session_state.last_search
            and search_player
            and search_player.strip()
        ):
            st.session_state.last_search = search_player
            search_for_player()

        recent_players = get_recent_players_with_latest_play()
        if recent_players:
            st.markdown("---")
            st.subheader("Recent Players")
            for player_data in recent_players:
                player = player_data["player"]
                latest_play = player_data["latest_play"]
                st.markdown(
                    f"""<a href="/?player={player}&page=overview" target="_self">**{player}**</br>*:small[Last played: {latest_play}]*</a>""",  # noqa: E501
                    unsafe_allow_html=True,
                )

        # Changelog section
        st.markdown("---")
        with st.expander("📋 Changelog", expanded=False):
            changelog_content = get_changelog_content()
            st.markdown(changelog_content)
