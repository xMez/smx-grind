import asyncio
import threading
from pathlib import Path

import orjson
import psutil
import streamlit as st

from smx import (
    get_duckdb_processor,
    get_optimized_dataframe,
    get_scores,
)
from views.overview import overview
from views.scores import scores_view
from views.search import search

st.set_page_config(page_title="SMX DuckDB Streaming", layout="wide")


@st.cache_resource
def get_global_lock() -> threading.Lock:
    return threading.Lock()


global_lock = get_global_lock()


def get_recent_players_with_latest_play() -> list[dict[str, str]]:
    """Get recent players with their latest play times from cache state."""
    # Get cache state file
    cache_dir = Path("cache")
    state_file = cache_dir / "state.json"

    if not state_file.exists():
        return []

    try:
        with state_file.open("rb") as file:
            state = orjson.loads(file.read())
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


def player_selectbox(all_players: list[str], url_player: str) -> None:
    if url_player and url_player not in all_players and url_player != "None":
        all_players = [url_player, *all_players]

    new_selected_player = st.selectbox(
        "Select main player:",
        options=all_players if all_players else [],
        index=all_players.index(url_player) if url_player in all_players else 0,
        placeholder="Select a player or type a new name...",
        help="Select from existing players or type a new player name and press Enter",
        key="player_selector",
        accept_new_options=True,
    )

    if new_selected_player and new_selected_player != url_player:
        st.query_params.player = new_selected_player
        st.rerun()


def rivals_selectboxes(all_players: list[str], url_player: str) -> None:  # noqa: C901
    st.write("**Select rivals:**")

    for i in range(3):
        if f"rival{i + 1}" not in st.session_state:
            st.session_state[f"rival{i + 1}"] = ""

    available_rivals = [p for p in all_players if p != url_player]
    for i in range(3):
        current_rival = st.session_state[f"rival{i + 1}"]

        other_rivals = [
            st.session_state[f"rival{j + 1}"]
            for j in range(3)
            if j != i and st.session_state[f"rival{j + 1}"]
        ]

        available_for_this_slot = [r for r in available_rivals if r not in other_rivals]
        rivals_options = ["", *available_for_this_slot]

        rival_index = 0
        if current_rival and current_rival in rivals_options:
            rival_index = rivals_options.index(current_rival)

        rival = st.selectbox(
            f"Rival {i + 1}:",
            options=rivals_options,
            index=rival_index,
            placeholder=f"Select rival {i + 1} or type a new name...",
            key=f"rival{i + 1}_selector",
            label_visibility="collapsed",
            accept_new_options=True,
        )

        st.session_state[f"rival{i + 1}"] = (
            rival.strip() if rival and rival.strip() else ""
        )

    final_rivals = [""] * 3
    for i in range(3):
        rival = st.session_state[f"rival{i + 1}"]
        if rival:
            final_rivals[i] = rival

    filtered_rivals = [r for r in final_rivals if r]

    if filtered_rivals != st.session_state.rivals:
        st.session_state.rivals = filtered_rivals

        for i in range(3):
            rival_param = f"rival{i + 1}"
            if final_rivals[i]:
                st.query_params[rival_param] = final_rivals[i]
            elif rival_param in st.query_params:
                del st.query_params[rival_param]

        st.rerun()


@st.dialog(
    "🔒 Confirm Player Removal",
    width="medium",
    dismissible=True,
    on_dismiss="ignore",
)
def remove_player_dialog(player_name: str) -> None:
    """Dialog for confirming player data removal with password protection."""
    st.warning(f"⚠️ This will permanently delete ALL scores for **{player_name}**")

    # Password input
    password = st.text_input(
        "Enter password to confirm removal:",
        type="password",
        placeholder="Enter password...",
        key="remove_password_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Remove", type="secondary", use_container_width=True):
            if password == st.secrets.get["DELETE_PASSWORD"]:
                try:
                    processor = get_duckdb_processor()
                    processor.clear_cache_for_players([player_name])
                    st.success(
                        f"✅ Successfully removed all data for **{player_name}**",
                    )

                    st.session_state.redirect_to_search = True
                    st.rerun()

                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Error removing player data: {e}")
            else:
                st.error("❌ Incorrect password. Please try again.")

    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    if st.session_state.get("redirect_to_search", False):
        st.session_state.redirect_to_search = False
        st.query_params.clear()
        st.rerun()

    query_params = st.query_params
    url_player = query_params.get("player", None)
    current_page = query_params.get("page", "search")

    if "rivals" not in st.session_state:
        st.session_state.rivals = []

    if not url_player or current_page == "search":
        st.query_params.clear()
        search()
        return

    if current_page not in ("overview", "scores"):
        st.query_params.page = "overview"
        st.rerun()
        return

    url_rivals = []
    for i in range(3):
        rival_param = query_params.get(f"rival{i + 1}", None)
        if rival_param and rival_param.strip():
            url_rivals.append(rival_param.strip())

    for i in range(3):
        rival_key = f"rival{i + 1}"
        if rival_key not in st.session_state:
            if i < len(url_rivals):
                st.session_state[rival_key] = url_rivals[i]
            elif i < len(st.session_state.rivals):
                st.session_state[rival_key] = st.session_state.rivals[i]
            else:
                st.session_state[rival_key] = ""

    st.title(f"🎵 {url_player}'s Scores")

    with st.sidebar:
        if st.button("🏠 Back to Main Page", use_container_width=True, type="primary"):
            st.query_params.clear()
            st.rerun()

        st.markdown("---")

        if st.button(
            "📊 Overview",
            use_container_width=True,
            type="primary" if current_page == "overview" else "secondary",
        ):
            st.query_params.page = "overview"
            st.rerun()

        if st.button(
            "🎯 Scores",
            use_container_width=True,
            type="primary" if current_page == "scores" else "secondary",
        ):
            st.query_params.page = "scores"
            st.rerun()

        st.markdown("---")

        recent_players_data = get_recent_players_with_latest_play()
        all_players = [p["player"] for p in recent_players_data]

        player_selectbox(all_players, url_player)
        rivals_selectboxes(all_players, url_player)

        if st.button("🗑️ Clear Rivals", use_container_width=True):
            for i in range(3):
                st.session_state[f"rival{i + 1}"] = ""
            st.session_state.rivals = []

            for i in range(3):
                rival_param = f"rival{i + 1}"
                if rival_param in st.query_params:
                    del st.query_params[rival_param]

            st.rerun()

        st.markdown("---")

        timespan_options = {
            "ALL": None,
            "1 year": 365,
            "6 months": 180,
            "3 months": 90,
            "1 month": 30,
            "This year": "this_year",
            "This month": "this_month",
        }

        # Initialize timespan in session state
        if "selected_timespan" not in st.session_state:
            st.session_state.selected_timespan = "ALL"

        selected_timespan = st.selectbox(
            "Filter by timespan:",
            options=list(timespan_options.keys()),
            index=list(timespan_options.keys()).index(
                st.session_state.selected_timespan,
            ),
            help="Filter data to show only scores from the selected time period",
        )

        # Update session state if timespan changed
        if selected_timespan != st.session_state.selected_timespan:
            st.session_state.selected_timespan = selected_timespan
            st.rerun()

        difficulty_options = {
            "Wild 19": "wild 19",
            "Wild 20": "wild 20",
            "Wild 21": "wild 21",
            "Wild 22": "wild 22",
            "Wild 23": "wild 23",
            "Wild 24": "wild 24",
            "Wild 25": "wild 25",
            "Wild 26": "wild 26",
            "Wild 27": "wild 27",
        }

        if "selected_difficulties" not in st.session_state:
            st.session_state.selected_difficulties = list(difficulty_options.keys())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("All", use_container_width=True, type="secondary"):
                st.session_state.selected_difficulties = list(difficulty_options.keys())
                st.rerun()
        with col2:
            if st.button("None", use_container_width=True, type="secondary"):
                st.session_state.selected_difficulties = []
                st.rerun()

        selected_difficulties = st.pills(
            "Filter by difficulty:",
            options=list(difficulty_options.keys()),
            default=st.session_state.selected_difficulties,
            selection_mode="multi",
            help="Filter data to show only scores from the selected difficulties.",
        )

        if selected_difficulties != st.session_state.selected_difficulties:
            st.session_state.selected_difficulties = selected_difficulties
            st.rerun()

        if st.button("🔄 Force Full Refresh", use_container_width=True):
            processor = get_duckdb_processor()
            all_players = [url_player, *st.session_state.rivals]
            processor.clear_cache_for_players(all_players)
            st.rerun()

        if st.button("🗑️ Remove Player Data", use_container_width=True):
            remove_player_dialog(url_player)

        st.markdown("---")

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_players = [url_player, *st.session_state.rivals]

    try:
        for i, player in enumerate(all_players):
            status_text.write(
                f"🔄 Processing {player} ({i + 1}/{len(all_players)})...",
            )

            with global_lock:
                asyncio.run(
                    get_scores(
                        player,
                        status_text=status_text,
                        progress_bar=progress_bar,
                    ),
                )

        df = get_optimized_dataframe(
            all_players,
        )

        if df is None or df.height == 0:
            st.warning("No scores found for the selected players")
            return

        selected_difficulty_values = [
            difficulty_options[d] for d in selected_difficulties
        ]

        if current_page == "scores":
            scores_view(
                url_player,
                selected_difficulty_values,
            )
        else:
            overview(
                url_player,
                df,
                timespan_options[selected_timespan],
                selected_difficulty_values,
            )
    finally:
        progress_bar.empty()
        status_text.empty()


if __name__ == "__main__":
    main()
