import plotly.graph_objects as go
import polars as pl
import streamlit as st

from duckdb_utils import get_worst_songs_duckdb
from utils import filter_by_timespan


def get_ordered_players(selected_player: str, unique_players: list[str]) -> list[str]:
    """Get players ordered with selected_player first,
    then rivals, then remaining players."""
    ordered_players = []

    if selected_player in unique_players:
        ordered_players.append(selected_player)

    # Add rivals in order
    for i in range(3):
        rival_key = f"rival{i + 1}"
        rival_name = st.session_state.get(rival_key)
        if (
            rival_name
            and rival_name in unique_players
            and rival_name not in ordered_players
        ):
            ordered_players.append(rival_name)

    # Add any remaining players not in slots (fallback)
    for player in unique_players:
        if player not in ordered_players:
            ordered_players.append(player)

    return ordered_players


def get_player_colors(
    selected_player: str,
    ordered_players: list[str],
) -> dict[str, str]:
    """Get color mapping for players with consistent color scheme."""
    player_colors = {
        selected_player: "rgba(147, 51, 234, 0.8)",  # Vibrant purple
    }

    # Rival colors based on slot position
    rival_colors = [
        "rgba(59, 130, 246, 0.8)",  # rival1 = blue
        "rgba(245, 158, 11, 0.8)",  # rival2 = gold
        "rgba(239, 68, 68, 0.8)",  # rival3 = red
    ]

    # Map rivals to their slot positions from session state
    for i in range(3):
        rival_key = f"rival{i + 1}"
        rival_name = st.session_state.get(rival_key)
        if rival_name and rival_name in ordered_players and i < len(rival_colors):
            player_colors[rival_name] = rival_colors[i]

    return player_colors


def calculate_rolling_average(
    df: pl.DataFrame,
    value_col: str,
    window_size: str = "14d",
) -> pl.DataFrame:
    """Calculate 14-day rolling average using Polars rolling_mean_by."""
    if df.height == 0:
        return pl.DataFrame(
            {
                "player": [],
                "date": [],
                value_col: [],
                f"rolling_avg_{value_col}": [],
            },
        )

    # Ensure datetime
    df = df.with_columns(
        pl.col("date").str.to_datetime().alias("date")
        if df["date"].dtype not in [pl.Datetime, pl.Date]
        else pl.col("date"),
    )

    return df.sort(["player", "date"]).with_columns(
        pl.col(value_col)
        .rolling_mean_by("date", window_size, min_samples=1, closed="both")
        .over("player")
        .alias(f"rolling_avg_{value_col}"),
    )


def apply_plotly_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    barmode: str | None = None,
) -> None:
    """Apply common plotly layout settings."""
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 50, "r": 50, "t": 120, "b": 50},
        xaxis={
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128,128,128,0.2)",
            "rangemode": "normal",
            "automargin": True,
        },
        yaxis={
            "title": y_title,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128,128,128,0.2)",
            "automargin": True,
        },
    )

    if barmode:
        fig.update_layout(barmode=barmode)


def render_worst_songs_for_player(player: str, difficulty: str) -> None:
    """Render worst songs for a single player using DuckDB."""
    worst_songs = get_worst_songs_duckdb(player=player, difficulty=difficulty, limit=10)

    if worst_songs.height > 0:
        st.write(f"**{player}**")

        worst_scores_pd = worst_songs.to_pandas()

        for _, row in worst_scores_pd.iterrows():
            st.write(
                f"**{row['score']:,.0f}** | {row['song']}  \n"
                f"<small>{row['created_at'][:10]}</small>",
                unsafe_allow_html=True,
            )
    else:
        st.write(f"**{player}**")
        st.write("No scores for this difficulty")


def render_worst_songs_section(
    selected_player: str,
    df: pl.DataFrame,
    difficulty_values: list[str] | None = None,
) -> None:
    """Render the worst songs section for all players and difficulties."""
    if len(df) > 0 and "difficulty" in df.columns:
        st.subheader("10 Worst Scoring Songs per Player per Difficulty (All Time)")

        all_players_in_data = df["player"].unique()
        if selected_player in all_players_in_data:
            rivals_in_order = []
            for i in range(3):
                rival_key = f"rival{i + 1}"
                rival_name = st.session_state.get(rival_key)
                if rival_name and rival_name in all_players_in_data:
                    rivals_in_order.append(rival_name)
            players = [selected_player, *rivals_in_order]
        else:
            players = all_players_in_data

        if difficulty_values and len(difficulty_values) > 0:
            difficulties = [
                d for d in df["difficulty"].unique() if d in difficulty_values
            ]
        else:
            difficulties = []

        if difficulties:
            difficulty_tabs = st.tabs([f"**{diff}**" for diff in sorted(difficulties)])

            for tab_idx, difficulty in enumerate(sorted(difficulties)):
                with difficulty_tabs[tab_idx]:
                    player_cols = st.columns(len(players))

                    for col_idx, player in enumerate(players):
                        with player_cols[col_idx]:
                            render_worst_songs_for_player(player, difficulty)
        else:
            st.info(
                "No difficulties selected. Please select difficulties from the sidebar to view worst scoring songs.",  # noqa: E501
            )


def render_songs_chart(selected_player: str, filtered_df: pl.DataFrame) -> None:
    """Render the songs played chart."""
    df_with_date = filtered_df.with_columns(
        pl.col("created_at")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ")
        .dt.date()
        .alias("date"),
    )

    daily_songs = (
        df_with_date.group_by(["player", "date"])
        .agg(pl.col("song").n_unique().alias("songs_played"))
        .sort(["player", "date"])
    )

    songs_rolling_df = calculate_rolling_average(
        daily_songs,
        "songs_played",
    )

    fig3 = go.Figure()

    songs_unique_players = songs_rolling_df["player"].unique().to_list()
    ordered_players = get_ordered_players(selected_player, songs_unique_players)
    player_colors = get_player_colors(selected_player, ordered_players)

    if songs_rolling_df.height > 0:
        for i, player in enumerate(ordered_players):
            player_data = songs_rolling_df.filter(pl.col("player") == player)
            if player_data.height > 0:
                color = player_colors.get(
                    player,
                    f"rgba({100 + i * 50}, {100 + i * 30}, {200 - i * 40}, 0.8)",
                )  # Fallback color
                fig3.add_trace(
                    go.Scatter(
                        x=player_data["date"].to_list(),
                        y=player_data["rolling_avg_songs_played"].to_list(),
                        mode="lines+markers",
                        name=f"{player} - Songs",
                        line={"width": 3, "color": color},
                        marker={"size": 6, "color": color},
                        yaxis="y",
                    ),
                )

    apply_plotly_layout(
        fig3,
        "14-Day Rolling Average Songs Played",
        "Date",
        "Songs Played",
    )

    st.plotly_chart(
        fig3,
        use_container_width=True,
        key="songs_played_chart",
    )


def render_scores_chart(selected_player: str, filtered_df: pl.DataFrame) -> None:
    """Render the scores chart."""
    df_with_date = filtered_df.with_columns(
        pl.col("created_at")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ")
        .dt.date()
        .alias("date"),
    )

    daily_avg = (
        df_with_date.group_by(["player", "date"])
        .agg(pl.col("score").mean().alias("avg_score"))
        .sort(["player", "date"])
    )

    scores_rolling_df = calculate_rolling_average(
        daily_avg,
        "avg_score",
    )

    fig = go.Figure()

    unique_players = scores_rolling_df["player"].unique().to_list()
    ordered_players = get_ordered_players(selected_player, unique_players)
    player_colors = get_player_colors(selected_player, ordered_players)

    if scores_rolling_df.height > 0:
        for i, player in enumerate(ordered_players):
            player_data = scores_rolling_df.filter(pl.col("player") == player)
            if player_data.height > 0:
                color = player_colors.get(
                    player,
                    f"rgba({100 + i * 50}, {100 + i * 30}, {200 - i * 40}, 0.8)",
                )  # Fallback color
                fig.add_trace(
                    go.Scatter(
                        x=player_data["date"].to_list(),
                        y=player_data["rolling_avg_avg_score"].to_list(),
                        mode="lines+markers",
                        name=f"{player} - Score",
                        line={"width": 3, "color": color},
                        marker={"size": 6, "color": color},
                        yaxis="y",
                    ),
                )

    apply_plotly_layout(
        fig,
        "14-Day Rolling Average Score",
        "Date",
        "Average Score",
    )

    st.plotly_chart(fig, use_container_width=True, key="time_series_chart")


def render_difficulty_chart(selected_player: str, filtered_df: pl.DataFrame) -> None:
    """Render the difficulty chart."""
    difficulty_avg = (
        filtered_df.group_by(["player", "difficulty"])
        .agg(pl.col("score").mean().alias("avg_score"))
        .sort(["player", "difficulty"])
    )

    difficulty_avg_pd = difficulty_avg.to_pandas()

    fig2 = go.Figure()

    difficulty_unique_players = difficulty_avg_pd["player"].unique()
    ordered_players = get_ordered_players(selected_player, difficulty_unique_players)
    player_colors = get_player_colors(selected_player, ordered_players)

    # Add bars for each player with consistent colors
    for i, player in enumerate(ordered_players):
        player_data = difficulty_avg_pd[difficulty_avg_pd["player"] == player]
        color = player_colors.get(
            player,
            f"rgba({100 + i * 50}, {100 + i * 30}, {200 - i * 40}, 0.8)",
        )  # Fallback color
        fig2.add_trace(
            go.Bar(
                x=player_data["difficulty"],
                y=player_data["avg_score"],
                name=player,
                marker_color=color,
                marker_line_width=1,
                marker_line_color="white",
            ),
        )

    apply_plotly_layout(
        fig2,
        "Average Score by Difficulty",
        "Difficulty",
        "Average Score",
        barmode="group",
    )

    st.plotly_chart(fig2, use_container_width=True, key="difficulty_chart")


def render_graphs(selected_player: str, filtered_df: pl.DataFrame) -> None:
    """Render the graphs with the given filtered data."""
    if len(filtered_df) > 0:
        col1, col2, col3 = st.columns(3)

        if "created_at" in filtered_df.columns:
            with col1:
                render_songs_chart(selected_player, filtered_df)

        if "created_at" in filtered_df.columns:
            with col2:
                render_scores_chart(selected_player, filtered_df)

        if "difficulty" in filtered_df.columns:
            with col3:
                render_difficulty_chart(selected_player, filtered_df)


def overview(
    selected_player: str,
    df: pl.DataFrame,
    timespan_value: str,
    difficulty_values: list[str] | None = None,
) -> None:
    if df.height > 0:
        filtered_df = filter_by_timespan(df, timespan_value).sort(
            "created_at",
            descending=True,
        )

        if difficulty_values and len(difficulty_values) > 0:
            filtered_df = filtered_df.filter(
                pl.col("difficulty").is_in(difficulty_values),
            )
        else:
            filtered_df = filtered_df.filter(pl.lit(value=False))

        if st.session_state.rivals:
            players_text = f"{selected_player} vs {', '.join(st.session_state.rivals)}"
        else:
            players_text = selected_player

        st.write(
            f"Found {len(df)} total scores from {players_text}"
            f" ({len(filtered_df)} in selected time period for graphs)",
        )

        if "created_at" in filtered_df.columns:
            latest_date = filtered_df["created_at"].max()
            st.metric("Latest Score", latest_date)

        render_graphs(selected_player, filtered_df)

        render_worst_songs_section(selected_player, df, difficulty_values)

        st.subheader("Detailed Scores")
        st.dataframe(filtered_df, width="stretch")
    else:
        st.warning("No scores found for the selected players")
