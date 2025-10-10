import polars as pl
import streamlit as st
from pandas.io.formats.style import Styler

from smx import get_highscores
from utils import get_ordered_players

# Score category thresholds
SCORE_100K = 100000
SCORE_99K = 99000
SCORE_97K = 97000
SCORE_90K = 90000
SCORE_80K = 80000


def create_comparison_dataframe(
    player: str,
    rivals: list[str],
    highscores: pl.DataFrame,
) -> pl.DataFrame:
    """Create comparison dataframe with main player scores vs rivals."""
    comparison_rows = []
    songs = highscores.filter(pl.col("player") == player)

    for song_row in songs.iter_rows(named=True):
        song = song_row["song"]
        score = song_row["score"]
        date = song_row["created_at"]
        difficulty = song_row["difficulty"]

        row = {
            "song": song,
            "difficulty": difficulty,
            "score": score,
            "date": date,
        }
        for rival in rivals:
            rival_scores = highscores.filter(
                (pl.col("song") == song) & (pl.col("player") == rival),
            )

            if rival_scores.height > 0:
                rival_score = rival_scores["score"][0]
                rival_date = rival_scores["created_at"][0]
                delta = int(score - rival_score)

                row[f"{rival}_score"] = rival_score
                row[f"{rival}_date"] = rival_date
                row[f"{rival}_delta"] = delta
            else:
                row[f"{rival}_score"] = None
                row[f"{rival}_date"] = None
                row[f"{rival}_delta"] = None

        comparison_rows.append(row)

    return pl.DataFrame(comparison_rows)


def prepare_display_dataframe(
    comparison_df: pl.DataFrame,
    rivals: list[str],
) -> pl.DataFrame:
    """Prepare the dataframe for display with proper column formatting."""
    date_columns = ["date"] + [f"{rival}_date" for rival in rivals]
    comparison_df = comparison_df.with_columns(
        [
            pl.col(col).str.slice(0, 10).alias(f"{col}_clean")
            for col in date_columns
            if col in comparison_df.columns
        ],
    )

    base_columns = ["song", "difficulty", "score"]
    delta_columns = [f"{rival}_delta" for rival in rivals]
    date_columns = ["date_clean"] + [f"{rival}_date_clean" for rival in rivals]

    all_columns = base_columns + delta_columns + date_columns
    available_columns = [col for col in all_columns if col in comparison_df.columns]

    display_df = comparison_df.select(available_columns)

    rename_dict = {}
    for col in display_df.columns:
        clean_name = col.replace("_", " ")
        clean_name = clean_name.replace(" Clean", "")
        rename_dict[col] = clean_name

    if rename_dict:
        display_df = display_df.rename(rename_dict)

    return display_df


def render_score_breakdown(
    current_player: str,
    rivals: list[str],
    highscores: pl.DataFrame,
    difficulty_values: list[str] | None,
) -> None:
    """Render the score category breakdown section."""
    st.markdown("### Score Category Breakdown")

    all_players = [current_player, *rivals]

    # Filter by difficulty if specified
    filtered_df = highscores
    if difficulty_values and len(difficulty_values) > 0:
        filtered_df = filtered_df.filter(
            pl.col("difficulty").is_in(difficulty_values),
        )

    # Calculate score categories for each player
    breakdown_data = []
    for player in all_players:
        player_scores = filtered_df.filter(pl.col("player") == player)

        if player_scores.height > 0:
            scores = player_scores["score"].to_list()

            # Count scores in each category
            count_100k = sum(1 for s in scores if s >= SCORE_100K)
            count_99k = sum(1 for s in scores if SCORE_99K <= s < SCORE_100K)
            count_97k = sum(1 for s in scores if SCORE_97K <= s < SCORE_99K)
            count_90k = sum(1 for s in scores if SCORE_90K <= s < SCORE_97K)
            count_80k = sum(1 for s in scores if SCORE_80K <= s < SCORE_90K)
            count_under_80k = sum(1 for s in scores if s < SCORE_80K)
            total = len(scores)

            breakdown_data.append(
                {
                    "Player": player,
                    "100k": count_100k,
                    "99k+": count_99k,
                    "97k+": count_97k,
                    "90k+": count_90k,
                    "80k+": count_80k,
                    "<80k": count_under_80k,
                    "Total": total,
                },
            )
        else:
            breakdown_data.append(
                {
                    "Player": player,
                    "100k": 0,
                    "99k+": 0,
                    "97k+": 0,
                    "90k+": 0,
                    "80k+": 0,
                    "<80k": 0,
                    "Total": 0,
                },
            )

    if breakdown_data:
        breakdown_df = pl.DataFrame(breakdown_data)

        column_order = [
            "Player",
            "100k",
            "99k+",
            "97k+",
            "90k+",
            "80k+",
            "<80k",
            "Total",
        ]
        available_columns = [col for col in column_order if col in breakdown_df.columns]
        breakdown_df = breakdown_df.select(available_columns).sort(
            "100k",
            "99k+",
            "97k+",
            "90k+",
            "80k+",
            "<80k",
            "Total",
            descending=True,
        )

        breakdown_pandas = breakdown_df.to_pandas()

        def style_breakdown(val: int, col_name: str) -> str:
            if val == 0:
                return ""

            color_map = {
                "100k": "background-color: rgba(147, 51, 234, 0.8); color: white",  # Vibrant purple  # noqa: E501
                "99k+": "background-color: rgba(59, 130, 246, 0.8); color: white",  # Vibrant blue  # noqa: E501
                "97k+": "background-color: rgba(245, 158, 11, 0.8); color: white",  # Vibrant gold  # noqa: E501
                "90k+": "background-color: rgba(34, 197, 94, 0.8); color: white",  # Vibrant green  # noqa: E501
                "80k+": "background-color: rgba(34, 197, 94, 0.5); color: white",  # Vibrant green 50%  # noqa: E501
                "<80k": "background-color: rgba(239, 68, 68, 0.8); color: white",  # Vibrant red  # noqa: E501
            }
            return color_map.get(col_name, "")

        styled_breakdown = breakdown_pandas.style
        for col in breakdown_pandas.columns:
            if col != "Player":
                styled_breakdown = styled_breakdown.map(
                    lambda x, col_name=col: style_breakdown(x, col_name),
                    subset=[col],
                )

        st.dataframe(styled_breakdown, hide_index=True, width="stretch")


def get_score_color(score: int) -> str:
    """Get color styling for a score value."""
    if score >= 100000:  # noqa: PLR2004
        return "background-color: rgba(147, 51, 234, 0.8)"  # Vibrant purple
    if score >= 99000:  # noqa: PLR2004
        return "background-color: rgba(59, 130, 246, 0.8)"  # Vibrant blue
    if score >= 97000:  # noqa: PLR2004
        return "background-color: rgba(245, 158, 11, 0.8)"  # Vibrant gold
    if score >= 93000:  # noqa: PLR2004
        return "background-color: rgba(34, 197, 94, 0.8)"  # Vibrant green
    if score >= 85000:  # noqa: PLR2004
        intensity = (score - 85000) / 8000  # 0-1 scale from 80k to 90k
        return f"background-color: rgba(34, 197, 94, {0.3 + intensity * 0.5})"
    # <80k = red gradient all the way down to 0
    intensity = score / 85000  # 0-1 scale from 0 to 80k
    return f"background-color: rgba(239, 68, 68, {0.3 + intensity * 0.5})"


def get_delta_color(score: int) -> str:
    """Get color styling for a delta value."""
    if score > 0:
        intensity = min(abs(score) / 10000, 1)
        alpha = 0.3 + intensity * 0.5
        return f"background-color: rgba(34, 197, 94, {alpha})"
    if score < 0:
        intensity = min(abs(score) / 10000, 1)
        alpha = 0.3 + intensity * 0.5
        return f"background-color: rgba(239, 68, 68, {alpha})"
    return "background-color: rgba(128, 128, 128, 0.2)"  # Tie: gray


def style_scores(val: int, *, is_delta: bool = False) -> str:
    """Apply styling to score or delta values."""
    if val is None or val == "":
        return ""

    try:
        if is_delta:
            return get_delta_color(val)
        return get_score_color(val)
    except (ValueError, TypeError):
        return ""


def prepare_styled_dataframe(display_df: pl.DataFrame) -> Styler:
    """Prepare and style the pandas dataframe for display."""
    pandas_df = display_df.to_pandas()

    # Ensure delta columns remain as integers in pandas
    for col in pandas_df.columns:
        if "Delta" in col:
            pandas_df[col] = pandas_df[col].astype("Int64")  # Nullable integer type

    styled_df = pandas_df.style
    for col in pandas_df.columns:
        if "Score" in col:
            styled_df = styled_df.map(
                lambda x: style_scores(x, is_delta=False),
                subset=[col],
            )
        elif "Delta" in col:
            styled_df = styled_df.map(
                lambda x: style_scores(x, is_delta=True),
                subset=[col],
            )

    return styled_df


def render_scores_table(display_df: pl.DataFrame) -> None:
    """Render the styled scores comparison table."""
    styled_df = prepare_styled_dataframe(display_df)

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        height=(min(display_df.height, 25) + 1) * 35 + 3,
    )


def scores_view(
    player: str,
    difficulty_values: list[str] | None = None,
) -> None:
    """Show best scores per song for main player
    with delta to rivals' best scores as a dataframe."""

    highscores_df = get_highscores()
    if highscores_df.height == 0:
        st.warning(
            "No highscores found in the database. "
            "Please fetch scores for players first.",
        )
        return

    unique_players = highscores_df["player"].unique().to_list()
    ordered_players = get_ordered_players(player, unique_players)

    rivals = [p for p in ordered_players if p != player]

    highscores_df = highscores_df.filter(pl.col("player").is_in(ordered_players))

    if highscores_df.height == 0:
        st.warning(
            "No highscores found for the selected players. "
            "Please fetch scores for players first.",
        )
        return

    if difficulty_values and len(difficulty_values) > 0:
        highscores_df = highscores_df.filter(
            pl.col("difficulty").is_in(difficulty_values),
        )
    else:
        st.warning("Please select at least one difficulty to view scores")
        return

    if highscores_df.height == 0:
        st.warning("No scores found for the selected difficulty")
        return

    if rivals:
        st.write(f"**Comparing against:** {', '.join(rivals)}")

    comparison_df = create_comparison_dataframe(player, rivals, highscores_df)

    if comparison_df.height == 0:
        st.warning("No comparison data found")
        return

    display_df = prepare_display_dataframe(comparison_df, rivals)

    st.subheader("Best Scores Comparison")

    render_score_breakdown(player, rivals, highscores_df, difficulty_values)

    st.markdown("---")

    render_scores_table(display_df)
