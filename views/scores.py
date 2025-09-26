import polars as pl
import streamlit as st
from pandas.io.formats.style import Styler

from duckdb_utils import get_best_scores_duckdb, get_score_categories_duckdb
from utils import filter_by_timespan


def create_comparison_dataframe(
    main_player: str,
    rivals: list[str],
    best_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Create comparison dataframe with main player scores vs rivals."""
    comparison_rows = []
    main_player_songs = best_scores.filter(pl.col("player") == main_player)

    for song_row in main_player_songs.iter_rows(named=True):
        song = song_row["song"]
        main_score = song_row["best_score"]
        main_date = song_row["latest_date"]
        difficulty = song_row["difficulty"]

        row = {
            "song": song,
            "difficulty": difficulty,
            "main_score": main_score,
            "main_date": main_date,
        }
        for rival in rivals:
            rival_scores = best_scores.filter(
                (pl.col("song") == song) & (pl.col("player") == rival),
            )

            if rival_scores.height > 0:
                rival_score = rival_scores["best_score"][0]
                rival_date = rival_scores["latest_date"][0]
                delta = int(main_score - rival_score)

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
    date_columns = ["main_date"] + [f"{rival}_date" for rival in rivals]
    comparison_df = comparison_df.with_columns(
        [
            pl.col(col).str.slice(0, 10).alias(f"{col}_clean")
            for col in date_columns
            if col in comparison_df.columns
        ],
    )

    base_columns = ["song", "difficulty", "main_score"]
    delta_columns = [f"{rival}_delta" for rival in rivals]
    date_columns = ["main_date_clean"] + [f"{rival}_date_clean" for rival in rivals]

    all_columns = base_columns + delta_columns + date_columns
    available_columns = [col for col in all_columns if col in comparison_df.columns]

    display_df = comparison_df.select(available_columns)

    rename_dict = {}
    for col in display_df.columns:
        clean_name = col.replace("_", " ").title()
        clean_name = clean_name.replace(" Clean", "")
        rename_dict[col] = clean_name

    if rename_dict:
        display_df = display_df.rename(rename_dict)

    return display_df


def render_score_breakdown(
    main_player: str,
    rivals: list[str],
    timespan_value: str,
    difficulty_values: list[str] | None,
) -> None:
    """Render the score category breakdown section."""
    st.markdown("### Score Category Breakdown")

    all_players = [main_player, *rivals]
    categories_data = get_score_categories_duckdb(
        timespan_value=timespan_value,
        difficulty_values=difficulty_values,
        players=all_players,
    )

    breakdown_data = []
    for player in all_players:
        if player in categories_data:
            breakdown_data.append({"Player": player, **categories_data[player]})
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
        height=(display_df.height + 1) * 35 + 3,
    )


def scores_view(
    main_player: str,
    df: pl.DataFrame,
    timespan_value: str,
    difficulty_values: list[str] | None = None,
) -> None:
    """Show best scores per song for main player
    with delta to rivals' best scores as a dataframe."""

    if df is None or df.height == 0:
        st.warning("No scores found for the selected players")
        return

    filtered_df = filter_by_timespan(df, timespan_value)

    if difficulty_values and len(difficulty_values) > 0:
        filtered_df = filtered_df.filter(pl.col("difficulty").is_in(difficulty_values))
    else:
        filtered_df = filtered_df.filter(pl.lit(value=False))

    if filtered_df.height == 0:
        st.warning("No scores found for the selected time period and difficulty")
        return

    all_players = [p for p in filtered_df["player"].unique() if p]
    rivals = [p for p in all_players if p != main_player]
    if rivals:
        st.write(f"**Comparing against:** {', '.join(rivals)}")

    best_scores = get_best_scores_duckdb(
        timespan_value=timespan_value,
        difficulty_values=difficulty_values,
        players=all_players,
    )

    comparison_df = create_comparison_dataframe(main_player, rivals, best_scores)

    if comparison_df.height == 0:
        st.warning("No comparison data found")
        return

    display_df = prepare_display_dataframe(comparison_df, rivals)

    st.subheader("Best Scores Comparison")

    render_score_breakdown(main_player, rivals, timespan_value, difficulty_values)

    st.markdown("---")

    render_scores_table(display_df)
