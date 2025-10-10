import polars as pl
import streamlit as st

from duckdb_utils import filter_by_timespan_duckdb, get_duckdb_connection


def filter_by_timespan(
    df: pl.DataFrame,
    timespan_value: str | None = None,
) -> pl.DataFrame:
    """Filter DataFrame by timespan using DuckDB for better performance."""
    if timespan_value is None or timespan_value == "all":
        return df

    conn = get_duckdb_connection()

    timespan_filter = filter_by_timespan_duckdb(timespan_value)

    if not timespan_filter:
        return df

    pandas_df = df.to_pandas()
    conn.register("temp_df", pandas_df)

    query = f"SELECT * FROM temp_df {timespan_filter}"  # noqa: S608
    result = conn.execute(query).fetchall()

    conn.unregister("temp_df")

    if not result:
        return pl.DataFrame()

    columns = [desc[0] for desc in conn.description]
    return pl.DataFrame(result, schema=columns, orient="row")


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

    return ordered_players
