from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl

from smx import get_duckdb_processor


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection to the scores database."""

    processor = get_duckdb_processor()
    return processor.conn


def filter_by_timespan_duckdb(
    timespan_value: str | None = None,
) -> str:
    """
    Generate DuckDB SQL for filtering by timespan.

    Args:
        table_name: Name of the table to filter
        timespan_value: Timespan filter value

    Returns:
        SQL WHERE clause for timespan filtering
    """
    if timespan_value is None or timespan_value == "all":
        return ""

    if timespan_value == "this_year":
        current_year = datetime.now(UTC).year
        return f"WHERE EXTRACT(YEAR FROM created_at::TIMESTAMP) = {current_year}"

    if timespan_value == "this_month":
        now = datetime.now(UTC)
        current_year = now.year
        current_month = now.month
        return f"WHERE EXTRACT(YEAR FROM created_at::TIMESTAMP) = {current_year} AND EXTRACT(MONTH FROM created_at::TIMESTAMP) = {current_month}"  # noqa: E501

    # Handle numeric timespan values (days)
    try:
        days = int(timespan_value)
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""
    else:
        return f"WHERE created_at::TIMESTAMP >= '{cutoff_str}'"


def filter_by_difficulty_duckdb(difficulty_values: list[str] | None = None) -> str:
    """
    Generate DuckDB SQL for filtering by difficulty.

    Args:
        difficulty_values: List of difficulty values to filter by

    Returns:
        SQL WHERE clause for difficulty filtering
    """
    if difficulty_values is None:
        return ""
    if len(difficulty_values) == 0:
        return "WHERE 1=0"

    # Escape single quotes in difficulty values
    escaped_values = []
    for d in difficulty_values:
        escaped_d = d.replace("'", "''")
        escaped_values.append(f"'{escaped_d}'")
    values_str = ", ".join(escaped_values)
    return f"WHERE difficulty IN ({values_str})"


def get_best_scores_duckdb(
    table_name: str = "scores",
    timespan_value: str | None = None,
    difficulty_values: list[str] | None = None,
    players: list[str] | None = None,
) -> pl.DataFrame:
    """
    Get best scores per song per player using DuckDB.

    Args:
        table_name: Name of the table to query
        timespan_value: Timespan filter value
        difficulty_values: List of difficulty values to filter by
        players: List of players to filter by

    Returns:
        Polars DataFrame with best scores
    """
    conn = get_duckdb_connection()

    where_conditions = []

    timespan_filter = filter_by_timespan_duckdb(timespan_value)
    if timespan_filter:
        where_conditions.append(timespan_filter.replace("WHERE ", ""))

    difficulty_filter = filter_by_difficulty_duckdb(difficulty_values)
    if difficulty_filter:
        where_conditions.append(difficulty_filter.replace("WHERE ", ""))

    if players and len(players) > 0:
        escaped_players = []
        for p in players:
            escaped_p = p.replace("'", "''")
            escaped_players.append(f"'{escaped_p}'")
        players_str = ", ".join(escaped_players)
        where_conditions.append(f"player IN ({players_str})")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    query = f"""
    SELECT
        song,
        player,
        MAX(score) as best_score,
        MAX(created_at) as latest_date,
        FIRST(difficulty) as difficulty
    FROM {table_name}
    {where_clause}
    GROUP BY song, player
    ORDER BY song, player
    """  # noqa: S608

    result = conn.execute(query).fetchall()
    if not result:
        return pl.DataFrame(
            {
                "song": [],
                "player": [],
                "best_score": [],
                "latest_date": [],
                "difficulty": [],
            },
        )

    return pl.DataFrame(
        result,
        schema=["song", "player", "best_score", "latest_date", "difficulty"],
        orient="row",
    )


def get_daily_aggregates_duckdb(
    table_name: str = "scores",
    timespan_value: str | None = None,
    difficulty_values: list[str] | None = None,
    players: list[str] | None = None,
    aggregate_type: str = "scores",
) -> pl.DataFrame:
    """
    Get daily aggregates (scores or songs) using DuckDB.

    Args:
        table_name: Name of the table to query
        timespan_value: Timespan filter value
        difficulty_values: List of difficulty values to filter by
        players: List of players to filter by
        aggregate_type: Type of aggregation ("scores" or "songs")

    Returns:
        Polars DataFrame with daily aggregates
    """
    conn = get_duckdb_connection()

    where_conditions = []

    timespan_filter = filter_by_timespan_duckdb(timespan_value)
    if timespan_filter:
        where_conditions.append(timespan_filter.replace("WHERE ", ""))

    difficulty_filter = filter_by_difficulty_duckdb(difficulty_values)
    if difficulty_filter:
        where_conditions.append(difficulty_filter.replace("WHERE ", ""))

    if players and len(players) > 0:
        escaped_players = []
        for p in players:
            escaped_p = p.replace("'", "''")
            escaped_players.append(f"'{escaped_p}'")
        players_str = ", ".join(escaped_players)
        where_conditions.append(f"player IN ({players_str})")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    if aggregate_type == "scores":
        group_by = "player, DATE(created_at::TIMESTAMP)"
        select_columns = (
            "player, DATE(created_at::TIMESTAMP) as date, AVG(score) as avg_score"
        )
    else:
        group_by = "player, DATE(created_at::TIMESTAMP)"
        select_columns = "player, DATE(created_at::TIMESTAMP) as date, COUNT(DISTINCT song) as songs_played"  # noqa: E501

    query = f"""
    SELECT
        {select_columns}
    FROM {table_name}
    {where_clause}
    GROUP BY {group_by}
    ORDER BY player, date
    """  # noqa: S608

    result = conn.execute(query).fetchall()
    if not result:
        if aggregate_type == "scores":
            return pl.DataFrame({"player": [], "date": [], "avg_score": []})
        return pl.DataFrame({"player": [], "date": [], "songs_played": []})

    if aggregate_type == "scores":
        return pl.DataFrame(
            result,
            schema=["player", "date", "avg_score"],
            orient="row",
        )
    return pl.DataFrame(result, schema=["player", "date", "songs_played"], orient="row")


def get_difficulty_averages_duckdb(
    table_name: str = "scores",
    timespan_value: str | None = None,
    difficulty_values: list[str] | None = None,
    players: list[str] | None = None,
) -> pl.DataFrame:
    """
    Get average scores by difficulty using DuckDB.

    Args:
        table_name: Name of the table to query
        timespan_value: Timespan filter value
        difficulty_values: List of difficulty values to filter by
        players: List of players to filter by

    Returns:
        Polars DataFrame with difficulty averages
    """
    conn = get_duckdb_connection()

    where_conditions = []

    timespan_filter = filter_by_timespan_duckdb(timespan_value)
    if timespan_filter:
        where_conditions.append(timespan_filter.replace("WHERE ", ""))

    difficulty_filter = filter_by_difficulty_duckdb(difficulty_values)
    if difficulty_filter:
        where_conditions.append(difficulty_filter.replace("WHERE ", ""))

    if players and len(players) > 0:
        escaped_players = []
        for p in players:
            escaped_p = p.replace("'", "''")
            escaped_players.append(f"'{escaped_p}'")
        players_str = ", ".join(escaped_players)
        where_conditions.append(f"player IN ({players_str})")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    query = f"""
    SELECT
        player,
        difficulty,
        AVG(score) as avg_score
    FROM {table_name}
    {where_clause}
    GROUP BY player, difficulty
    ORDER BY player, difficulty
    """  # noqa: S608

    result = conn.execute(query).fetchall()
    if not result:
        return pl.DataFrame({"player": [], "difficulty": [], "avg_score": []})

    return pl.DataFrame(
        result,
        schema=["player", "difficulty", "avg_score"],
        orient="row",
    )


def get_worst_songs_duckdb(
    table_name: str = "scores",
    player: str | None = None,
    difficulty: str | None = None,
    limit: int = 10,
) -> pl.DataFrame:
    """
    Get worst songs for a specific player and difficulty using DuckDB.

    Args:
        table_name: Name of the table to query
        player: Player name to filter by
        difficulty: Difficulty to filter by
        limit: Number of worst songs to return

    Returns:
        Polars DataFrame with worst songs
    """
    conn = get_duckdb_connection()

    where_conditions = []

    if player:
        escaped_player = player.replace("'", "''")
        where_conditions.append(f"player = '{escaped_player}'")

    if difficulty:
        escaped_difficulty = difficulty.replace("'", "''")
        where_conditions.append(f"difficulty = '{escaped_difficulty}'")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    query = f"""
    SELECT
        song,
        MAX(score) as score,
        MAX(created_at) as created_at,
        MAX(mean_abs_offset) as mean_abs_offset
    FROM {table_name}
    {where_clause}
    GROUP BY song
    ORDER BY score ASC
    LIMIT {limit}
    """  # noqa: S608

    result = conn.execute(query).fetchall()
    if not result:
        return pl.DataFrame(
            {"song": [], "score": [], "created_at": [], "mean_abs_offset": []},
        )

    return pl.DataFrame(
        result,
        schema=["song", "score", "created_at", "mean_abs_offset"],
        orient="row",
    )


def get_score_category(best_score: int) -> str:
    """Get the category for a given score."""
    if best_score >= 100000:  # noqa: PLR2004
        return "100k"
    if best_score >= 99000:  # noqa: PLR2004
        return "99k+"
    if best_score >= 97000:  # noqa: PLR2004
        return "97k+"
    if best_score >= 90000:  # noqa: PLR2004
        return "90k+"
    if best_score >= 80000:  # noqa: PLR2004
        return "80k+"
    return "<80k"


def count_score_categories(result: list) -> dict[str, dict[str, int]]:
    """Count scores by category for each player."""
    categories = {}
    for player, _, best_score in result:
        if player not in categories:
            categories[player] = {
                "100k": 0,
                "99k+": 0,
                "97k+": 0,
                "90k+": 0,
                "80k+": 0,
                "<80k": 0,
                "Total": 0,
            }

        category = get_score_category(best_score)
        categories[player][category] += 1
        categories[player]["Total"] += 1

    return categories


def get_score_categories_duckdb(
    table_name: str = "scores",
    timespan_value: str | None = None,
    difficulty_values: list[str] | None = None,
    players: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """
    Get score categories breakdown using DuckDB.

    Args:
        table_name: Name of the table to query
        timespan_value: Timespan filter value
        difficulty_values: List of difficulty values to filter by
        players: List of players to filter by

    Returns:
        Dictionary with player names as keys and category counts as values
    """
    conn = get_duckdb_connection()

    where_conditions = []

    timespan_filter = filter_by_timespan_duckdb(timespan_value)
    if timespan_filter:
        where_conditions.append(timespan_filter.replace("WHERE ", ""))

    difficulty_filter = filter_by_difficulty_duckdb(difficulty_values)
    if difficulty_filter:
        where_conditions.append(difficulty_filter.replace("WHERE ", ""))

    if players and len(players) > 0:
        escaped_players = []
        for p in players:
            escaped_p = p.replace("'", "''")
            escaped_players.append(f"'{escaped_p}'")
        players_str = ", ".join(escaped_players)
        where_conditions.append(f"player IN ({players_str})")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    query = f"""
    SELECT
        player,
        song,
        MAX(score) as best_score
    FROM {table_name}
    {where_clause}
    GROUP BY player, song
    """  # noqa: S608

    result = conn.execute(query).fetchall()
    return count_score_categories(result)
