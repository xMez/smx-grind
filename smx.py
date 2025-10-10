import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import duckdb
import orjson
import polars as pl
from streamlit import logger
from streamlit.delta_generator import DeltaGenerator

SMX_API = "https://api.smx.573.no"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Process data page by page to minimize memory usage
PAGE_SIZE = 100  # Process 100 scores at a time from API


async def get_extra_data(
    session: aiohttp.ClientSession,
    score_id: int,
) -> dict[str, Any] | None:
    """Fetch extra offset data for a specific score."""
    try:
        async with session.get(f"{SMX_API}/extra/{score_id}") as response:
            if response.ok:
                return await response.json()
            return None
    except Exception:  # noqa: BLE001
        return None


async def fetch_page(
    session: aiohttp.ClientSession,
    username: str,
    skip: int,
    take: int,
) -> list[dict[str, Any]]:
    """Fetch a single page of scores from the API."""
    query = {
        "gamer.username": username,
        "_take": take,
        "_skip": skip,
        "_order": "asc",
    }
    params = {
        "q": orjson.dumps(query).decode("utf-8"),
    }

    async with session.get(f"{SMX_API}/scores", params=params) as response:
        return await response.json()


async def enrich_scores_with_extra_data(
    session: aiohttp.ClientSession,
    scores: list[dict[str, Any]],
    page: int,
    status_text: DeltaGenerator,
) -> None:
    """Add extra data to each score in the list."""
    if not scores:
        return

    if len(scores) > 10:  # noqa: PLR2004
        status_text.write(
            f"📄 **Page {page}** - Fetching extra data for {len(scores)} scores...",
        )

    tasks = []
    score_indices = []

    for i, score in enumerate(scores):
        score_id = score.get("id")
        if score_id:
            tasks.append(get_extra_data(session, score_id))
            score_indices.append(i)
        else:
            score["extra"] = None

    if tasks:
        extra_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(extra_data_results):
            score_idx = score_indices[i]
            if isinstance(result, Exception) or result is None:
                scores[score_idx]["extra"] = None
            else:
                scores[score_idx]["extra"] = result


class DuckDBStreamingProcessor:
    """DuckDB-based processor with streaming transformation."""

    def __init__(self) -> None:
        self.cache_dir = CACHE_DIR
        self.db_path = self.cache_dir / "scores_duckdb_streaming.db"
        self.state_file = self.cache_dir / "state.json"
        self.conn = duckdb.connect(str(self.db_path))
        self.table_name = "scores"
        self._create_table()

        # Check if table exists and has correct schema, recreate if needed
        try:
            result = self.conn.execute(
                f"PRAGMA table_info({self.table_name})",
            ).fetchall()
            if result and len(result) != 19:  # Should have 19 columns  # noqa: PLR2004
                print("Schema mismatch detected, recreating table...")  # noqa: T201
                self.recreate_table()
        except Exception:  # noqa: BLE001
            self._create_table()

    def _create_table(self) -> None:
        """Create the scores table with only the transformed columns."""
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                created_at VARCHAR,
                song VARCHAR,
                difficulty VARCHAR,
                grade INTEGER,
                score INTEGER,
                perfect1 INTEGER,
                perfect2 INTEGER,
                early INTEGER,
                late INTEGER,
                misses INTEGER,
                green INTEGER,
                yellow INTEGER,
                red INTEGER,
                max_combo INTEGER,
                full_combo BOOLEAN,
                mean_abs_offset DOUBLE,
                mean_offset DOUBLE,
                max_offset DOUBLE,
                player VARCHAR
            )
        """)

        # Create the highscores table with unique constraint
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS highscores (
                created_at VARCHAR,
                song VARCHAR,
                difficulty VARCHAR,
                grade INTEGER,
                score INTEGER,
                max_combo INTEGER,
                full_combo BOOLEAN,
                player VARCHAR,
                UNIQUE(song, difficulty, player)
            )
        """)

    def append_page_to_cache(
        self,
        raw_scores: list[dict[str, Any]],
        player: str | None = None,
    ) -> None:
        """Transform and append scores immediately using streaming approach."""
        if not raw_scores:
            return

        page_df = self._transform_page(raw_scores)

        if page_df.height > 0:
            self.append_scores_to_cache(page_df)

            # Check for new high scores if player is provided
            if player:
                self.check_and_insert_new_high_scores(page_df, player)

    def _transform_page(
        self,
        raw_scores: list[dict[str, Any]],
    ) -> pl.DataFrame:
        """Transform a single page of scores using main.py transformation."""
        if not raw_scores:
            return pl.DataFrame()

        df = pl.json_normalize(raw_scores, strict=False)
        return self._apply_main_transformation(df)

    def _apply_main_transformation(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply the transformation from main.py."""
        if df.height == 0:
            return df

        all_players = df["gamer.username"].unique().to_list()

        try:
            return (
                df.filter(
                    pl.col("gamer.username").is_in(all_players),
                    pl.col("cleared"),
                    pl.col("chart.difficulty_display") == "wild",
                )
                .with_columns(
                    song=pl.concat_str("song.title", pl.lit(" - "), "song.artist"),
                    difficulty=pl.concat_str(
                        "chart.difficulty_display",
                        pl.lit(" "),
                        "chart.difficulty",
                    ),
                    # Handle missing extra.offsets data gracefully - set to null if data is missing  # noqa: E501
                    mean_abs_offset=pl.when(pl.col("extra.offsets").is_not_null())
                    .then(
                        pl.col("extra.offsets")
                        .list.gather_every(3, offset=2)
                        .list.eval(pl.element().drop_nulls().abs())
                        .list.mean(),
                    )
                    .otherwise(None),
                    mean_offset=pl.when(pl.col("extra.offsets").is_not_null())
                    .then(
                        pl.col("extra.offsets")
                        .list.gather_every(3, offset=2)
                        .list.mean(),
                    )
                    .otherwise(None),
                    max_offset=pl.when(pl.col("extra.offsets").is_not_null())
                    .then(
                        pl.col("extra.offsets")
                        .list.gather_every(3, offset=2)
                        .list.max(),
                    )
                    .otherwise(None),
                    player=pl.col("gamer.username"),
                )
                .select(
                    "created_at",
                    "song",
                    "difficulty",
                    "grade",
                    "score",
                    "perfect1",
                    "perfect2",
                    "early",
                    "late",
                    "misses",
                    "green",
                    "yellow",
                    "red",
                    "max_combo",
                    "full_combo",
                    "mean_abs_offset",
                    "mean_offset",
                    "max_offset",
                    "player",
                )
            )

        except Exception:  # noqa: BLE001
            return (
                df.filter(
                    pl.col("gamer.username").is_in(all_players),
                    pl.col("cleared"),
                    pl.col("chart.difficulty_display") == "wild",
                )
                .with_columns(
                    song=pl.concat_str("song.title", pl.lit(" - "), "song.artist"),
                    difficulty=pl.concat_str(
                        "chart.difficulty_display",
                        pl.lit(" "),
                        "chart.difficulty",
                    ),
                    mean_abs_offset=pl.lit(None),
                    mean_offset=pl.lit(None),
                    max_offset=pl.lit(None),
                    player=pl.col("gamer.username"),
                )
                .select(
                    "created_at",
                    "song",
                    "difficulty",
                    "grade",
                    "score",
                    "perfect1",
                    "perfect2",
                    "early",
                    "late",
                    "misses",
                    "green",
                    "yellow",
                    "red",
                    "max_combo",
                    "full_combo",
                    "mean_abs_offset",
                    "mean_offset",
                    "max_offset",
                    "player",
                )
            )

    def load_cache_state(self, username: str) -> tuple[str | None, int]:
        """Load the cached state (timestamp and offset) for a username."""
        if not self.state_file.exists():
            return None, 0  # No cache, start from beginning

        try:
            with self.state_file.open("r") as f:
                state = orjson.loads(f.read())
                player_state = state.get("players", {}).get(username, {})
                return player_state.get("latest_timestamp"), player_state.get(
                    "offset",
                    0,
                )
        except (orjson.JSONDecodeError, KeyError):
            return None, 0

    def save_cache_state(
        self,
        username: str,
        latest_timestamp: str,
        offset: int,
    ) -> None:
        """Save the cache state (timestamp and offset) for a username."""
        # Load existing state or create new
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as file:
                    state = orjson.loads(file.read())
            except (orjson.JSONDecodeError, KeyError):
                state = {"players": {}}
        else:
            state = {"players": {}}

        # Update player state
        if "players" not in state:
            state["players"] = {}

        state["players"][username] = {
            "latest_timestamp": latest_timestamp,
            "offset": offset,
            "last_updated": datetime.now(UTC).isoformat(),
        }

        with self.state_file.open("wb") as file:
            file.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))

    def load_cached_scores(self) -> pl.DataFrame:
        """Load cached scores from DuckDB table."""
        try:
            result = self.conn.execute(f"SELECT * FROM {self.table_name}").fetchall()  # noqa: S608
            if not result:
                return pl.DataFrame()

            columns = [desc[0] for desc in self.conn.description]
            return pl.DataFrame(result, schema=columns, orient="row")
        except Exception:  # noqa: BLE001
            return pl.DataFrame()

    def append_scores_to_cache(self, new_scores_df: pl.DataFrame) -> None:
        """Append new scores to the DuckDB table."""
        if new_scores_df.height == 0:
            return

        try:
            pandas_df = new_scores_df.to_pandas()

            self.conn.register("temp_scores", pandas_df)

            self.conn.execute(
                f"INSERT INTO {self.table_name} SELECT * FROM temp_scores",  # noqa: S608
            )

            self.conn.unregister("temp_scores")
        except Exception as e:  # noqa: BLE001
            print(f"Error inserting scores: {e}")  # noqa: T201
            # Fallback: convert to list of tuples and insert row by row
            for row in new_scores_df.iter_rows(named=True):
                values = tuple(row.values())
                placeholders = ", ".join(["?" for _ in values])
                self.conn.execute(
                    f"INSERT INTO {self.table_name} VALUES ({placeholders})",  # noqa: S608
                    values,
                )

    def get_final_dataframe(self) -> pl.DataFrame:
        """Load cached data from DuckDB table instead of using transformed_data."""
        return self.load_cached_scores()

    def get_all_unique_songs(self) -> pl.DataFrame:
        """Get all unique songs from the database across all players."""
        try:
            result = self.conn.execute(
                f"SELECT DISTINCT song, difficulty FROM {self.table_name} ORDER BY song, difficulty",  # noqa: E501, S608
            ).fetchall()

            if not result:
                return pl.DataFrame({"song": [], "difficulty": []})

            # Convert to Polars DataFrame
            return pl.DataFrame(result, schema=["song", "difficulty"], orient="row")
        except Exception as e:  # noqa: BLE001
            print(f"Error getting all unique songs: {e}")  # noqa: T201
            return pl.DataFrame({"song": [], "difficulty": []})

    def get_all_players(self) -> list[str]:
        """Get all unique players from the database."""
        try:
            result = self.conn.execute(
                f"SELECT DISTINCT player FROM {self.table_name}",  # noqa: S608
            ).fetchall()
            return [row[0] for row in result if row[0]]
        except Exception as e:  # noqa: BLE001
            print(f"Error getting all players: {e}")  # noqa: T201
            return []

    def insert_highscores(self, highscores_df: pl.DataFrame) -> None:
        """Insert highscores into the highscores table."""
        if highscores_df.height == 0:
            return

        try:
            # Ensure we only have the correct columns for highscores table
            required_columns = [
                "created_at",
                "song",
                "difficulty",
                "grade",
                "score",
                "max_combo",
                "full_combo",
                "player",
            ]

            # Select only the required columns
            filtered_df = highscores_df.select(required_columns)
            pandas_df = filtered_df.to_pandas()
            self.conn.register("temp_highscores", pandas_df)
            self.conn.execute("INSERT INTO highscores SELECT * FROM temp_highscores")
            self.conn.unregister("temp_highscores")
        except Exception as e:  # noqa: BLE001
            print(f"Error inserting highscores: {e}")  # noqa: T201
            # Fallback: convert to list of tuples and insert row by row
            for row in highscores_df.iter_rows(named=True):
                values = (
                    row["created_at"],
                    row["song"],
                    row["difficulty"],
                    row["grade"],
                    row["score"],
                    row["max_combo"],
                    row["full_combo"],
                    row["player"],
                )
                self.conn.execute(
                    "INSERT INTO highscores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )

    def insert_highscores_safe(self, highscores_df: pl.DataFrame) -> None:
        """Insert highscores into the highscores table with duplicate prevention."""
        if highscores_df.height == 0:
            return

        try:
            # Ensure we only have the correct columns for highscores table
            required_columns = [
                "created_at",
                "song",
                "difficulty",
                "grade",
                "score",
                "max_combo",
                "full_combo",
                "player",
            ]

            # Select only the required columns
            filtered_df = highscores_df.select(required_columns)
            pandas_df = filtered_df.to_pandas()
            self.conn.register("temp_highscores", pandas_df)

            # Use INSERT OR IGNORE to prevent duplicates
            self.conn.execute("""
                INSERT OR IGNORE INTO highscores
                SELECT * FROM temp_highscores
            """)
            self.conn.unregister("temp_highscores")
        except Exception as e:  # noqa: BLE001
            print(f"Error inserting highscores safely: {e}")  # noqa: T201
            # Fallback: convert to list of tuples and insert row by row with IGNORE
            for row in highscores_df.iter_rows(named=True):
                values = (
                    row["created_at"],
                    row["song"],
                    row["difficulty"],
                    row["grade"],
                    row["score"],
                    row["max_combo"],
                    row["full_combo"],
                    row["player"],
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO highscores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )

    def get_highscores(self, player: str | None = None) -> pl.DataFrame:
        """Get highscores from the database, optionally filtered by player."""
        try:
            if player:
                result = self.conn.execute(
                    "SELECT * FROM highscores WHERE player = ? ORDER BY song ASC",
                    (player,),
                ).fetchall()
            else:
                result = self.conn.execute(
                    "SELECT * FROM highscores ORDER BY song ASC",
                ).fetchall()

            if not result:
                return pl.DataFrame()

            columns = [desc[0] for desc in self.conn.description]
            return pl.DataFrame(result, schema=columns, orient="row")
        except Exception as e:  # noqa: BLE001
            print(f"Error getting highscores: {e}")  # noqa: T201
            return pl.DataFrame()

    def clear_highscores_for_players(self, players: list) -> None:
        """Clear highscores for specific players."""
        if not players:
            return

        players_str = "', '".join(players)
        self.conn.execute(
            f"DELETE FROM highscores WHERE player IN ('{players_str}')",  # noqa: S608
        )

    def update_highscores_for_player(self, player: str) -> None:
        """Update highscores for a specific player based on their current scores."""
        try:
            # Get all scores for the player from the main scores table
            result = self.conn.execute(
                f"SELECT created_at, song, difficulty, grade, score, max_combo, "  # noqa: S608
                f"full_combo, player FROM {self.table_name} WHERE player = ? "
                f"ORDER BY score DESC",
                (player,),
            ).fetchall()

            if not result:
                return

            # Clear existing highscores for this player
            self.conn.execute(
                "DELETE FROM highscores WHERE player = ?",
                (player,),
            )

            # Insert all scores as highscores (ordered by score DESC)
            for row in result:
                self.conn.execute(
                    "INSERT INTO highscores (created_at, song, difficulty, grade, "
                    "score, max_combo, full_combo, player) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )

        except Exception as e:  # noqa: BLE001
            print(f"Error updating highscores for player {player}: {e}")  # noqa: T201

    def check_and_insert_new_high_scores(
        self,
        new_scores_df: pl.DataFrame,
        player: str,
    ) -> None:
        """Check for new high scores and insert only those that are better."""
        if new_scores_df.height == 0:
            return

        try:
            # Get current highscores for this player as a DataFrame
            current_highscores_df = self.get_highscores(player)

            if current_highscores_df.height == 0:
                # No existing highscores, all new scores are high scores
                # Use INSERT OR IGNORE to prevent duplicates
                self.insert_highscores_safe(new_scores_df)
                return

            # Find new high scores using chained DataFrame operations
            new_high_scores_df = (
                new_scores_df.join(
                    current_highscores_df.select(
                        ["song", "difficulty", "score"],
                    ).rename(
                        {"score": "current_score"},
                    ),
                    on=["song", "difficulty"],
                    how="left",
                )
                .filter(
                    (
                        pl.col("current_score").is_null()
                    )  # New song/difficulty combination
                    | (pl.col("score") > pl.col("current_score")),  # Better score
                )
                .select(
                    [
                        "created_at",
                        "song",
                        "difficulty",
                        "grade",
                        "score",
                        "max_combo",
                        "full_combo",
                        "player",
                    ],
                )
            )

            if new_high_scores_df.height > 0:
                # Remove existing scores for songs/difficulties with new high scores
                songs_to_update = new_high_scores_df.select(
                    ["song", "difficulty"],
                ).unique()

                for row in songs_to_update.iter_rows(named=True):
                    self.conn.execute(
                        "DELETE FROM highscores WHERE player = ? AND song = ? "
                        "AND difficulty = ?",
                        (player, row["song"], row["difficulty"]),
                    )

                # Insert new high scores using safe method
                self.insert_highscores_safe(new_high_scores_df)

        except Exception as e:  # noqa: BLE001
            print(f"Error checking new high scores for player {player}: {e}")  # noqa: T201

    def clear_cache_for_players(self, players: list) -> None:
        """Clear cached data for specific players."""
        if not players:
            return

        players_str = "', '".join(players)
        self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE player IN ('{players_str}')",  # noqa: S608
        )

        # Also clear highscores for these players
        self.clear_highscores_for_players(players)

        for player in players:
            self.save_cache_state(player, None, 0)

    def recreate_table(self) -> None:
        """Recreate the table with the correct schema."""
        self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.conn.execute("DROP TABLE IF EXISTS highscores")
        self._create_table()

    def cleanup(self) -> None:
        """Clean up DuckDB resources."""
        if hasattr(self, "conn"):
            self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink(missing_ok=True)


class DuckDBProcessorManager:
    """Singleton manager for DuckDBStreamingProcessor instances."""

    _instance: DuckDBStreamingProcessor | None = None

    @classmethod
    def get_processor(cls) -> DuckDBStreamingProcessor:
        """Get or create a DuckDBStreamingProcessor instance."""
        if cls._instance is None:
            cls._instance = DuckDBStreamingProcessor()
        return cls._instance

    @classmethod
    def reset_processor(cls) -> None:
        """Reset the processor, cleaning up any prior state."""
        if cls._instance is not None:
            with contextlib.suppress(Exception):
                cls._instance.cleanup()
        cls._instance = DuckDBStreamingProcessor()


def get_duckdb_processor() -> DuckDBStreamingProcessor:
    """Get or create a global DuckDBStreamingProcessor instance."""
    return DuckDBProcessorManager.get_processor()


def reset_duckdb_streaming() -> None:
    """Reset the global processor, cleaning up any prior state."""
    DuckDBProcessorManager.reset_processor()


def should_stop_fetching(
    data: list,
    latest_timestamp: str,
    status_text: DeltaGenerator,
) -> bool:
    """Check if we should stop fetching based on various conditions."""
    if not data or len(data) == 0:
        return True

    if latest_timestamp and data:
        last_score_timestamp = data[-1].get("created_at")
        if last_score_timestamp and last_score_timestamp <= latest_timestamp:
            status_text.write("✅ **Up to date!** No new scores found.")
            return True

    return False


def _update_progress_bar(
    progress_bar: DeltaGenerator,
    page: int,
    total_scores: int,
    take: int,
) -> None:
    """Update the progress bar with current progress."""
    estimated_total_pages = max(1, total_scores // take + 1) if total_scores > 0 else 10
    current_progress = min(
        page / max(estimated_total_pages, page + 1),
        0.9,
    )
    progress_bar.progress(current_progress)


async def _process_single_page(  # noqa: PLR0913
    session: aiohttp.ClientSession,
    username: str,
    skip: int,
    take: int,
    page: int,
    status_text: DeltaGenerator,
    progress_bar: DeltaGenerator,
    processor: DuckDBStreamingProcessor,
    latest_timestamp: str | None,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Process a single page of scores."""
    status_text.write(f"📄 Fetching page {page}...")

    data = await fetch_page(session, username, skip, take)

    if should_stop_fetching(data, latest_timestamp, status_text):
        return [], 0, None

    if not data:
        status_text.write("No more data, stopping.")
        return [], 0, None

    await enrich_scores_with_extra_data(session, data, page, status_text)
    processor.append_page_to_cache(data, username)

    total_scores = len(data)
    latest_fetched = data[-1].get("created_at") if data else None

    _update_progress_bar(progress_bar, page, total_scores, take)

    return data, total_scores, latest_fetched


async def get_scores(
    username: str,
    status_text: DeltaGenerator,
    progress_bar: DeltaGenerator | None = None,
    max_pages: int = 1000,
) -> int:
    """
    Get scores for a user using DuckDB streaming approach with caching.

    Args:
        username: Username to fetch scores for
        status_text: Streamlit status text component
        progress_bar: Streamlit progress bar component
        max_pages: Maximum number of pages to fetch
        incremental: If True, only fetch new scores since last update

    Returns:
        Number of scores processed
    """
    processor = get_duckdb_processor()

    # Ensure highscores exist for this player before fetching new scores
    if not ensure_highscores_for_player(username):
        status_text.write(f"⚠️ No existing scores found for {username}")

    incremental = True
    try:
        latest_timestamp, cached_offset = (
            processor.load_cache_state(username) if incremental else (None, 0)
        )

        total_scores = 0
        page = 0
        skip = cached_offset
        take = 100
        latest_fetched = latest_timestamp

        async with aiohttp.ClientSession() as session:
            while page < max_pages:
                page += 1

                data, page_scores, page_latest = await _process_single_page(
                    session,
                    username,
                    skip,
                    take,
                    page,
                    status_text,
                    progress_bar,
                    processor,
                    latest_timestamp,
                )

                if not data:
                    break

                total_scores += page_scores
                if page_latest:
                    latest_fetched = page_latest

                current_offset = skip + len(data)
                if latest_fetched:
                    processor.save_cache_state(username, latest_fetched, current_offset)

                if len(data) < take:
                    break

                skip += take

        if progress_bar:
            progress_bar.progress(1.0)
        if total_scores > 0:
            status_text.write(f"✅ **Complete!** Fetched {total_scores} new scores")
        else:
            status_text.write("✅ **Complete!** No new scores found")

        return total_scores

    finally:
        pass


def get_optimized_dataframe(
    all_players: list | None = None,
) -> pl.DataFrame:
    """
    Get optimized dataframe for a user using DuckDB streaming approach.

    Args:
        all_players: List of all players to filter by

    Returns:
        Optimized Polars DataFrame
    """
    processor = get_duckdb_processor()

    df = processor.get_final_dataframe()
    if all_players and df.height > 0 and "player" in df.columns:
        df = df.filter(pl.col("player").is_in(all_players))

    return df


def get_all_unique_songs() -> pl.DataFrame:
    """
    Get all unique songs from the database using DuckDB streaming approach.

    Returns:
        Polars DataFrame with columns: song, difficulty
    """
    processor = get_duckdb_processor()
    return processor.get_all_unique_songs()


def insert_highscores(highscores_df: pl.DataFrame) -> None:
    """
    Insert highscores into the database.

    Args:
        highscores_df: Polars DataFrame with columns: created_at, song, difficulty,
                      grade, score, max_combo, full_combo, player
    """
    processor = get_duckdb_processor()
    processor.insert_highscores(highscores_df)


def get_highscores(player: str | None = None) -> pl.DataFrame:
    """
    Get highscores from the database.

    Args:
        player: Optional player name to filter by. If None, returns all highscores.

    Returns:
        Polars DataFrame with columns: created_at, song, difficulty, grade,
        score, max_combo, full_combo, player
    """
    processor = get_duckdb_processor()
    return processor.get_highscores(player)


def update_highscores_for_player(player: str) -> None:
    """
    Update highscores for a specific player based on their current scores.

    Args:
        player: Player name to update highscores for
    """
    processor = get_duckdb_processor()
    processor.update_highscores_for_player(player)


def ensure_highscores_for_player(player: str) -> bool:
    """
    Ensure highscores exist for a specific player.
    Only populate if no highscores table exists at all.

    Args:
        player: Player name to check/populate highscores for

    Returns:
        True if highscores exist or were successfully populated, False otherwise
    """
    processor = get_duckdb_processor()

    # Check if highscores table exists and has any records for this player
    try:
        processor.get_highscores(player)
        # If we can get highscores (even if empty), the table exists and is set up
    except Exception as e:  # noqa: BLE001
        # Table doesn't exist or is corrupted, need to recreate
        # Continue to check if player has scores
        logger.warning(f"Error getting highscores for player {player}: {e}")
    else:
        return True

    # Check if player has any scores in the main table
    player_scores = processor.conn.execute(
        f"SELECT COUNT(*) FROM {processor.table_name} WHERE player = ?",  # noqa: S608
        (player,),
    ).fetchone()

    # Only populate if this is the first time setting up highscores
    # The check_and_insert_new_high_scores method will handle individual score updates
    return bool(player_scores and player_scores[0] > 0)
