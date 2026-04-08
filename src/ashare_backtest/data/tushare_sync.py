from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import gzip
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import time
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_TUSHARE_API_URL = "http://api.waditu.com/dataapi"
ALL_ACTIVE_UNIVERSE = "all_active"
DEFAULT_BENCHMARK_SYMBOL = "000300.SH"
DEFAULT_BENCHMARK_OUTPUT = "storage/parquet/benchmarks/000300.SH.parquet"
DEFAULT_TUSHARE_MAX_RETRIES = 5
DEFAULT_TUSHARE_RETRY_DELAY_SECONDS = 2.0
DEFAULT_SQLITE_COMMIT_INTERVAL = 20
RETRYABLE_HTTP_STATUS_CODES = {429, 502, 503, 504}


@dataclass(frozen=True)
class TushareSyncSummary:
    start_date: str
    end_date: str
    open_trade_dates: int
    stock_basic_rows: int
    active_symbols: int
    daily_rows: int
    daily_trade_dates: int


@dataclass(frozen=True)
class TushareBenchmarkSyncSummary:
    symbol: str
    start_date: str
    end_date: str
    rows: int
    output_path: str


class TushareClient:
    def __init__(
        self,
        token: str,
        timeout: int = 60,
        api_url: str = DEFAULT_TUSHARE_API_URL,
        max_retries: int = DEFAULT_TUSHARE_MAX_RETRIES,
        retry_delay_seconds: float = DEFAULT_TUSHARE_RETRY_DELAY_SECONDS,
    ) -> None:
        self.token = token
        self.timeout = timeout
        self.api_url = api_url.rstrip("/")
        self.max_retries = max(1, max_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "ashare-backtest/0.1.0",
        }

    def query(self, api_name: str, fields: str, **params: object) -> pd.DataFrame:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params,
            "fields": fields,
        }
        request = Request(
            url=f"{self.api_url}/{api_name}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    encoding = response.headers.get("Content-Encoding", "")
                    if "gzip" in encoding:
                        raw = gzip.decompress(raw)
                    text = raw.decode("utf-8")
                break
            except HTTPError as exc:
                if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < self.max_retries:
                    exc.read()
                    time.sleep(self.retry_delay_seconds * attempt)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Tushare HTTP error on {api_name}: {exc.code} {body}") from exc
            except (URLError, IncompleteRead) as exc:
                if attempt >= self.max_retries:
                    detail = exc.reason if isinstance(exc, URLError) else str(exc)
                    raise RuntimeError(f"Tushare network error on {api_name}: {detail}") from exc
                time.sleep(self.retry_delay_seconds * attempt)

        result = json.loads(text)
        if result.get("code") != 0:
            raise RuntimeError(f"Tushare API error on {api_name}: {result.get('msg')}")
        data = result["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])

    def trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.query(
            "trade_cal",
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            fields="exchange,cal_date,is_open,pretrade_date",
        )

    def stock_basic(self, list_status: str) -> pd.DataFrame:
        return self.query(
            "stock_basic",
            exchange="",
            list_status=list_status,
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
        )

    def daily(self, trade_date: str) -> pd.DataFrame:
        return self.query(
            "daily",
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )

    def daily_basic(self, trade_date: str) -> pd.DataFrame:
        return self.query(
            "daily_basic",
            trade_date=trade_date,
            fields="ts_code,trade_date,turnover_rate",
        )

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        return self.query("adj_factor", trade_date=trade_date, fields="ts_code,trade_date,adj_factor")

    def stk_limit(self, trade_date: str) -> pd.DataFrame:
        return self.query(
            "stk_limit",
            trade_date=trade_date,
            fields="ts_code,trade_date,up_limit,down_limit",
        )

    def suspend_d(self, trade_date: str) -> pd.DataFrame:
        return self.query(
            "suspend_d",
            trade_date=trade_date,
            suspend_type="S",
            fields="ts_code,trade_date,suspend_type",
        )

    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.query(
            "index_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount",
        )


class TushareSQLiteSync:
    def __init__(
        self,
        sqlite_path: str | Path,
        client: TushareClient,
        commit_interval: int = DEFAULT_SQLITE_COMMIT_INTERVAL,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.client = client
        self.commit_interval = max(1, commit_interval)

    def sync(self, start_date: str | None = None, end_date: str | None = None) -> TushareSyncSummary:
        resolved_start, resolved_end = self._resolve_window(start_date, end_date)
        with sqlite3.connect(self.sqlite_path) as conn:
            calendar = self._normalize_calendar(self.client.trade_calendar(resolved_start, resolved_end))
            self._upsert_calendar(conn, calendar)

            instruments = self._fetch_all_stock_basic()
            self._upsert_instruments(conn, instruments)
            self._refresh_all_active_universe(conn, instruments.loc[instruments["is_active"]].copy())
            conn.commit()

            open_dates = calendar.loc[calendar["is_open"], "trade_date"].tolist()
            daily_rows = 0
            for index, trade_date_item in enumerate(open_dates, start=1):
                daily_rows += self._sync_trade_date(conn, trade_date_item)
                if index % self.commit_interval == 0:
                    conn.commit()

            conn.commit()

        return TushareSyncSummary(
            start_date=pd.Timestamp(resolved_start).date().isoformat(),
            end_date=pd.Timestamp(resolved_end).date().isoformat(),
            open_trade_dates=len(open_dates),
            stock_basic_rows=len(instruments),
            active_symbols=int(instruments["is_active"].sum()),
            daily_rows=daily_rows,
            daily_trade_dates=len(open_dates),
        )

    def _resolve_window(self, start_date: str | None, end_date: str | None) -> tuple[str, str]:
        today = date.today().strftime("%Y%m%d")
        resolved_end = end_date or today
        if start_date:
            return start_date, resolved_end

        if not self.sqlite_path.exists():
            return (date.today() - timedelta(days=30)).strftime("%Y%m%d"), resolved_end

        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("select max(trade_date) from equity_daily_bars").fetchone()
        if row is None or row[0] is None:
            return (date.today() - timedelta(days=30)).strftime("%Y%m%d"), resolved_end
        next_date = datetime.strptime(row[0], "%Y-%m-%d").date() + timedelta(days=1)
        return next_date.strftime("%Y%m%d"), resolved_end

    def _fetch_all_stock_basic(self) -> pd.DataFrame:
        frames = [self.client.stock_basic(status) for status in ("L", "D", "P")]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._normalize_stock_basic(combined)

    def _sync_trade_date(self, conn: sqlite3.Connection, trade_date: str) -> int:
        compact_date = pd.Timestamp(trade_date).strftime("%Y%m%d")
        daily = self.client.daily(compact_date)
        if daily.empty:
            return 0
        basic = self.client.daily_basic(compact_date)
        adj = self.client.adj_factor(compact_date)
        limits = self.client.stk_limit(compact_date)
        suspended = self.client.suspend_d(compact_date)
        merged = self._merge_daily(daily=daily, basic=basic, adj=adj, limits=limits, suspended=suspended)
        self._upsert_daily_bars(conn, merged)
        return len(merged)

    def _upsert_calendar(self, conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
        rows = [
            (
                row.trade_date,
                int(row.is_open),
                0,
                "Synced from Tushare trade_cal",
            )
            for row in frame.itertuples(index=False)
        ]
        conn.executemany(
            """
            insert into trading_calendar (trade_date, is_open, has_night_session, notes)
            values (?, ?, ?, ?)
            on conflict(trade_date) do update set
                is_open=excluded.is_open,
                has_night_session=excluded.has_night_session,
                notes=excluded.notes
            """,
            rows,
        )

    def _upsert_instruments(self, conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
        existing_created_at = {
            row[0]: row[1]
            for row in conn.execute("select symbol, created_at from equity_instruments")
        }
        now = datetime.now(UTC).isoformat(sep=" ", timespec="seconds")
        rows = []
        for row in frame.itertuples(index=False):
            rows.append(
                (
                    row.symbol,
                    row.exchange,
                    row.name,
                    _nullable(row.listing_date),
                    _nullable(row.delisting_date),
                    _nullable(row.board),
                    _nullable(row.industry_level_1),
                    _nullable(row.industry_level_2),
                    int(row.is_st),
                    int(row.is_active),
                    existing_created_at.get(row.symbol, now),
                    now,
                )
            )
        conn.executemany(
            """
            insert into equity_instruments (
                symbol, exchange, name, listing_date, delisting_date, board,
                industry_level_1, industry_level_2, is_st, is_active, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(symbol) do update set
                exchange=excluded.exchange,
                name=excluded.name,
                listing_date=excluded.listing_date,
                delisting_date=excluded.delisting_date,
                board=excluded.board,
                industry_level_1=excluded.industry_level_1,
                industry_level_2=excluded.industry_level_2,
                is_st=excluded.is_st,
                is_active=excluded.is_active,
                updated_at=excluded.updated_at
            """,
            rows,
        )

    def _refresh_all_active_universe(self, conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
        conn.execute(
            "delete from equity_universe_memberships where universe_name = ? and source = ?",
            (ALL_ACTIVE_UNIVERSE, "tushare_stock_basic"),
        )
        rows = [
            (
                ALL_ACTIVE_UNIVERSE,
                row.symbol,
                _nullable(row.listing_date) or date.today().isoformat(),
                None,
                "tushare_stock_basic",
            )
            for row in frame.itertuples(index=False)
        ]
        conn.executemany(
            """
            insert into equity_universe_memberships (
                universe_name, symbol, effective_date, expiry_date, source
            )
            values (?, ?, ?, ?, ?)
            on conflict(universe_name, symbol, effective_date) do update set
                expiry_date=excluded.expiry_date,
                source=excluded.source
            """,
            rows,
        )

    def _upsert_daily_bars(self, conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
        rows = [
            (
                row.symbol,
                row.trade_date,
                row.open_price,
                row.high_price,
                row.low_price,
                row.close_price,
                row.prev_close_price,
                row.adj_factor,
                row.volume,
                row.turnover_amount,
                row.turnover_rate,
                row.limit_up_price,
                row.limit_down_price,
                int(row.is_suspended),
                int(row.is_limit_up),
                int(row.is_limit_down),
            )
            for row in frame.itertuples(index=False)
        ]
        conn.executemany(
            """
            insert into equity_daily_bars (
                symbol, trade_date, open_price, high_price, low_price, close_price, prev_close_price,
                adj_factor, volume, turnover_amount, turnover_rate, limit_up_price, limit_down_price,
                is_suspended, is_limit_up, is_limit_down
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(symbol, trade_date) do update set
                open_price=excluded.open_price,
                high_price=excluded.high_price,
                low_price=excluded.low_price,
                close_price=excluded.close_price,
                prev_close_price=excluded.prev_close_price,
                adj_factor=excluded.adj_factor,
                volume=excluded.volume,
                turnover_amount=excluded.turnover_amount,
                turnover_rate=excluded.turnover_rate,
                limit_up_price=excluded.limit_up_price,
                limit_down_price=excluded.limit_down_price,
                is_suspended=excluded.is_suspended,
                is_limit_up=excluded.is_limit_up,
                is_limit_down=excluded.is_limit_down
            """,
            rows,
        )

    @staticmethod
    def _normalize_calendar(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "is_open"])
        result = frame.copy()
        result["trade_date"] = pd.to_datetime(result["cal_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        result["is_open"] = result["is_open"].astype(int).astype(bool)
        return result.loc[:, ["trade_date", "is_open"]].sort_values("trade_date").reset_index(drop=True)

    @staticmethod
    def _normalize_stock_basic(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "exchange",
                    "name",
                    "listing_date",
                    "delisting_date",
                    "board",
                    "industry_level_1",
                    "industry_level_2",
                    "is_st",
                    "is_active",
                ]
            )
        result = frame.copy()
        result["symbol"] = result["ts_code"].astype(str)
        result["exchange"] = result["symbol"].str.split(".").str[-1].map({"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}).fillna("")
        result["listing_date"] = pd.to_datetime(result["list_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        result["delisting_date"] = pd.to_datetime(result["delist_date"], format="%Y%m%d", errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        result["board"] = result["market"].map(
            {
                "主板": "main_board",
                "创业板": "chinext",
                "科创板": "star",
                "北交所": "bj",
                "CDR": "cdr",
            }
        ).fillna("unknown")
        result["industry_level_1"] = result["industry"].where(result["industry"].notna(), None)
        result["industry_level_2"] = None
        result["is_st"] = result["name"].astype(str).str.upper().str.contains("ST", regex=False)
        result["is_active"] = result["list_status"].astype(str).eq("L")
        deduped = result.sort_values(["symbol", "is_active"], ascending=[True, False]).drop_duplicates(
            subset=["symbol"],
            keep="first",
        )
        return deduped.loc[
            :,
            [
                "symbol",
                "exchange",
                "name",
                "listing_date",
                "delisting_date",
                "board",
                "industry_level_1",
                "industry_level_2",
                "is_st",
                "is_active",
            ],
        ].reset_index(drop=True)

    @staticmethod
    def _merge_daily(
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        adj: pd.DataFrame,
        limits: pd.DataFrame,
        suspended: pd.DataFrame,
    ) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "prev_close_price",
                    "adj_factor",
                    "volume",
                    "turnover_amount",
                    "turnover_rate",
                    "limit_up_price",
                    "limit_down_price",
                    "is_suspended",
                    "is_limit_up",
                    "is_limit_down",
                ]
            )
        result = daily.rename(
            columns={
                "ts_code": "symbol",
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "close": "close_price",
                "pre_close": "prev_close_price",
                "vol": "volume",
                "amount": "turnover_amount",
            }
        ).copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        for frame in (basic, adj, limits):
            if not frame.empty:
                frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime(
                    "%Y-%m-%d"
                )

        basic_frame = basic.rename(columns={"ts_code": "symbol"}).loc[:, ["symbol", "trade_date", "turnover_rate"]] if not basic.empty else pd.DataFrame(columns=["symbol", "trade_date", "turnover_rate"])
        adj_frame = adj.rename(columns={"ts_code": "symbol"}).loc[:, ["symbol", "trade_date", "adj_factor"]] if not adj.empty else pd.DataFrame(columns=["symbol", "trade_date", "adj_factor"])
        limits_frame = (
            limits.rename(columns={"ts_code": "symbol", "up_limit": "limit_up_price", "down_limit": "limit_down_price"})
            .loc[:, ["symbol", "trade_date", "limit_up_price", "limit_down_price"]]
            if not limits.empty
            else pd.DataFrame(columns=["symbol", "trade_date", "limit_up_price", "limit_down_price"])
        )

        result = result.merge(basic_frame, on=["symbol", "trade_date"], how="left")
        result = result.merge(adj_frame, on=["symbol", "trade_date"], how="left")
        result = result.merge(limits_frame, on=["symbol", "trade_date"], how="left")

        suspended_symbols = set(suspended["ts_code"].astype(str).tolist()) if not suspended.empty else set()
        result["is_suspended"] = result["symbol"].isin(suspended_symbols)
        result["adj_factor"] = result["adj_factor"].fillna(1.0)
        result["is_limit_up"] = result["limit_up_price"].notna() & (
            (result["close_price"] - result["limit_up_price"]).abs() < 1e-6
        )
        result["is_limit_down"] = result["limit_down_price"].notna() & (
            (result["close_price"] - result["limit_down_price"]).abs() < 1e-6
        )
        return result.loc[
            :,
            [
                "symbol",
                "trade_date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "prev_close_price",
                "adj_factor",
                "volume",
                "turnover_amount",
                "turnover_rate",
                "limit_up_price",
                "limit_down_price",
                "is_suspended",
                "is_limit_up",
                "is_limit_down",
            ],
        ].sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def resolve_tushare_token(explicit_token: str | None = None) -> str | None:
    return explicit_token or os.getenv("TUSHARE_TOKEN")


class TushareBenchmarkSync:
    def __init__(self, client: TushareClient) -> None:
        self.client = client

    def sync(
        self,
        symbol: str = DEFAULT_BENCHMARK_SYMBOL,
        start_date: str | None = None,
        end_date: str | None = None,
        output_path: str | Path = DEFAULT_BENCHMARK_OUTPUT,
    ) -> TushareBenchmarkSyncSummary:
        resolved_end = end_date or date.today().strftime("%Y%m%d")
        resolved_start = start_date or (date.today() - timedelta(days=365 * 3)).strftime("%Y%m%d")
        frame = self.client.index_daily(symbol, resolved_start, resolved_end)
        normalized = self._normalize_index_daily(frame)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(target, index=False)
        min_date = normalized["trade_date"].min().date().isoformat() if not normalized.empty else ""
        max_date = normalized["trade_date"].max().date().isoformat() if not normalized.empty else ""
        return TushareBenchmarkSyncSummary(
            symbol=symbol,
            start_date=min_date,
            end_date=max_date,
            rows=len(normalized),
            output_path=target.as_posix(),
        )

    @staticmethod
    def _normalize_index_daily(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "prev_close",
                    "change",
                    "pct_chg",
                    "volume",
                    "amount",
                ]
            )
        result = frame.rename(
            columns={
                "ts_code": "symbol",
                "pre_close": "prev_close",
                "vol": "volume",
            }
        ).copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d")
        numeric_columns = ["open", "high", "low", "close", "prev_close", "change", "pct_chg", "volume", "amount"]
        for column in numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result.loc[
            :,
            ["symbol", "trade_date", "open", "high", "low", "close", "prev_close", "change", "pct_chg", "volume", "amount"],
        ].sort_values("trade_date").reset_index(drop=True)


def _nullable(value: object) -> object:
    return None if pd.isna(value) else value
