import asyncio
import logging
import math
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from termcolor import colored

from utils import (
    JSON,
    build_treasurydirect_header,
    get_active_cusips,
    historical_auction_cols,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class CUSIP_Curve:
    _use_ust_issue_date: bool = False
    _global_timeout: int = 10
    _historical_auctions_df: pd.DataFrame = (None,)

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False  # performance benchmarking mainly
    _no_logs_plz: bool = False

    def __init__(
        self,
        use_ust_issue_date: Optional[bool] = False,
        global_timeout: int = 10,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
        no_logs_plz: Optional[bool] = False,
    ):
        self._use_ust_issue_date = use_ust_issue_date
        self._global_timeout = global_timeout

        self._historical_auctions_df = self.get_auctions_df()

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = no_logs_plz

        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.WARNING)

        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

        if self._debug_verbose or self._info_verbose or self._error_verbose:
            self._logger.setLevel(logging.DEBUG)

    async def _build_fetch_tasks_historical_treasury_auctions(
        self,
        client: httpx.AsyncClient,
        assume_data_size=True,
        uid: Optional[str | int] = None,
        return_df: Optional[bool] = False,
        as_of_date: Optional[datetime] = None,  # active cusips as of
    ):
        MAX_TREASURY_GOV_API_CONTENT_SIZE = 10000
        NUM_REQS_NEEDED_TREASURY_GOV_API = 2

        def get_treasury_query_sizing() -> List[str]:
            base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]=1&page[size]=1"
            res = requests.get(base_url, headers=build_treasurydirect_header())
            if res.ok:
                meta = res.json()["meta"]
                size = meta["total-count"]
                number_requests = math.ceil(size / MAX_TREASURY_GOV_API_CONTENT_SIZE)
                return [
                    f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                    for i in range(0, number_requests)
                ]
            else:
                raise ValueError(f"UST Auctions - Query Sizing Bad Status: ", {res.status_code})

        links = (
            get_treasury_query_sizing()
            if not assume_data_size
            else [
                f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                for i in range(0, NUM_REQS_NEEDED_TREASURY_GOV_API)
            ]
        )
        self._logger.debug(f"UST Auctions - Number of Links to Fetch: {len(links)}")
        self._logger.debug(f"UST Auctions - Links: {links}")

        async def fetch(
            client: httpx.AsyncClient,
            url,
            as_of_date: Optional[datetime] = None,
            return_df: Optional[bool] = False,
            uid: Optional[str | int] = None,
        ):
            try:
                response = await client.get(url, headers=build_treasurydirect_header())
                response.raise_for_status()
                json_data: JSON = response.json()
                if as_of_date:
                    df = get_active_cusips(
                        auction_json=json_data["data"],
                        as_of_date=as_of_date,
                        use_issue_date=self._use_ust_issue_date,
                    )
                    if uid:
                        return df[historical_auction_cols()], uid
                    return df[historical_auction_cols()]

                if return_df and not as_of_date:
                    if uid:
                        return (
                            pd.DataFrame(json_data["data"])[historical_auction_cols()],
                            uid,
                        )
                    return pd.DataFrame(json_data["data"])[historical_auction_cols()]
                if uid:
                    return json_data["data"], uid
                return json_data["data"]
            except httpx.HTTPStatusError as e:
                self._logger.debug(f"UST Prices - Bad Status: {response.status_code}")
                if uid:
                    return pd.DataFrame(columns=historical_auction_cols()), uid
                return pd.DataFrame(columns=historical_auction_cols())
            except Exception as e:
                self._logger.debug(f"UST Prices - Error: {e}")
                if uid:
                    return pd.DataFrame(columns=historical_auction_cols()), uid
                return pd.DataFrame(columns=historical_auction_cols())

        tasks = [
            fetch(
                client=client,
                url=url,
                as_of_date=as_of_date,
                return_df=return_df,
                uid=uid,
            )
            for url in links
        ]
        return tasks

    def get_auctions_df(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        async def build_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            tasks = await self._build_fetch_tasks_historical_treasury_auctions(client=client, as_of_date=as_of_date, return_df=True)
            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            async with httpx.AsyncClient() as client:
                all_data = await build_tasks(client=client, as_of_date=as_of_date)
                return all_data

        dfs = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_df: pd.DataFrame = pd.concat(dfs)
        auctions_df = auctions_df.sort_values(by=["auction_date"], ascending=False)
        return auctions_df

    async def _fetch_prices_from_treasury_date_search(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        cusips: List[str],
        uid: Optional[int | str],
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        payload = {
            "priceDate.month": date.month,
            "priceDate.day": date.day,
            "priceDate.year": date.year,
            "submit": "Show Prices",
        }
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            # "Content-Length": "100",
            "Content-Type": "application/x-www-form-urlencoded",
            "Dnt": "1",
            "Host": "savingsbonds.gov",
            "Origin": "https://savingsbonds.gov",
            "Referer": "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate",
            "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }
        self._logger.debug(f"UST Prices - {date} Payload: {payload}")
        cols_to_return = ["cusip", "offer_price", "bid_price", "eod_price"]
        retries = 0
        try:
            while retries < max_retries:
                try:
                    url = "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate"
                    response = await client.post(
                        url,
                        data=payload,
                        headers=headers,
                        follow_redirects=False,
                        timeout=self._global_timeout,
                    )
                    if response.is_redirect:
                        redirect_url = response.headers.get("Location")
                        self._logger.debug(f"UST Prices - {date} Redirecting to {redirect_url}")
                        response = await client.get(redirect_url)

                    response.raise_for_status()
                    tables = pd.read_html(response.content, header=0)
                    df = tables[0]
                    if cusips:
                        missing_cusips = [cusip for cusip in cusips if cusip not in df["CUSIP"].values]
                        if missing_cusips:
                            self._logger.warning(f"UST Prices Warning - The following CUSIPs are not found in the DataFrame: {missing_cusips}")
                    df = df[df["CUSIP"].isin(cusips)] if cusips else df
                    df.columns = df.columns.str.lower()
                    df = df.query("`security type` not in ['TIPS', 'MARKET BASED FRN']")
                    df = df.rename(
                        columns={
                            "buy": "offer_price",
                            "sell": "bid_price",
                            "end of day": "eod_price",
                        }
                    )
                    if uid:
                        return date, df[cols_to_return], uid
                    return date, df[cols_to_return]

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"UST Prices - Bad Status for {date}: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return date, df[cols_to_return], uid
                        return date, df[cols_to_return]
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"UST Prices - Error for {date}: {e}")
                    print(colored(e, "red"))
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST Prices - Max retries exceeded for {date}")
        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, pd.DataFrame(columns=cols_to_return), uid
            return date, pd.DataFrame(columns=cols_to_return)

    async def _fetch_prices_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_prices_from_treasury_date_search(*args, **kwargs)

    async def _build_fetch_tasks_historical_cusip_prices(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_prices_with_semaphore(
                semaphore,
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
            )
            for date in dates
        ]
        return tasks

    def github_headers(self, path: str):
        return {
            "authority": "raw.githubusercontent.com",
            "method": "GET",
            "path": path,
            "scheme": "https",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "dnt": "1",
            "pragma": "no-cache",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

    async def _fetch_ust_prices_from_github(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        cols_to_return: Optional[List[str]] = [
            "cusip",
            "offer_yield",
            "bid_yield",
            "eod_yield",
        ],
        cusip_ref_replacement_dict: Optional[Dict[str, str]] = None,
        return_transpose_df: Optional[bool] = False,
        assume_otrs: Optional[bool] = False,
        set_cusips_as_index: Optional[bool] = False,
    ):
        date_str = date.strftime("%Y-%m-%d")
        headers = self.github_headers(path=f"/cleeclee123/CUSIP-Set/main/{date_str}.json")
        url = f"https://raw.githubusercontent.com/cleeclee123/CUSIP-Set/main/{date_str}.json"
        retries = 0
        cols_to_return_copy = cols_to_return.copy()
        try:
            while retries < max_retries:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    res_json = response.json()
                    df = pd.DataFrame(res_json["data"])
                    if df.empty:
                        self._logger.error(f"UST Prices GitHub - Data is Empty for {date}")
                        if uid:
                            return date, None, uid
                        return date, None

                    df["issue_date"] = pd.to_datetime(df["issue_date"])
                    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

                    if cusips:
                        missing_cusips = [cusip for cusip in cusips if cusip not in df["cusip"].values]
                        if missing_cusips:
                            self._logger.warning(f"UST Prices Warning - The following CUSIPs are not found in the DataFrame: {missing_cusips}")
                    df = df[df["cusip"].isin(cusips)] if cusips else df

                    if cusip_ref_replacement_dict:
                        df["cusip"] = df["cusip"].replace(cusip_ref_replacement_dict)

                    if assume_otrs and "original_security_term" in df.columns:
                        df_coups = df.sort_values(by=["issue_date"], ascending=False)
                        df_coups = df_coups[(df_coups["security_type"] == "Note") | (df_coups["security_type"] == "Bond")]
                        df_coups = df_coups.groupby("original_security_term").first().reset_index()

                        df_bills = df.sort_values(by=["maturity_date"], ascending=True)
                        df_bills = df_bills[df_bills["security_type"] == "Bill"]
                        df_bills = df_bills.groupby("security_term").last().reset_index()

                        df_bills.loc[df_bills['security_term'] == '4-Week', 'original_security_term'] = '4-Week' 
                        df_bills.loc[df_bills['security_term'] == '8-Week', 'original_security_term'] = '8-Week' 
                        df_bills.loc[df_bills['security_term'] == '13-Week', 'original_security_term'] = '13-Week' 
                        df_bills.loc[df_bills['security_term'] == '17-Week', 'original_security_term'] = '17-Week' 
                        df_bills.loc[df_bills['security_term'] == '26-Week', 'original_security_term'] = '26-Week' 
                        df_bills.loc[df_bills['security_term'] == '52-Week', 'original_security_term'] = '52-Week' 

                        df = pd.concat([df_bills, df_coups])
                        # print(date)
                        # print(df) 
                        cusip_to_term_dict = dict(zip(df["cusip"], df["original_security_term"]))
                        df["cusip"] = df["cusip"].replace(cusip_to_term_dict)

                    if set_cusips_as_index:
                        df = df.set_index("cusip")
                        cols_to_return_copy.remove("cusip")

                    if return_transpose_df:
                        df = df[cols_to_return_copy].T
                    else:
                        df = df[cols_to_return_copy]

                    if uid:
                        return date, df, uid
                    return date, df
                except httpx.ConnectTimeout as e:
                    self._logger.debug(f"UST Prices GitHub - Timeout for {date}")
                    if uid:
                        return date, None, uid
                    return date, None
                except httpx.HTTPStatusError as e:
                    self._logger.error(f"UST Prices GitHub - HTTPX Error for {date}: {e}")
                    if response.status_code == 404:
                        self._logger.debug(f"UST Prices GitHub - Status Code: {response.status_code} for {date}")
                        if uid:
                            return date, None, uid
                        return date, None
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST Prices GitHub - Throttled for {date}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"UST Prices GitHub - Error for {date}: {e}")
                    print()
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST Prices GitHub - Throttled for {date}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST Prices GitHub - Max retries exceeded for {date}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, None, uid
            return date, None

    async def _fetch_ust_prices_from_github_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_ust_prices_from_github(*args, **kwargs)

    async def _build_fetch_tasks_historical_cusip_prices_github(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
        cols_to_return: Optional[List[str]] = [
            "cusip",
            "offer_yield",
            "bid_yield",
            "eod_yield",
        ],
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_ust_prices_from_github_with_semaphore(
                semaphore,
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
                cols_to_return=cols_to_return,
            )
            for date in dates
        ]
        return tasks

    def get_historical_cts_INTERNAL(
        self,
        start_date: datetime,
        end_date: datetime,
        use_bid_side: Optional[bool] = False,
        use_offer_side: Optional[bool] = False,
        use_mid_side: Optional[bool] = False,
        use_prices: Optional[bool] = False,
        max_concurrent_tasks: int = 64,
        max_keepalive_connections: Optional[int] = 10,
        timeout: Optional[int] = 30,
    ):
        cols_to_return = ["cusip"]
        if use_bid_side:
            cols_to_return.append("bid_price" if use_prices else "bid_yield")
        elif use_offer_side:
            cols_to_return.append("offer_price" if use_prices else "offer_yield")
        elif use_mid_side:
            cols_to_return.append("mid_price" if use_prices else "mid_yield")
        else:
            cols_to_return.append("eod_price" if use_prices else "eod_yield")

        async def build_tasks(
            client: httpx.AsyncClient,
            dates: List[datetime],
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for date in dates:
                task = self._fetch_ust_prices_from_github_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    date=date,
                    cols_to_return=cols_to_return,
                    assume_otrs=True,
                    set_cusips_as_index=True,
                )
                tasks.append(task)

            return await asyncio.gather(*tasks)

        async def run_fetch_all(dates: List[datetime]):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                all_data = await build_tasks(
                    client=client,
                    dates=dates,
                )
                return all_data

        bdays = (
            pd.date_range(
                start=start_date,
                end=end_date,
                freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
            )
            .to_pydatetime()
            .tolist()
        )
        results: List[Tuple[datetime, pd.DataFrame]] = asyncio.run(run_fetch_all(dates=bdays))
        results_dict: Dict[datetime, pd.DataFrame] = {dt: df for dt, df in results if dt is not None and df is not None}
        combined_df = pd.concat(results_dict).reset_index()
        final_df = combined_df.pivot(index="level_0", columns="cusip", values=cols_to_return[1])
        final_df = final_df.rename_axis("Date").reset_index()
        final_df.columns.name = None

        mapping = {
            "Date": 0,
            "4-Week": 0.077,
            "8-Week": 0.15,
            "13-Week": 0.25,
            "17-Week": 0.33,
            "26-Week": 0.5,
            "52-Week": 1,
            "2-Year": 2,
            "3-Year": 3,
            "5-Year": 5,
            "7-Year": 7,
            "10-Year": 10,
            "20-Year": 20,
            "30-Year": 30,
        }
        existing_cols = [col for col in final_df.columns if col in mapping]
        cols_sorted = sorted(existing_cols, key=lambda col: mapping[col])
        return final_df[cols_sorted]
