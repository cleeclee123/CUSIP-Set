import asyncio
import logging
from datetime import datetime
from typing import List, Optional, TypeAlias, Tuple

import math
import requests
import httpx
import pandas as pd

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def build_treasurydirect_header(
    host_str: Optional[str] = "api.fiscaldata.treasury.gov",
    cookie_str: Optional[str] = None,
    origin_str: Optional[str] = None,
    referer_str: Optional[str] = None,
):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7,application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": cookie_str or "",
        "DNT": "1",
        "Host": host_str or "",
        "Origin": origin_str or "",
        "Referer": referer_str or "",
        "Pragma": "no-cache",
        "Sec-CH-UA": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    }


def historical_auction_cols():
    return [
        "cusip",
        "security_type",
        "auction_date",
        "issue_date",
        "maturity_date",
        "price_per100",
        "allocation_pctage",
        "avg_med_yield",
        "bid_to_cover_ratio",
        "comp_accepted",
        "comp_tendered",
        "corpus_cusip",
        "currently_outstanding",
        "direct_bidder_accepted",
        "direct_bidder_tendered",
        "est_pub_held_mat_by_type_amt",
        "fima_included",
        "fima_noncomp_accepted",
        "fima_noncomp_tendered",
        "high_discnt_rate",
        "high_investment_rate",
        "high_price",
        "high_yield",
        "indirect_bidder_accepted",
        "indirect_bidder_tendered",
        "int_rate",
        "low_investment_rate",
        "low_price",
        "low_discnt_margin",
        "low_yield",
        "max_comp_award",
        "max_noncomp_award",
        "noncomp_accepted",
        "noncomp_tenders_accepted",
        "offering_amt",
        "security_term",
        "original_security_term",
        "security_term_week_year",
        "primary_dealer_accepted",
        "primary_dealer_tendered",
        "reopening",
        "total_accepted",
        "total_tendered",
        "treas_retail_accepted",
        "treas_retail_tenders_accepted",
    ]


def get_active_cusips(
    auction_json: Optional[JSON] = None,
    historical_auctions_df: Optional[pd.DataFrame] = None,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:
    if not auction_json and historical_auctions_df is None:
        return pd.DataFrame(columns=historical_auction_cols())

    if auction_json and historical_auctions_df is None:
        historical_auctions_df = pd.DataFrame(auction_json)

    historical_auctions_df["issue_date"] = pd.to_datetime(
        historical_auctions_df["issue_date"]
    )
    historical_auctions_df["maturity_date"] = pd.to_datetime(
        historical_auctions_df["maturity_date"]
    )
    historical_auctions_df["auction_date"] = pd.to_datetime(
        historical_auctions_df["auction_date"]
    )

    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains(
            "29-Year", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains(
            "30-", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"

    historical_auctions_df = historical_auctions_df[
        (historical_auctions_df["security_type"] == "Bill")
        | (historical_auctions_df["security_type"] == "Note")
        | (historical_auctions_df["security_type"] == "Bond")
    ]
    # historical_auctions_df = historical_auctions_df.drop(
    #     historical_auctions_df[
    #         (historical_auctions_df["security_type"] == "Bill")
    #         & (
    #             historical_auctions_df["original_security_term"]
    #             != historical_auctions_df["security_term"]
    #         )
    #     ].index
    # )
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df[
            "auction_date" if not use_issue_date else "issue_date"
        ].dt.date
        <= as_of_date.date()
    ]
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["maturity_date"] >= as_of_date
    ]
    historical_auctions_df = historical_auctions_df.drop_duplicates(
        subset=["cusip"], keep="first"
    )
    historical_auctions_df["int_rate"] = pd.to_numeric(
        historical_auctions_df["int_rate"], errors="coerce"
    )
    return historical_auctions_df


class FedInvestFetcher:
    _use_ust_issue_date: bool = False
    _global_timeout: int = 10

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

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = no_logs_plz

        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
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
                raise ValueError(
                    f"UST Auctions - Query Sizing Bad Status: ", {res.status_code}
                )

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
            tasks = await self._build_fetch_tasks_historical_treasury_auctions(
                client=client, as_of_date=as_of_date, return_df=True
            )
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
        max_retries=3,
        backoff_factor=1,
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
                        self._logger.debug(
                            f"UST Prices - {date} Redirecting to {redirect_url}"
                        )
                        response = await client.get(redirect_url)

                    response.raise_for_status()
                    tables = pd.read_html(response.content, header=0)
                    df = tables[0]
                    if cusips:
                        missing_cusips = [
                            cusip for cusip in cusips if cusip not in df["CUSIP"].values
                        ]
                        if missing_cusips:
                            self._logger.warning(
                                f"UST Prices Warning - The following CUSIPs are not found in the DataFrame: {missing_cusips}"
                            )
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
                    self._logger.error(
                        f"UST Prices - Bad Status for {date}: {response.status_code}"
                    )
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"UST Prices - Error for {date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying..."
                    )
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
        max_concurrent_tasks: int = 100,
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

    def _fetch_public_dotcome_jwt(self) -> str:
        try:
            jwt_headers = {
                "authority": "prod-api.154310543964.hellopublic.com",
                "method": "GET",
                "path": "/static/anonymoususer/credentials.json",
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "dnt": "1",
                "origin": "https://public.com",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "cross-site",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "x-app-version": "web-1.0.9",
            }
            jwt_url = "https://prod-api.154310543964.hellopublic.com/static/anonymoususer/credentials.json"
            jwt_res = requests.get(jwt_url, headers=jwt_headers)
            jwt_str = jwt_res.json()["jwt"]
            return jwt_str
        except Exception as e:
            self._logger.error(f"Public.com JWT Request Failed: {e}")
            return None

    async def _fetch_cusip_timeseries_public_dotcom(
        self,
        client: httpx.AsyncClient,
        cusip: str,
        jwt_str: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ):
        cols_to_return = ["Date", "Price", "YTM"]  # YTW is same as YTM for cash USTs
        retries = 0
        try:
            while retries < max_retries:
                try:
                    span = "MAX"
                    data_headers = {
                        "authority": "prod-api.154310543964.hellopublic.com",
                        "method": "GET",
                        "path": f"/fixedincomegateway/v1/graph/data?cusip={cusip}&span={span}",
                        "scheme": "https",
                        "accept": "*/*",
                        "accept-encoding": "gzip, deflate, br, zstd",
                        "accept-language": "en-US,en;q=0.9",
                        "cache-control": "no-cache",
                        "content-type": "application/json",
                        "dnt": "1",
                        "origin": "https://public.com",
                        "pragma": "no-cache",
                        "priority": "u=1, i",
                        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Windows"',
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "cross-site",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        "x-app-version": "web-1.0.9",
                        "authorization": jwt_str,
                    }

                    data_url = f"https://prod-api.154310543964.hellopublic.com/fixedincomegateway/v1/graph/data?cusip={cusip}&span={span}"
                    response = await client.get(data_url, headers=data_headers)
                    response.raise_for_status()
                    df = pd.DataFrame(response.json()["data"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df["unitPrice"] = pd.to_numeric(df["unitPrice"]) * 100
                    df["yieldToWorst"] = pd.to_numeric(df["yieldToWorst"]) * 100
                    df.columns = cols_to_return
                    if start_date:
                        df = df[df["Date"].dt.date >= start_date.date()]
                    if end_date:
                        df = df[df["Date"].dt.date <= end_date.date()]
                    if uid:
                        return cusip, df, uid
                    return cusip, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        f"Public.com - Bad Status: {response.status_code}"
                    )
                    if response.status_code == 404:
                        if uid:
                            return cusip, pd.DataFrame(columns=cols_to_return), uid
                        return cusip, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"Public.com - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Public.com - Error: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"Public.com - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Public.com - Max retries exceeded for {cusip}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return cusip, pd.DataFrame(columns=cols_to_return), uid
            return cusip, pd.DataFrame(columns=cols_to_return)

    async def _fetch_cusip_timeseries_public_dotcome_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_timeseries_public_dotcom(*args, **kwargs)

    def public_dotcom_timeseries_api(
        self,
        cusips: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        refresh_jwt: Optional[bool] = False,
        max_concurrent_tasks: int = 64,
    ):
        if refresh_jwt or not self._public_dotcom_jwt:
            self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()
            if not self._public_dotcom_jwt:
                raise ValueError("Public.com JWT Request Failed")

        async def build_tasks(
            client: httpx.AsyncClient,
            cusips: List[str],
            start_date: datetime,
            end_date: datetime,
            jwt_str: str,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_cusip_timeseries_public_dotcome_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    cusip=cusip,
                    start_date=start_date,
                    end_date=end_date,
                    jwt_str=jwt_str,
                )
                for cusip in cusips
            ]
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            cusips: List[str], start_date: datetime, end_date: datetime, jwt_str: str
        ):
            async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
                all_data = await build_tasks(
                    client=client,
                    cusips=cusips,
                    start_date=start_date,
                    end_date=end_date,
                    jwt_str=jwt_str,
                )
                return all_data

        dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(
                cusips=cusips,
                start_date=start_date,
                end_date=end_date,
                jwt_str=self._public_dotcom_jwt,
            )
        )
        return dict(dfs)

    # def public_dotcom_specifc_dates_api(
    #     self,
    #     dates: List[datetime],
    #     cusips: List[str],
    #     refresh_jwt: Optional[bool] = False,
    #     max_concurrent_tasks: int = 64,
    # ):
    #     if refresh_jwt or not self._public_dotcom_jwt:
    #         self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()
    #         if not self._public_dotcom_jwt:
    #             raise ValueError("Public.com JWT Request Failed")

    #     async def build_tasks(
    #         client: httpx.AsyncClient,
    #         cusips: List[str],
    #         start_date: datetime,
    #         end_date: datetime,
    #         jwt_str: str,
    #     ):
    #         semaphore = asyncio.Semaphore(max_concurrent_tasks)
    #         tasks = []
    #         for date in dates:
    #             curr_date_tasks = [self._fetch_cusip_timeseries_public_dotcome_with_semaphore(
    #                 semaphore=semaphore,
    #                 client=client,
    #                 cusip=cusip,
    #                 start_date=start_date,
    #                 end_date=end_date,
    #                 jwt_str=jwt_str,
    #             )
    #             for cusip in cusips
    #         ]
    #         return await asyncio.gather(*tasks)

    #     async def run_fetch_all(
    #         cusips: List[str], start_date: datetime, end_date: datetime, jwt_str: str
    #     ):
    #         async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
    #             all_data = await build_tasks(
    #                 client=client,
    #                 cusips=cusips,
    #                 start_date=start_date,
    #                 end_date=end_date,
    #                 jwt_str=jwt_str,
    #             )
    #             return all_data

    #     dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
    #         run_fetch_all(
    #             cusips=cusips,
    #             start_date=start_date,
    #             end_date=end_date,
    #             jwt_str=self._public_dotcom_jwt,
    #         )
    #     )
    #     return dict(dfs)