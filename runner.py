import asyncio
import multiprocessing as mp
import os
import time
from datetime import datetime
from functools import partial
from typing import Dict, TypeAlias
from pandas.tseries.offsets import BDay
from collections import defaultdict

import httpx
import numpy as np
import pandas as pd
import ujson as json
from concurrent.futures import ProcessPoolExecutor, as_completed

from RL_BondPricer import RL_BondPricer
from QL_BondPricer import QL_BondPricer
from script import FedInvestFetcher
from ct import CUSIP_Curve

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def calculate_yields(row, as_of_date, use_quantlib=True):
    if use_quantlib:
        offer_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]),
            price=row["offer_price"],
        )
        bid_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]),
            price=row["bid_price"],
        )
        eod_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]),
            price=row["eod_price"],
        )
    else:
        offer_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]) / 100,
            price=row["offer_price"],
        )
        bid_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]) / 100,
            price=row["bid_price"],
        )
        eod_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=float(row["int_rate"]) / 100,
            price=row["eod_price"],
        )

    return offer_yield, bid_yield, eod_yield


def runner(dates):
    async def build_tasks(client: httpx.AsyncClient, dates):
        tasks = await FedInvestFetcher(use_ust_issue_date=True, error_verbose=True)._build_fetch_tasks_historical_cusip_prices(
            client=client, dates=dates, max_concurrent_tasks=5
        )
        return await asyncio.gather(*tasks)

    async def run_fetch_all(dates):
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
        async with httpx.AsyncClient(limits=limits) as client:
            all_data = await build_tasks(client=client, dates=dates)
            return all_data

    results = asyncio.run(run_fetch_all(dates=dates))
    return dict(results)


def get_business_days_groups(start_date: datetime, end_date: datetime, group_size=3):
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    business_day_groups = [[bday.to_pydatetime() for bday in date_range[i : i + group_size].tolist()] for i in range(0, len(date_range), group_size)]
    return business_day_groups


def ust_labeler(mat_date: datetime | pd.Timestamp):
    return mat_date.strftime("%b %y") + "s"


def process_dataframe(key: datetime, df: pd.DataFrame, raw_auctions_df: pd.DataFrame):
    try:
        raw_auctions_df["label"] = raw_auctions_df["maturity_date"].apply(ust_labeler)
        raw_auctions_df = raw_auctions_df.sort_values(by=["issue_date"], ascending=False)
        raw_auctions_df.loc[raw_auctions_df["security_term"] == "4-Week", "original_security_term"] = "4-Week"
        raw_auctions_df.loc[raw_auctions_df["security_term"] == "8-Week", "original_security_term"] = "8-Week"
        otr_cusips = raw_auctions_df.groupby("original_security_term").first().reset_index()["cusip"].to_list()
        raw_auctions_df["is_on_the_run"] = raw_auctions_df["cusip"].isin(otr_cusips)

        cusip_ref_df = raw_auctions_df[raw_auctions_df["cusip"].isin(df["cusip"].to_list())][
            [
                "cusip",
                "security_type",
                "security_term",
                "original_security_term",
                "auction_date",
                "issue_date",
                "maturity_date",
                "int_rate",
                "high_investment_rate",
                "label",
                "is_on_the_run",
            ]
        ]

        merged_df = pd.merge(left=df, right=cusip_ref_df, on=["cusip"])
        merged_df = merged_df.replace("null", np.nan)

        merged_df["eod_yield"] = merged_df.apply(
            lambda row: RL_BondPricer.bond_price_to_ytm(
                type=row["security_type"],
                issue_date=row["issue_date"],
                maturity_date=row["maturity_date"],
                as_of=key,
                coupon=float(row["int_rate"]) / 100,
                price=row["eod_price"],
            ),
            axis=1,
        )
        merged_df["bid_yield"] = merged_df.apply(
            lambda row: RL_BondPricer.bond_price_to_ytm(
                type=row["security_type"],
                issue_date=row["issue_date"],
                maturity_date=row["maturity_date"],
                as_of=key,
                coupon=float(row["int_rate"]) / 100,
                price=row["bid_price"],
            ),
            axis=1,
        )
        merged_df["offer_yield"] = merged_df.apply(
            lambda row: RL_BondPricer.bond_price_to_ytm(
                type=row["security_type"],
                issue_date=row["issue_date"],
                maturity_date=row["maturity_date"],
                as_of=key,
                coupon=float(row["int_rate"]) / 100,
                price=row["offer_price"],
            ),
            axis=1,
        )

        merged_df["mid_price"] = (merged_df["offer_price"] + merged_df["bid_price"]) / 2
        merged_df["mid_yield"] = (merged_df["offer_yield"] + merged_df["bid_yield"]) / 2
        merged_df = merged_df[
            [
                "cusip",
                "security_type",
                "security_term",
                "original_security_term",
                "auction_date",
                "issue_date",
                "maturity_date",
                "int_rate",
                "high_investment_rate",
                "label",
                "is_on_the_run",
                "bid_price",
                "offer_price",
                "mid_price",
                "eod_price",
                "bid_yield",
                "offer_yield",
                "mid_yield",
                "eod_yield",
            ]
        ]
        merged_df = merged_df.replace({np.nan: None})
        records = merged_df.to_dict(orient="records")
        json_structure = {"data": records}
        return key, json_structure
    except Exception as e:
        print(bcolors.FAIL + f"FAILED DF PROCESSING {key} - {str(e)}" + bcolors.ENDC)
        return key, {"data": None}


def parallel_process(dict_df, raw_auctions_df):
    result_dict = {}

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_dataframe, key, df, raw_auctions_df): key for key, df in dict_df.items()}
        for future in as_completed(futures):
            key, json_structure = future.result()
            result_dict[key] = json_structure

    return result_dict


if __name__ == "__main__":
    t1_parent = time.time()
    t1 = time.time()

    y2bday: pd.Timestamp = datetime.today() - BDay(10)
    y2bday = y2bday.to_pydatetime()
    # y2bday = y2bday.replace(hour=0, minute=0, second=0, microsecond=0)

    ybday: pd.Timestamp = datetime.today() - BDay(1)
    ybday = ybday.to_pydatetime()
    # ybday = ybday.replace(hour=0, minute=0, second=0, microsecond=0)

    start_date = y2bday
    end_date = ybday
    # start_date = datetime(2024, 9, 25)
    # end_date = datetime(2024, 10, 2)

    print(bcolors.OKBLUE + f"Fetching UST Prices for {start_date} and {end_date}" + bcolors.ENDC)
    weeks = get_business_days_groups(start_date, end_date, group_size=20)

    raw_auctions_df = FedInvestFetcher(use_ust_issue_date=True, error_verbose=True).get_auctions_df()
    raw_auctions_df["issue_date"] = pd.to_datetime(raw_auctions_df["issue_date"])
    raw_auctions_df["maturity_date"] = pd.to_datetime(raw_auctions_df["maturity_date"])
    raw_auctions_df["auction_date"] = pd.to_datetime(raw_auctions_df["auction_date"])
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains("29-Year", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains("30-", case=False, na=False),
        "original_security_term",
    ] = "30-Year"

    raw_auctions_df = raw_auctions_df[
        (raw_auctions_df["security_type"] == "Bill") | (raw_auctions_df["security_type"] == "Note") | (raw_auctions_df["security_type"] == "Bond")
    ]

    raw_auctions_df.loc[raw_auctions_df["security_term"] == "4-Week", "original_security_term"] = "4-Week"
    raw_auctions_df.loc[raw_auctions_df["security_term"] == "8-Week", "original_security_term"] = "8-Week"
    raw_auctions_df.loc[raw_auctions_df["security_term"] == "17-Week", "original_security_term"] = "17-Week"

    raw_auctions_df.loc[raw_auctions_df["security_term"] == "13-Week", "original_security_term"] = "13-Week"
    raw_auctions_df.loc[raw_auctions_df["security_term"] == "26-Week", "original_security_term"] = "26-Week"
    raw_auctions_df.loc[raw_auctions_df["security_term"] == "52-Week", "original_security_term"] = "52-Week"

    raw_auctions_df_1 = raw_auctions_df[raw_auctions_df["security_type"] == "Bill"].drop_duplicates(subset=["cusip"], keep="first")
    raw_auctions_df_2 = raw_auctions_df[raw_auctions_df["security_type"] != "Bill"].drop_duplicates(subset=["cusip"], keep="last")

    raw_auctions_df = pd.concat([raw_auctions_df_1, raw_auctions_df_2])

    for week in weeks:
        dict_df: Dict[datetime, pd.DataFrame] = runner(dates=week)
        output_directory = r"C:\Users\chris\CUSIP-Set"
        to_write = parallel_process(dict_df, raw_auctions_df)

        for key, json_structure in to_write.items():
            try:
                date_str = key.strftime("%Y-%m-%d")
                file_name = f"{date_str}.json"
                file_path = os.path.join(output_directory, file_name)

                with open(file_path, "w") as json_file:
                    json.dump(json_structure, json_file, indent=4, default=str)

                print(bcolors.OKGREEN + f"WROTE {key} to JSON" + bcolors.ENDC)

            except Exception as e:
                print(bcolors.FAIL + f"FAILED JSON WRITE {key} - {str(e)}" + bcolors.ENDC)

    print(f"FedInvest Scraper Script took: {time.time() - t1} seconds")

    # print(bcolors.OKBLUE + "STARTING TIMESERIES SCRIPT" + bcolors.ENDC)

    time.sleep(10)

    t1 = time.time()

    input_directory = r"C:\Users\chris\CUSIP-Set"
    output_directory = r"C:\Users\chris\CUSIP-Timeseries"

    cusip_timeseries = defaultdict(list)

    keys_to_include = [
        "Date",
        "cusip",
        "bid_price",
        "offer_price",
        "mid_price",
        "eod_price",
        "bid_yield",
        "offer_yield",
        "mid_yield" "eod_yield",
    ]

    """ Entire Dir """

    for file_name in os.listdir(input_directory):
        try:
            if file_name.endswith(".json"):
                file_path = os.path.join(input_directory, file_name)
                with open(file_path, "r") as json_file:
                    daily_data = json.load(json_file)

                date_str = file_name.split(".json")[0]
                date = datetime.strptime(date_str, "%Y-%m-%d")

                for entry in daily_data["data"]:
                    cusip = entry["cusip"]
                    to_write = {
                        "Date": date_str,
                        "bid_price": entry["bid_price"],
                        "offer_price": entry["offer_price"],
                        "mid_price": entry["mid_price"],
                        "eod_price": entry["eod_price"],
                        "bid_yield": entry["bid_yield"],
                        "offer_yield": entry["offer_yield"],
                        "mid_yield": entry["mid_yield"],
                        "eod_yield": entry["eod_yield"],
                    }
                    cusip_timeseries[cusip].append(to_write)

            print(bcolors.OKBLUE + f"Saw {file_name}" + bcolors.ENDC)

        except Exception as e:
            print(bcolors.FAIL + f"FAILED {file_name} - {str(e)}" + bcolors.ENDC)

    for cusip, timeseries in cusip_timeseries.items():
        try:
            output_file = os.path.join(output_directory, f"{cusip}.json")
            with open(output_file, "w") as json_file:
                json.dump(timeseries, json_file, indent=4, default=str)
            print(bcolors.OKGREEN + f"Wrote time series for CUSIP {cusip} to {output_file}" + bcolors.ENDC)
        except Exception as e:
            print(bcolors.FAIL + f"FAILED to Write {cusip} to {output_file}" + bcolors.ENDC)

    print(f"Timeseries Script took: {time.time() - t1} seconds")

    ########################################################################################

    # print(bcolors.OKBLUE + "STARTING CT TIMESERIES SCRIPT" + bcolors.ENDC)
    # t1 = time.time()

    # def write_df_to_json(
    #     df: pd.DataFrame,
    #     file_path: str,
    #     orient: str = "records",
    #     date_format: str = "iso",
    # ):
    #     df.to_json(file_path, orient=orient, date_format=date_format, indent=4)

    # cusip_curve_builder = CUSIP_Curve(use_ust_issue_date=True, error_verbose=True)
    # ybday: pd.Timestamp = (datetime.today() - BDay(1))
    # ybday = ybday.to_pydatetime()
    # ybday = ybday.replace(hour=0, minute=0, second=0, microsecond=0)
    # print(bcolors.OKBLUE + f"Fetching to {ybday}" + bcolors.ENDC)
    # ct_bid_df = cusip_curve_builder.get_historical_cts_INTERNAL(
    #     start_date=datetime(2008, 5, 30),
    #     end_date=ybday,
    #     use_bid_side=True,
    #     max_concurrent_tasks=64,
    #     max_keepalive_connections=12,
    # )
    # write_df_to_json(
    #     df=ct_bid_df,
    #     file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_bid_side.json",
    #     date_format="iso",
    # )
    # print(ct_bid_df)
    # print(bcolors.OKGREEN + f"Wrote Bid CT Yields time series" + bcolors.ENDC)

    # ct_offer_df = cusip_curve_builder.get_historical_cts_INTERNAL(
    #     start_date=datetime(2008, 5, 30),
    #     end_date=ybday,
    #     use_offer_side=True,
    #     max_concurrent_tasks=64,
    #     max_keepalive_connections=12,
    # )
    # write_df_to_json(
    #     df=ct_offer_df,
    #     file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_offer_side.json",
    #     date_format="iso",
    # )
    # print(ct_offer_df)
    # print(bcolors.OKGREEN + f"Wrote offer CT Yields time series" + bcolors.ENDC)

    # ct_eod_df = cusip_curve_builder.get_historical_cts_INTERNAL(
    #     start_date=datetime(2008, 5, 30),
    #     end_date=ybday,
    #     max_concurrent_tasks=64,
    #     max_keepalive_connections=12,
    # )
    # write_df_to_json(
    #     df=ct_eod_df,
    #     file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_eod_side.json",
    #     date_format="iso",
    # )
    # print(ct_eod_df)
    # print(bcolors.OKGREEN + f"Wrote EOD CT Yields time series" + bcolors.ENDC)

    # ct_mid_df = cusip_curve_builder.get_historical_cts_INTERNAL(
    #     start_date=datetime(2008, 5, 30),
    #     end_date=ybday,
    #     use_mid_side=True,
    #     max_concurrent_tasks=64,
    #     max_keepalive_connections=12,
    # )
    # write_df_to_json(
    #     df=ct_mid_df,
    #     file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_mid_side.json",
    #     date_format="iso",
    # )
    # print(ct_mid_df)
    # print(bcolors.OKGREEN + f"Wrote Mid CT Yields time series" + bcolors.ENDC)
    # print(f"CT Timeseries Script took: {time.time() - t1} seconds")

    # print(f"Everything Script took: {time.time() - t1_parent} seconds")
