import asyncio
import multiprocessing as mp
import os
import time
from datetime import datetime
from functools import partial
from typing import Dict, TypeAlias

import httpx
import numpy as np
import pandas as pd
import ujson as json
from concurrent.futures import ProcessPoolExecutor, as_completed


from RL_BondPricer import RL_BondPricer
from script import FedInvestFetcher

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


def calculate_yields(row, as_of_date):
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
        tasks = await FedInvestFetcher(
            use_ust_issue_date=True, error_verbose=True
        )._build_fetch_tasks_historical_cusip_prices(
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
    business_day_groups = [
        [bday.to_pydatetime() for bday in date_range[i : i + group_size].tolist()]
        for i in range(0, len(date_range), group_size)
    ]
    return business_day_groups


def ust_labeler(mat_date: datetime | pd.Timestamp):
    return mat_date.strftime("%b %y") + "s"


def process_dataframe(key: datetime, df: pd.DataFrame, raw_auctions_df: pd.DataFrame):
    try:
        raw_auctions_df["label"] = raw_auctions_df["maturity_date"].apply(ust_labeler)
        raw_auctions_df = raw_auctions_df.sort_values(
            by=["issue_date"], ascending=False
        )
        otr_cusips = (
            raw_auctions_df.groupby("original_security_term")
            .first()
            .reset_index()["cusip"]
            .to_list()
        )
        raw_auctions_df["is_on_the_run"] = raw_auctions_df["cusip"].isin(otr_cusips)

        cusip_ref_df = raw_auctions_df[
            raw_auctions_df["cusip"].isin(df["cusip"].to_list())
        ][
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
        futures = {
            executor.submit(process_dataframe, key, df, raw_auctions_df): key
            for key, df in dict_df.items()
        }
        for future in as_completed(futures):
            key, json_structure = future.result()
            result_dict[key] = json_structure

    return result_dict


if __name__ == "__main__":
    t1 = time.time()
    start_date = datetime(2024, 8, 26)
    end_date = datetime(2024, 8, 27)
    weeks = get_business_days_groups(start_date, end_date, group_size=20)
    # weeks.reverse()

    raw_auctions_df = FedInvestFetcher(
        use_ust_issue_date=True, error_verbose=True
    ).get_auctions_df()
    raw_auctions_df["issue_date"] = pd.to_datetime(raw_auctions_df["issue_date"])
    raw_auctions_df["maturity_date"] = pd.to_datetime(raw_auctions_df["maturity_date"])
    raw_auctions_df["auction_date"] = pd.to_datetime(raw_auctions_df["auction_date"])
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains(
            "29-Year", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains(
            "30-", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"

    raw_auctions_df = raw_auctions_df[
        (raw_auctions_df["security_type"] == "Bill")
        | (raw_auctions_df["security_type"] == "Note")
        | (raw_auctions_df["security_type"] == "Bond")
    ]
    # raw_auctions_df = raw_auctions_df.drop(
    #     raw_auctions_df[
    #         (raw_auctions_df["security_type"] == "Bill")
    #         & (
    #             raw_auctions_df["original_security_term"]
    #             != raw_auctions_df["security_term"]
    #         )
    #     ].index
    # )
    raw_auctions_df = raw_auctions_df.drop_duplicates(subset=["cusip"], keep="last")

    for week in weeks:
        dict_df: Dict[datetime, pd.DataFrame] = runner(dates=week)
        output_directory = r"C:\Users\chris\CUSIP-Set"
        to_write = parallel_process(dict_df, raw_auctions_df)

        # for key, df in dict_df.items():
        #     try:
        #         cusip_ref_df = raw_auctions_df[
        #             raw_auctions_df["cusip"].isin(df["cusip"].to_list())
        #         ][["cusip", "security_type", "issue_date", "maturity_date", "int_rate"]]
        #         merged_df = pd.merge(left=df, right=cusip_ref_df, on=["cusip"])
        #         merged_df = merged_df.replace("null", np.nan)

        #         """MP"""
        #         # calculate_yields_partial = partial(
        #         #     calculate_yields, as_of_date=key
        #         # )
        #         # with mp.Pool(mp.cpu_count()) as pool:
        #         #     results = pool.map(
        #         #         calculate_yields_partial, [row for _, row in merged_df.iterrows()]
        #         #     )
        #         # offer_yields, bid_yields, eod_yields = zip(*results)
        #         # merged_df["offer_yield"] = offer_yields
        #         # merged_df["bid_yield"] = bid_yields
        #         # merged_df["eod_yield"] = eod_yields

        #         """"ST"""
        #         merged_df["eod_yield"] = merged_df.apply(
        #             lambda row: RL_BondPricer.bond_price_to_ytm(
        #                 type=row["security_type"],
        #                 issue_date=row["issue_date"],
        #                 maturity_date=row["maturity_date"],
        #                 as_of=key,
        #                 coupon=float(row["int_rate"]) / 100,
        #                 price=row["eod_price"],
        #             ),
        #             axis=1,
        #         )
        #         merged_df["bid_yield"] = merged_df.apply(
        #             lambda row: RL_BondPricer.bond_price_to_ytm(
        #                 type=row["security_type"],
        #                 issue_date=row["issue_date"],
        #                 maturity_date=row["maturity_date"],
        #                 as_of=key,
        #                 coupon=float(row["int_rate"]) / 100,
        #                 price=row["bid_price"],
        #             ),
        #             axis=1,
        #         )
        #         merged_df["offer_yield"] = merged_df.apply(
        #             lambda row: RL_BondPricer.bond_price_to_ytm(
        #                 type=row["security_type"],
        #                 issue_date=row["issue_date"],
        #                 maturity_date=row["maturity_date"],
        #                 as_of=key,
        #                 coupon=float(row["int_rate"]) / 100,
        #                 price=row["offer_price"],
        #             ),
        #             axis=1,
        #         )

        #         merged_df["mid_price"] = (
        #             merged_df["offer_price"] + merged_df["bid_price"]
        #         ) / 2
        #         merged_df["mid_yield"] = (
        #             merged_df["offer_yield"] + merged_df["bid_yield"]
        #         ) / 2

        #         merged_df = merged_df[["cusip", "bid_price", "offer_price", "mid_price", "eod_price", "bid_yield", "offer_yield", "eod_yield", "eod_yield"]]
        #         merged_df = merged_df.replace({np.nan: None})
        #         records = merged_df.to_dict(orient='records')
        #         json_structure = {"data": records}

        for key, json_structure in to_write.items():
            try:
                date_str = key.strftime("%Y-%m-%d")
                file_name = f"{date_str}.json"
                file_path = os.path.join(output_directory, file_name)

                with open(file_path, "w") as json_file:
                    json.dump(json_structure, json_file, indent=4, default=str)

                print(bcolors.OKGREEN + f"WROTE {key} to JSON" + bcolors.ENDC)

            except Exception as e:
                print(
                    bcolors.FAIL + f"FAILED JSON WRITE {key} - {str(e)}" + bcolors.ENDC
                )

    print(f"Script took: {time.time() - t1} seconds")
