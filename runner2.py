import time
from datetime import datetime
from typing import TypeAlias
from pandas.tseries.offsets import BDay
import pandas as pd
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


if __name__ == "__main__":
    t1_parent = time.time()

    print(bcolors.OKBLUE + "STARTING CT TIMESERIES SCRIPT" + bcolors.ENDC)
    t1 = time.time()

    def write_df_to_json(
        df: pd.DataFrame,
        file_path: str,
        orient: str = "records",
        date_format: str = "iso",
    ):
        df.to_json(file_path, orient=orient, date_format=date_format, indent=4)

    cusip_curve_builder = CUSIP_Curve(use_ust_issue_date=True, error_verbose=True)
    ybday: pd.Timestamp = datetime.today() - BDay(1)
    ybday = ybday.to_pydatetime()
    ybday = ybday.replace(hour=0, minute=0, second=0, microsecond=0)
    print(bcolors.OKBLUE + f"Fetching to {ybday}" + bcolors.ENDC)
    ct_bid_df = cusip_curve_builder.get_historical_cts_INTERNAL(
        start_date=datetime(2008, 5, 30),
        end_date=ybday,
        use_bid_side=True,
        max_concurrent_tasks=64,
        max_keepalive_connections=12,
    )
    write_df_to_json(
        df=ct_bid_df,
        file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_bid_side.json",
        date_format="iso",
    )
    print(ct_bid_df
    print(bcolors.OKGREEN + f"Wrote Bid CT Yields time series" + bcolors.ENDC)

    ct_offer_df = cusip_curve_builder.get_historical_cts_INTERNAL(
        start_date=datetime(2008, 5, 30),
        end_date=ybday,
        use_offer_side=True,
        max_concurrent_tasks=64,
        max_keepalive_connections=12,
    )
    write_df_to_json(
        df=ct_offer_df,
        file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_offer_side.json",
        date_format="iso",
    )
    print(ct_offer_df)
    print(bcolors.OKGREEN + f"Wrote offer CT Yields time series" + bcolors.ENDC)

    ct_eod_df = cusip_curve_builder.get_historical_cts_INTERNAL(
        start_date=datetime(2008, 5, 30),
        end_date=ybday,
        max_concurrent_tasks=64,
        max_keepalive_connections=12,
    )
    write_df_to_json(
        df=ct_eod_df,
        file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_eod_side.json",
        date_format="iso",
    )
    print(ct_eod_df)
    print(bcolors.OKGREEN + f"Wrote EOD CT Yields time series" + bcolors.ENDC)

    ct_mid_df = cusip_curve_builder.get_historical_cts_INTERNAL(
        start_date=datetime(2008, 5, 30),
        end_date=ybday,
        use_mid_side=True,
        max_concurrent_tasks=64,
        max_keepalive_connections=12,
    )
    write_df_to_json(
        df=ct_mid_df,
        file_path=r"C:\Users\chris\CUSIP-Timeseries\historical_ct_yields_mid_side.json",
        date_format="iso",
    )
    print(ct_mid_df)
    print(bcolors.OKGREEN + f"Wrote Mid CT Yields time series" + bcolors.ENDC)
    print(f"CT Timeseries Script took: {time.time() - t1} seconds")

    print(f"Everything Script took: {time.time() - t1_parent} seconds")
