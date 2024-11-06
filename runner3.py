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

    ct_bid_df = cusip_curve_builder.get_historical_cts_INTERNAL(
        start_date=datetime(2024, 11, 1),
        end_date=ybday,
        use_bid_side=True,
        max_concurrent_tasks=64,
        max_keepalive_connections=12,
    )
    print(ct_bid_df)