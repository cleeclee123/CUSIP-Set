from datetime import datetime
from typing import Literal, Optional
from pandas.tseries.offsets import BDay
import numpy as np
import pandas as pd
import QuantLib as ql
import rateslib as rl


class QL_BondPricer:

    @staticmethod
    def _pydatetime_to_qldate(date: datetime) -> ql.Date:
        return ql.Date(date.day, date.month, date.year)
    
    @staticmethod
    def _pydatetime_to_rldate(date: datetime) -> rl.dt:
        return rl.dt(date.year, date.month, date.day)

    @staticmethod
    def _bill_price_to_ytm(maturity_date: datetime, as_of: datetime, price: float) -> float:
        t = (maturity_date - as_of).days
        if t <= 0:
            return np.nan

        F = 100.0
        HPY = (F - price) / price
        BEY = HPY * (365 / t)
        return BEY * 100

    # @staticmethod
    # def _coupon_bond_price_to_ytm(
    #     issue_date: datetime,
    #     maturity_date: datetime,
    #     as_of: datetime,
    #     coupon: float,
    #     price: float,
    # ) -> float:
    #     issue_ql_date = QL_BondPricer._pydatetime_to_qldate(issue_date)
    #     maturity_ql_date = QL_BondPricer._pydatetime_to_qldate(maturity_date)
    #     settlement_days = 1
    #     settlement_date = (pd.to_datetime(as_of) + pd.tseries.offsets.BDay(settlement_days)).to_pydatetime()
    #     settlement_ql_date = QL_BondPricer._pydatetime_to_qldate(settlement_date)

    #     clean_price = price
    #     coupon_rate = coupon / 100
    #     day_count = ql.ActualActual(ql.ActualActual.Bond)
    #     coupon_frequency = ql.Semiannual
    #     schedule = ql.Schedule(
    #         issue_ql_date,
    #         maturity_ql_date,
    #         ql.Period(coupon_frequency),
    #         ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    #         ql.Unadjusted,
    #         ql.Unadjusted,
    #         ql.DateGeneration.Backward,
    #         False,
    #     )

    #     bond = ql.FixedRateBond(settlement_days, 100.0, schedule, [coupon_rate], day_count)
    #     bond_price_handle = ql.QuoteHandle(ql.SimpleQuote(clean_price))
    #     bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(ql.FlatForward(settlement_ql_date, 0.0, day_count)))
    #     bond.setPricingEngine(bond_engine)
    #     ytm = bond.bondYield(
    #         bond_price_handle.currentLink().value(),
    #         day_count,
    #         ql.Compounded,
    #         coupon_frequency,
    #     )
    #     return ytm * 100
    
    @staticmethod
    def _coupon_bond_price_to_ytm(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ) -> float:
        fxb_ust = rl.FixedRateBond(
            effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
            fixed_rate=coupon * 100,
            spec="ust",
            calc_mode="ust_31bii",
        )
        settle_pd_ts: pd.Timestamp = as_of + BDay(1)
        return fxb_ust.ytm(price=price, settlement=settle_pd_ts.to_pydatetime())

    @staticmethod
    def _coupon_bond_ytm_to_price(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        ytm: float,
        dirty: Optional[bool] = False,
    ) -> float:
        fxb_ust = rl.FixedRateBond(
            effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
            fixed_rate=coupon * 100,
            spec="ust",
            calc_mode="ust_31bii",
        )
        settle_pd_ts: pd.Timestamp = as_of + BDay(1)
        return fxb_ust.price(
            ytm=ytm, settlement=settle_pd_ts.to_pydatetime(), dirty=dirty
        )

    @staticmethod
    def _bond_mod_duration(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        ytm: float,
    ) -> float:
        try:
            fxb_ust = rl.FixedRateBond(
                effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
                termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
                fixed_rate=coupon * 100,
                spec="ust",
            )
            settle_pd_ts: pd.Timestamp = as_of + BDay(1)
            return fxb_ust.duration(settlement=settle_pd_ts, ytm=ytm, metric="modified")
        except:
            return None

    @staticmethod
    def bond_price_to_ytm(
        type: Literal["Bill", "Note", "Bond"],
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ):
        try:
            if type == "Bill":
                return QL_BondPricer._bill_price_to_ytm(
                    maturity_date=maturity_date,
                    as_of=as_of,
                    price=price,
                )

            return QL_BondPricer._coupon_bond_price_to_ytm(
                issue_date=issue_date,
                maturity_date=maturity_date,
                as_of=as_of,
                coupon=coupon,
                price=price,
            )
        except:
            return np.nan
