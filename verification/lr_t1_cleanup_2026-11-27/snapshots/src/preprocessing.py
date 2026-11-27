"""Shared preprocessing for UNSW-NB15

All decisions come from notebook/eda.ipynb, import it with:
    import sys; sys.path.append("..")
    from src.preprocessing import build_preprocessor, load_split
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA = Path(__file__).resolve().parents[1] / "data"

# Not inputs: id is a row number, label/attack_cat are targets, is_ftp_login is a copy of ct_ftp_cmd
NOT_FEATURES = ["id", "label", "attack_cat", "is_ftp_login"]

# Categorical: keep the common values, everything else becomes "other"
TOP_PROTO = ["tcp", "udp", "unas", "ospf", "arp", "sctp", "any", "gre", "rsvp", "ipv6",
             "sun-nd", "swipe", "sep", "mobile", "pim"]
SERVICES = ["-", "http", "dns", "smtp", "ftp-data", "ftp", "ssh", "pop3", "snmp", "ssl",
            "dhcp", "irc", "radius"]
KEEP_STATE = ["FIN", "INT", "CON", "REQ", "RST"]
CAT_COLS = ["proto", "service", "state"]

# Numeric: heavily skewed (skew > 1) and never negative ---> log1p, then scale
LOG_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sloss", "dloss",
            "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat", "smean", "dmean",
            "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
            "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd",
            "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst"]
# Numeric: not skewed (or a 0/1 flag) ---> only scale
SCALE_COLS = ["sttl", "dttl", "swin", "dwin", "stcpb", "dtcpb", "is_sm_ips_ports"]


def group_rare(X):
    # Replace rare proto/state values with "other" (fixed lists from the EDA, nothing is learned)
    X = X.copy()
    X["proto"] = X["proto"].where(X["proto"].isin(TOP_PROTO), "other")
    X["state"] = X["state"].where(X["state"].isin(KEEP_STATE), "other")
    return X


def build_preprocessor():
    # Turn the raw columns into 73 numeric features: 35 one-hot + 31 log1p-scaled + 7 scaled
    categorical = make_pipeline(
        FunctionTransformer(group_rare, feature_names_out="one-to-one"),
        OneHotEncoder(categories=[TOP_PROTO + ["other"], SERVICES, KEEP_STATE + ["other"]],
                      handle_unknown="ignore", sparse_output=False),
    )
    log_scaled = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    return ColumnTransformer([
        ("cat", categorical, CAT_COLS),
        ("log", log_scaled, LOG_COLS),
        ("num", StandardScaler(), SCALE_COLS),
    ])


def load_split(name):
    # Load data/processed/{name}.csv (name = "train", "val" or "test") as X, y, attack_cat
    df = pd.read_csv(DATA / "processed" / f"{name}.csv")
    X = df.drop(columns=[c for c in NOT_FEATURES if c in df.columns])
    return X, df["label"], df["attack_cat"]
