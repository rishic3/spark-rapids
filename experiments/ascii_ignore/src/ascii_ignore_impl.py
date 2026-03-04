import re
import pandas as pd

STRIP_CHARS = re.compile(r"""[?()\-%=:&*"'><\\}{$@+/\]\[;~#,]""")

def ascii_ignore_impl(x):
    if x:
      return x.encode('ascii', 'ignore').decode('ascii').strip().replace('?', '').replace(')', '') \
                .replace('(', '').replace('%', '').replace('=', '').replace(':', '').replace('&', '') \
                .replace('*', '').replace('"', '').replace("'", '').replace(">", '').replace("<", '') \
                .replace("\\", '').replace("}", '').replace("{", '').replace("$", '').replace("@", '') \
                .replace("+", '').replace("/", '').replace("]", '').replace("[", '').replace(";", '') \
                .replace("~", '').replace("#", '').replace(",", '')
    else:
        return None


def ascii_ignore_pandas_impl(s: pd.Series) -> pd.Series:
    return (
        s.str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.strip()
        .str.replace(STRIP_CHARS, "", regex=True)
    )
