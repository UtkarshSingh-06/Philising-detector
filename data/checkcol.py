# import pandas as pd

# df = pd.read_csv("data/phishing.csv")
# print(df.columns)

import pandas as pd

df = pd.read_csv("data/phishing.csv")
print("📄 Columns in CSV:", df.columns.tolist())

if not {'url', 'Result'}.issubset(df.columns):
    raise ValueError("❌ 'url' and 'Result' columns are required!")
