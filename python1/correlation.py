import pandas as pd
df = pd.read_excel("Process1.xlsx")
print(df.describe())
df_SM = df[["Sleep_quality","Mental_health"]]
df_SA = df[["Sleep_quality","Academic_performance"]]
df_AM = df[["Academic_performance","Mental_health"]]

def corr(df):
    corr_p = df.corr(method = "Pearson")
    corr_s = df.corr(method = "Spearman")
    return(corr_p, corr_s)

print(df.corr(method = "pearson"))
for i in (df_AM, df_SA, df_SM):
    print(i.corr(method = "spearman"))

