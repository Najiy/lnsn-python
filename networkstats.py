import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set()
sns.set(font_scale=1.250)
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})


fig, axes = plt.subplots(2)
# fig.suptitle('A single ax with no data')


def readfile(file="D:\\Shared Gits\\nssn-python\\states\\networks.meta"):
    df = pd.read_csv(file)
    df = df.drop("Unnamed: 10", axis=1)
    return df


df = readfile()
# df = df.loc[df['hashid'] == '8081e6eeacfa9df3fc0259c54c34b37b']

# print(df)

palette = sns.color_palette("mako_r", 6)


sns.lineplot(
    ax=axes[0],
    data=df.loc[df["hashid"] == "13615c3c82ab3360f4bcec0d56371719"],
    x="tick",
    y="neurones",
    palette=palette,
    label="SIN",
    linestyle="dotted",
    color="black",
    linewidth=2
)
sns.lineplot(
    ax=axes[0],
    data=df.loc[df["hashid"] == "e3c641eff1fa53e66ac9112268e65533"],
    x="tick",
    y="neurones",
    palette=palette,
    label="SAW",
    linestyle="dashed",
    color="black",
    linewidth=2
)
sns.lineplot(
    ax=axes[0],
    data=df.loc[df["hashid"] == "8081e6eeacfa9df3fc0259c54c34b37b"],
    x="tick",
    y="neurones",
    palette=palette,
    label="SINSAW",
    linestyle="solid",
    color="black",
    linewidth=2
)
# axes[0].legend(labels=["SIN","SAW", "SINSAW"])
axes[0].set_ylabel("Neurone Count")
axes[0].set_xlabel("Tick")


sns.lineplot(
    ax=axes[1],
    data=df.loc[df["hashid"] == "13615c3c82ab3360f4bcec0d56371719"],
    x="tick",
    y="npruned",
    palette=palette,
    label="SIN",
    linestyle="dotted",
    color="black",
    linewidth=2
)
sns.lineplot(
    ax=axes[1],
    data=df.loc[df["hashid"] == "e3c641eff1fa53e66ac9112268e65533"],
    x="tick",
    y="npruned",
    palette=palette,
    label="SAW",
    linestyle="dashed",
    color="black",
    linewidth=2
)
sns.lineplot(
    ax=axes[1],
    data=df.loc[df["hashid"] == "8081e6eeacfa9df3fc0259c54c34b37b"],
    x="tick",
    y="npruned",
    palette=palette,
    label="SINSAW",
    linestyle="solid",
    color="black",
    linewidth=2
)
# axes[0].legend(labels=["SIN","SAW", "SINSAW"])
axes[1].set_ylabel("Composites Pruned")
axes[1].set_xlabel("Tick")


# plt.legend(labels=["SINSAW","SAW", "SINE"])
# plt.xlabel("Tick")
# plt.ylabel("Neurone Counts")

plt.tight_layout()
plt.show()
