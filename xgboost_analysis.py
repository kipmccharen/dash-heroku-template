import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as pltcolors
import pandas as pd 
from io import BytesIO
import base64

rawcsv = r"https://raw.githubusercontent.com/kipmccharen/dash-heroku-template/master/gss_xgboost_df.csv"

#define categories of items
labelsdicts = [{"name": "Born Female", "inkey": ["SEX: Female"], 
                "order": 2, "color":"orangered"},
        {"name": "Hometown Region", "inkey": ["REGION"], 
                "order": 4, "color":"dimgrey"},
        {"name": "Beliefs", "inkey": ["gree", "isfied", "Dissat"], 
                "order": 3, "color":"midnightblue"},
        {"name": "Upbringing/Age", "inkey": [], 
                "order": 1, "color":"black"}]

def anylist_in_string(src_list, string):
    """Return True if any value in [src_list] 
    exists in given [string]."""
    if not src_list or not string:
        return False
    for sl in src_list:
        if sl in string:
            return True
    return False

def addcolor(x):
    """Vectorized color value from lookup dictionary."""
    out = None
    for ld in [a for a in labelsdicts if a["inkey"]]:
        if anylist_in_string(ld['inkey'], x):
            out = ld["color"]
    if not out:
        out = "black"
    return pltcolors.to_rgba(out)

#from https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb
def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

def make_xgboost_plot():
    xg_df = pd.read_csv(rawcsv)

    #add color to each feature from dict above
    xg_df['color'] = xg_df.feature.apply(addcolor)

    #set base font size
    plt.rcParams.update({'font.size': 14})

    #create plot and set figure size
    statfig,statax = plt.subplots(figsize = (7,12))

    #create bar chart
    statax.barh(xg_df.feature, xg_df.imp_pct, align='center')
    #add feature names as labels
    statax.set_yticks(xg_df.feature)
    #turn on size to make horizontal bar chart
    statax.invert_yaxis()
    #remove outer box and x axis as unnecessary
    statax.spines["top"].set_visible(False)
    statax.spines["right"].set_visible(False)
    statax.spines["bottom"].set_visible(False)
    statax.get_xaxis().set_visible(False)

    # for each feature:
    # - add text to end of bar
    # - color the bar itself
    for i in range(len(xg_df.index)):
        rw = xg_df.iloc[i,:]

        #add text to end of each bar with appropriate color
        statax.text(x=rw['imp_pct'] +0.1, y=i, 
                s=str(round(rw['imp_pct'],2)) + '%',
                va = 'center', 
                color=rw["positive"],
                fontweight='bold')
        #statax color of each bar
        statax.get_children()[i].set_color(rw["color"])

    tsh = 12 #text start height

    #for each category of bar, add text in the plot
    for use_item in labelsdicts:
        statax.text(7, tsh-6+use_item["order"]*1.2,
                use_item["name"], 
                color=pltcolors.to_rgba(use_item["color"]), 
                fontsize='large', 
                fontweight='bold')

    #add text explaining meaning of number color
    statax.text(7,tsh + 1.5,"reduces",color="red",fontweight='bold')
    statax.text(11,tsh + 1.5,"increases",color="black",fontweight='bold')

    #final labels
    #statax.set_title("AI-Predicted % Importance on Income\n(independent of other features)")
    #statax.text(5,tsh + 4,"Source: 2018 General Social Survey (GSS)")
    statax.set_ylabel('')

    return fig_to_uri(statfig)