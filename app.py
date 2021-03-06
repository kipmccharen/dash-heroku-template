import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from dash import Dash
#from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from matplotlib import pyplot as plt
from matplotlib import colors as pltcolors
from io import BytesIO
import base64


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

gss_source = "https://raw.githubusercontent.com/kipmccharen/dash-heroku-template/master/gss2018.csv"
xgb_csv = r"https://raw.githubusercontent.com/kipmccharen/dash-heroku-template/master/gss_xgboost_df.csv"

gss = pd.read_csv(gss_source,
                encoding='cp1252', 
                na_values=['IAP','IAP,DK,NA,uncodeable', 'NOT SURE',
                'DK', 'IAP, DK, NA, uncodeable', '.a', "CAN'T CHOOSE"])

mycols = ['id', 'wtss', 'sex', 'educ', 'region', 'age', 'coninc',
          'prestg10', 'mapres10', 'papres10', 'sei10', 'satjob',
          'fechld', 'fefam', 'fepol', 'fepresch', 'meovrwrk'] 
gss_clean = gss[mycols]
gss_clean = gss_clean.rename({'wtss':'weight', 
                              'educ':'education', 
                              'coninc':'income', 
                              'prestg10':'job_prestige',
                              'mapres10':'mother_job_prestige', 
                              'papres10':'father_job_prestige', 
                              'sei10':'socioeconomic_index', 
                              'fechld':'relationship', 
                              'fefam':'male_breadwinner', 
                              'fehire':'hire_women', 
                              'fejobaff':'preference_hire_women', 
                              'fepol':'men_bettersuited', 
                              'fepresch':'child_suffer',
                              'meovrwrk':'men_overwork'},axis=1)

cat_order = ['strongly agree', 'agree', 
    'neither agree nor disagree', 
    'disagree', 'strongly disagree']
cat_type = CategoricalDtype(categories=cat_order,
                ordered=True)
colorder = ['relationship', 'male_breadwinner',
    'men_bettersuited', 'child_suffer', 'men_overwork']

for c in colorder:
    gss_clean[c] = gss_clean[c].astype(cat_type)

cat_type_sex = CategoricalDtype(categories=["female", "male"],
                ordered=True)
gss_clean['sex'] = gss_clean['sex'].astype(cat_type_sex)

gss_clean.age = gss_clean.age.replace({'89 or older':'89'})
gss_clean.age = gss_clean.age.astype('float')

gender_wage_gap_discussion = """## What is the Gender Wage Gap?

"Working women are paid less than working men", as concisely reported 
by the [Economic Policy Institute](https://www.epi.org/publication/what-is-the-gender-pay-gap-and-is-it-real/) (EPI).
Unfortunately this observation on differences in pay by sex appears to be consistent over time and
almost no matter how the data is measured. The EPI publication also discusses how the wage gap 
describes merely one dimension of discrimination against women and fails to capture the full depth 
of other problems that women face in the workforce, sometimes called the "glass ceiling".

While the gap in pay between men and women has narrowed since 1980, according to the [Pew Research Center](https://www.pewresearch.org/fact-tank/2019/03/22/gender-pay-gap-facts/), 
there has been little change since about 2005. In addition, women reported more discrimination 
in workplaces for each possible category inquired, according to the same Pew publication. 

One possible source of a sustained pay gap is the fact that women give birth to children,
which is a full time life event requiring time away from work. Unfortunately, despite laws which
protect maternity leave, significantly more women reported to Pew that motherhood has lead
to career interruptions which can have negative impacts on long-term earnings."""

gss_description = """## What is the GSS?

The General Social Survey (GSS) is a survey conducted every year by the University of Chicago's 
National Opinion Research Center ([NORC](https://www.norc.org/about/Pages/about-our-name.aspx)). 

The GSS attempts to track attitudes, behaviors, and personal attributes of americans across time, 
since the survey began in 1972. Over 6,000 variables are tracked by the GSS including basic details
such as survey year and respondant age, as well as more in-depth details such as travel time to work,
which presidential candidate the respondant voted for, and the respondant's level of agreement with
"It is much better for everyone involved if the man is the achiever outside the home and the woman 
takes care of the home and family" [see here](https://gssdataexplorer.norc.org/variables/706/vshow). 

This analysis is primarily about respondants' sex (male/female), income (annual year-standardized),
job prestige (independently rated in a [2012 lab study](https://gss.norc.org/Documents/reports/methodological-reports/MR122%20Occupational%20Prestige.pdf)), and finally respondant-reported 
levels of agreement with statements (like the one above) related to sex, family life, and work. 
"""

#%%
gss_grp = gss_clean[["sex", "income", "job_prestige", \
                "socioeconomic_index", "education"]]
gss_grp = round(gss_grp.groupby("sex").mean(),2).reset_index()
gss_grp.columns = [x.replace("_", " ").title() \
                for x in list(gss_grp.columns)]

table = ff.create_table(gss_grp)

bread = gss_clean[["sex", "male_breadwinner"]]
bread = bread.value_counts().reset_index()
bread.columns = ["Sex", "Male Breadwinner", "Count"]

fig1 = px.bar(bread, x="Male Breadwinner", y="Count", color='Sex',
            labels={'Count':'Count Of Response Selection', 
            'Male Breadwinner':'Agreement levels to: <br>"It is much better for everyone involved <br>if the man is the achiever outside the home <br>and the woman takes care of the home and family."'},
            barmode = 'group')
fig1.update_layout(showlegend=True)
fig1.update(layout=dict(title=dict(x=0.5)))

#%%
fig2 = px.scatter(gss_clean, x='job_prestige', y='income',
                color = 'sex',
                trendline='ols',
                #height=600, width=600,
                labels={'job_prestige':'Job Prestige', 
                    'income':'Income'},
                hover_data=['education', 'socioeconomic_index'])
fig2.update(layout=dict(title=dict(x=0.5)))

#%%
gss_grp = pd.melt(gss_clean, id_vars=["sex"], 
                value_vars=["income","job_prestige"], 
                var_name="variable", 
                value_name='value', ignore_index=True)

fig3 = px.box(gss_grp, x='sex', y = 'value', 
            color = 'sex',
                facet_col='variable', 
            labels={'sex':'', 'value':''})
            
fig3.for_each_annotation(lambda a: a.update( \
    text=a.text.replace("variable=", "").replace("_", " ").title()))
fig3.update_yaxes(matches=None)
fig3.update_layout(showlegend=False)
fig3.update(layout=dict(title=dict(x=0.5)))

#%%
gss6 = gss_clean[["income", "sex", "job_prestige"]]
gss6['job_prest_grp'] = pd.cut(gss6.job_prestige, 6,
                          labels = list(range(1,7)))
gss6 = gss6.dropna()

fig4 = px.box(gss6, x='sex', y = 'income',
            color = 'sex', facet_col='job_prest_grp', 
            color_discrete_map = {'male':'blue', 'female':'red'},
            facet_col_wrap=2,
            labels={'sex':'', 'value':''}, 
            #width=600, 
            height=600,
            hover_data=['job_prestige'])
            
fig4.for_each_annotation(lambda a: a.update( \
    text=a.text.replace("variable=", "").replace("_", " ").title()))
fig4.update_yaxes(matches=None)
fig4.update_layout(showlegend=False)
fig4.update(layout=dict(title=dict(x=0.5)))

sexcompare_avg = gss_clean.groupby('sex')['income'].mean()
sexcompare_avg = sexcompare_avg.round(2).reset_index()
sc_vals_avg = sexcompare_avg['income'].tolist()
sx = ["female", "male"]

sexcompare_med = gss_clean.groupby('sex')['income'].median()
sexcompare_med = sexcompare_med.round(2).reset_index()
sc_vals_med = sexcompare_med['income'].tolist()

violins = px.violin(gss_clean, x="income", 
            color="sex", box=True, 
            points="all",
            hover_data=gss_clean.columns)


#%%
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

xg_df = pd.read_csv(xgb_csv)

#add color to each feature from dict above
xg_df['color'] = xg_df.feature.apply(addcolor)

#set base font size
plt.rcParams.update({'font.size': 10})

#create plot and set figure size
statfig,statax = plt.subplots(figsize = (8,12))

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
plt.gcf().subplots_adjust(left=0.4, bottom=0.01, top=1)

xgplot = fig_to_uri(statfig)

#%%
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
# ;
# ;
# style={'fontColor': 'blue', 'background'='rgb(2,0,36)',
#   'background'= 'linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(5,31,145,1) 15%, rgba(27,144,237,1) 100%)'}

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    #"background-color": "#f8f9fa",
}



sidebar = html.Div(
    [
        html.H2("Pages", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Gender Wage Gap & GSS", 
                    href="/wage_gap-gss", id="wage_gap-gss"),
                dbc.NavLink("Income Violin Plots by Sex", 
                    href="/violin", id="violin"),
                dbc.NavLink("Summary Data by Sex", 
                    href="/table", id="table"),
                dbc.NavLink("Traditional Gender Role Agreement", 
                    href="/roles", id="roles"),
                dbc.NavLink("Job Prestige/Income/Sex", 
                    href="/prestige", id="prestige"),
                dbc.NavLink("Income and Prestige Distributions",
                    href="/diff_dist", id="diff_dist"),
                dbc.NavLink("Prestige Levels", 
                    href="/income_dist", id="income_dist"),
                dbc.NavLink("AI Importance", 
                    href="/AI", id="AI")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

styledict = {#'fontColor': 'rgb(253,253,253)', 
            #'font': '10px Calibri',
            #'border':'1px solid black',
            'margin-bottom':'50px', 
            'margin-left':'auto',
            'margin-right':'auto', 
            'text-align':'center',
            'max-width': '700px',
            #'background-color': 'white'
            }
#backgroundcolor = {'background':'rgb(135,206,235)'}
textlayout = {#'fontColor': 'rgb(253,253,253)', 
                'margin-left':'auto',
                'margin-right':'auto', 
                'max-width': '800px'}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

pages = ['wage_gap-gss', 'violin', 'table', 'roles', 
        'prestige', 'diff_dist', 'income_dist', 'AI']
pagecount = len(pages)
# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, pagecount)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/{i}" for i in pages]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/wage_gap-gss"]:
        return html.P([html.Div([
                html.H1("The Wage Gap According To GSS Data"),
                dcc.Markdown(children = gender_wage_gap_discussion),
            ],id='wage_gap',style=textlayout) ,
        html.Div([
                dcc.Markdown(children = gss_description)
            ],id='gss',style=textlayout)])
    elif pathname == "/violin":
        return html.P([
                html.H2("Income Violin Plots by Sex"),
                dcc.Graph(figure=violins),
                dcc.Markdown(children = """According to data from the GSS, do men have higher incomes than women?
                
                While complicated, it appears that the answer is yes. While the median income for men and women is the same, the average is lower for women, and we can see both the first and third quartile breaks are lower for women. """)
            ])
    elif pathname == "/table":
        return html.P([
                html.H2("Summary Data by Sex"),
                dcc.Graph(figure=table),
                dcc.Markdown(children = """The average differences in prestige, socioeconomic index, and education are not very different between men and women. It's peculiar that income is so much more different than these other factors.""")
            ])
    elif pathname == "/roles":
        return html.P([
                html.H2("Agreement with Traditional Gender Roles by Sex"),
                dcc.Graph(figure=fig1),
                dcc.Markdown(children = """In terms of agreeing that women should take care of the home and family, both sexes are in generally similar ratios in all categories except for strongly disagree which is about 2/3 women.""")
            ])
    elif pathname == "/prestige":
        return html.P([
                html.H2("Job Prestige vs Income, Colors by Sex"),
                dcc.Graph(figure=fig2),
                dcc.Markdown(children = """Job prestige has a very similar impact on income between men and women, the average lines are in the same direction and almost lining up but not quite. Women's income grows slightly less across prestige levels.""")
            ])
    elif pathname == "/diff_dist":
        return html.P([
                html.H2("Differences in Distribution by Sex\r\nof Income and Job Prestige"),
                dcc.Graph(figure=fig3),
                dcc.Markdown(children = """Building off the violin plots, if we compare the difference in job prestige between men and women, they are almost the same at every point. On average, women even have higher prestige jobs than men, albeit with a lower window. These don't seem to agree at all!""")
            ])
    elif pathname == "/income_dist":
        return html.P([
                html.H2("Income Distributions by Sex\r\nacross equally sized Job Prestige Levels 1-6"),
                dcc.Graph(figure=fig4),
                dcc.Markdown(children = """Even if we break down groups in 6 separate categories (1 is lowest, 6 is highest prestige), in terms of actually getting paid given their prestige, once again we see either parity or higher male income at every part of the distribution.""")
            ])
    elif pathname == "/AI":
        return html.P([
                html.H2("AI-Predicted % Importance on\r\nIncome (independent of other features)"),
                html.Div([html.Img(src=xgplot)]),
                dcc.Markdown(children = """Independent of other features, it seems like age, prestige, and education have the biggest overall impact on income, and as each one increases, so too does income. These are not particularly surprising or interesting, so they are all black. 

The other bar colors demonstrate that sex had a larger impact than any belief or hometown region category. The percent impact color tells us that being female had a negative impact on income. Other factors that had a negative impact on income seem to be beliefs which are less than extreme (agree, disagree, or neither agree nor disagree), or in short: apathy. """)
            ])
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# app.layout = html.Div(
#     [   dcc.Location(id="url"), sidebar, 
#         html.Div([
#                 html.H1("The Wage Gap According To GSS Data"),
#             ],id='h1_top',style=textlayout) ,
#         html.Div([
#                 dcc.Markdown(children = gender_wage_gap_discussion),
#             ],id='wage_gap',style=textlayout) ,
#         html.Div([
#                 dcc.Markdown(children = gss_description)
#             ],id='gss',style=textlayout) ,
        
#         html.Div([
#                 html.H2("Income Violin Plots by Sex"),
#                 dcc.Graph(figure=violins),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_violin',style=styledict) ,

#         html.Div([
#                 html.H2("Summary Data by Sex"),
#                 dcc.Graph(figure=table),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_table',style=styledict) ,
        
#         html.Div([
#                 html.H2("Agreement with Traditional Gender Roles by Sex"),
#                 dcc.Graph(figure=fig1),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_roles',style=styledict) ,
        
#         html.Div([
#                 html.H2("Job Prestige vs Income, Colors by Sex"),
#                 dcc.Graph(figure=fig2),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_prestige',style=styledict) ,
        
#         html.Div([
#                 html.H2("Differences in Distribution by Sex\r\nof Income and Job Prestige"),
#                 dcc.Graph(figure=fig3),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_diff_dist',style=styledict) ,


#         html.Div([
#                 html.H2("Income Distributions by Sex\r\nacross equally sized Job Prestige Levels 1-6"),
#                 dcc.Graph(figure=fig4),
#                 dcc.Markdown(children = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.""")
#             ],id='h2_income_dist',style=styledict) ,
        
#         html.Div([
#                 html.H2("AI-Predicted % Importance on\r\nIncome (independent of other features)"),
#                 html.Div([html.Img(src=xgplot)]),
#                 dcc.Markdown(children = """Independent of other features, it seems like age, prestige, and education have the biggest overall impact on income, and as each one increases, so too does income. These are not particularly surprising or interesting, so they are all black. 

# The other bar colors demonstrate that sex had a larger impact than any belief or hometown region category. The percent impact color tells us that being female had a negative impact on income. Other factors that had a negative impact on income seem to be beliefs which are less than extreme (agree, disagree, or neither agree nor disagree), or in short: apathy. """)
#             ],id='h2_AI',style=styledict) ,
        
#     ] #, style=backgroundcolor
# )

if __name__ == '__main__':
    app.run_server(debug=True)