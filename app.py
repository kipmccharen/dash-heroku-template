import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import dash
#from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from xgboost_analysis import make_xgboost_plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

gss_source = "https://raw.githubusercontent.com/kipmccharen/dash-heroku-template/master/gss2018.csv"
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

fig2 = px.scatter(gss_clean, x='job_prestige', y='income',
                color = 'sex',
                trendline='ols',
                height=600, width=600,
                labels={'job_prestige':'Job Prestige', 
                    'income':'Income'},
                hover_data=['education', 'socioeconomic_index'])
fig2.update(layout=dict(title=dict(x=0.5)))

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

gss6 = gss_clean[["income", "sex", "job_prestige"]]
gss6['job_prest_grp'] = pd.cut(gss6.job_prestige, 6,
                          labels = list(range(1,7)))
gss6 = gss6.dropna()

fig4 = px.box(gss6, x='sex', y = 'income',
            color = 'sex', facet_col='job_prest_grp', 
            color_discrete_map = {'male':'blue', 'female':'red'},
            facet_col_wrap=2,
            labels={'sex':'', 'value':''}, 
            width=1000, height=1000,
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

xgplot = make_xgboost_plot()

#app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1("The Wage Gap According To GSS Data"),
        
        dcc.Markdown(children = gender_wage_gap_discussion),
        dcc.Markdown(children = gss_description),
        
        html.H2("Income Violin Plots by Sex"),
        
        dcc.Graph(figure=violins),
        
        html.H2("Summary Data by Sex"),
        
        dcc.Graph(figure=table),
        
        html.H2("Agreement with Traditional Gender Roles by Sex"),
        
        dcc.Graph(figure=fig1),
        
        html.H2("Job Prestige vs Income, Colors by Sex"),
        
        dcc.Graph(figure=fig2),
        
        html.H2("Differences in Distribution by Sex\r\nof Income and Job Prestige"),
        
        dcc.Graph(figure=fig3),
        
        html.H2("Income Distributions by Sex\r\nacross equally sized Job Prestige Levels 1-6"),
        
        dcc.Graph(figure=fig4),
        
        html.H2("AI-Predicted % Importance on\r\nIncome (independent of other features)"),
        
        html.Div([html.Img(src=xgplot)])
        
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
    #mode='inline',  , port=8050