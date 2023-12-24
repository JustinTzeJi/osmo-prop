import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from numpy import nan
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from flipside import Flipside
import re
import collections
import base64

st.set_page_config(
    page_title="Attempt to classify Osmosis validators based on voting activity",
    page_icon="⚗️",
    layout="wide",
    menu_items={
        "Report a Bug": "https://github.com/JustinTzeJi/osmo-prop/issues",
        "About": "An experiment",
    },
)


class fsdata:
    def __init__(self):
        self.sdk = Flipside(st.secrets["api_keys"], "https://api-v2.flipsidecrypto.xyz")
        self.vote_data = """SELECT
  * EXCLUDE (
    MEMO,
    VERSION,
    CREATED_AT,
    FACT_VALIDATOR_VOTES_ID,
    INSERTED_TIMESTAMP,
    MODIFIED_TIMESTAMP
  )
FROM
  osmosis.gov.fact_governance_validator_votes"""
        self.prop_info = """SELECT proposal_id, proposal_description
FROM osmosis.gov.fact_governance_submit_proposal"""

        self.query_vote()
        self.query_prop()
        self.list_of_prop_()

    def query_vote(self):
        vote_df = pd.DataFrame(self.sdk.query(self.vote_data).records)
        vote_df = vote_df.rename(columns={"proposal_id":"PROPOSAL_ID", "validator":"VALIDATOR","validator_address":"VALIDATOR_ADDRESS","vote":"VOTE","voting_power":"VOTING_POWER"})
        self.vote_df = vote_df.copy()  # raw data
        vote_df.loc[
            vote_df[vote_df.VALIDATOR.isnull()].loc[:, "VALIDATOR"].index, "VALIDATOR"
        ] = vote_df[vote_df.loc[:, "VALIDATOR"].isnull()].loc[:, "VALIDATOR_ADDRESS"]
        vote_df = vote_df.pivot(
            index=["VALIDATOR", "VALIDATOR_ADDRESS"],
            columns="PROPOSAL_ID",
            values="VOTE",
        )
        vote_df = vote_df.replace(
            {float("nan"): -3, "NoWithVeto": -2, "No": -1, "Abstain": 0, "Yes": 1}
        )
        self.prop_votes = vote_df
        return self.prop_votes

    def query_prop(self):
        self.prop_df_ = pd.DataFrame(self.sdk.query(self.prop_info).records)
        return self.prop_df_

    def list_of_prop_(self):
        self.list_of_prop  = self.prop_votes.columns.to_list()
        return self.list_of_prop 

class analytics:
    def __init__(self, fsdata, index1, index2):
        self.val_data = fsdata.prop_votes
        self.main_data = fsdata.prop_votes
        self.index1 = index1
        self.index2 = index2
        self.pca_fn()
        self.cluster_fn()
        self.mergeclusterdata()

    def pca_fn(self):
        pca = PCA(n_components=3)
        self.Xt = pd.DataFrame(pca.fit_transform(self.val_data.reset_index(drop=True)))
        self.pca_data = self.Xt.copy()
        self.pca_data.index = self.val_data.index
        self.pca_data = self.pca_data.reset_index(drop=False)
        self.pca_components = pca.components_
        return self.Xt, self.pca_data, self.pca_components

    def cluster_fn(self):
        self.cluster = (
            KMeans(n_clusters=3, init="k-means++")
            .fit(self.val_data.reset_index(drop=True))
            .labels_
        )
        return self.cluster

    def mergeclusterdata(self):
        self.main_data =  self.main_data.reset_index().join(pd.Series(self.cluster, name="cluster"))
        return self.main_data


class keybert:
    def __init__(self, analytics, fsdata):
        self.cluster_data = analytics.main_data
        self.prop_df_ = fsdata.prop_df_
        self.cluster = analytics.cluster
        self.cluster_dict()
        self.keybert_dict()

    def cluster_dict(self):
        self.cluster_count = {}
        self.unique_clusters = sorted(np.unique(self.cluster))
        for cluster in self.unique_clusters:
            df = self.cluster_data[self.cluster_data["cluster"] == cluster]
            countdata = (
                df.drop(columns=["VALIDATOR_ADDRESS", "VALIDATOR", "cluster"])
                .apply(lambda x: x.value_counts())
                .T.stack()
                .to_frame()
                .reset_index()
                .rename(
                    columns={"level_0": "proposal_id", "level_1": "vote", 0: "count"}
                )
            )
            self.cluster_count[cluster] = countdata
        return self.cluster_count, self.unique_clusters

    def extract(self, doc):
        model = KeyBERT()
        keywords = model.extract_keywords(
            doc, vectorizer=KeyphraseCountVectorizer(stop_words="english"), use_mmr=True
        )
        return keywords

    def keybert_dict(self):
        vote_option = [-2, -1, 1]
        self.clusterkeyword = collections.defaultdict(dict)
        for cluster in self.cluster_count:
            for option in vote_option:
                df = self.cluster_count[cluster]
                top20prop = (
                    df[df["vote"] == option]
                    .sort_values(by=["count"], ascending=False)
                    .head(n=20)
                    .proposal_id.to_list()
                )
                top20prop_words = (
                    re.sub(
                        r"\S*https?:\S*",
                        "",
                        "".join(
                            str(x)
                            for x in self.prop_df_[
                                self.prop_df_.proposal_id.isin(top20prop)
                            ]["proposal_description"].to_list()
                        ),
                    )
                    .strip()
                    .replace("\\n", " ")
                )
                keywords = self.extract(top20prop_words)
                self.clusterkeyword[cluster][option] = keywords
        return self.clusterkeyword


class displays:
    def __init__(self, fsdata, analytics, keybert):
        self.fsdata = fsdata
        self.analytics = analytics
        self.keybert = keybert
        self.display_df_()
        self.df_heatmap_()
        self.df_pca_()
        self.pca_bar_()
        self.voting_power_()
        self.display_keybert()

    def display_df_(self):
        df = self.analytics.main_data
        df = df.replace(
            {-3: "Did Not Vote", -2: "No With Veto", -1: "No", 0: "Abstain", 1: "Yes"}
        )
        # df = df.drop(columns=["cluster"])
        self.display_df_bar = df
        df = df.set_index(["VALIDATOR", "VALIDATOR_ADDRESS"])
        self.display_df = df

    def df_heatmap_(self):
        # ---generate color and colorbar for heatmap
        bvals = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5])
        colors = ["#000000", "#ff0000", "#ff7a00", "#676767", "#228b22"]
        ticktext = ["DID NOT VOTE", "NO WITH VETO", "NO", "ABSTAIN", "YES"]
        tickvals = [-3, -2, -1, 0, 1]
        nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
        dcolorscale = []
        for k in range(len(colors)):
            dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
		
        df_heatmap = self.analytics.main_data.copy()
        # df_heatmap.index = df_heatmap.index.droplevel("VALIDATOR_ADDRESS")
        df_heatmap = df_heatmap.reset_index(drop=True).sort_values(by=["cluster", "VALIDATOR"])
        self.custom_data = (
            df_heatmap.loc[:, df_heatmap.columns != "cluster"]
            .replace(
                {
                    -3: "Did Not Vote",
                    -2: "No With Veto",
                    -1: "No",
                    0: "Abstain",
                    1: "Yes",
                }
            )
            .join(df_heatmap.loc[:, "cluster"])
        )

        # ----generate heatmap
        heatmap = go.Heatmap(
            z=df_heatmap.drop(columns=["cluster", "VALIDATOR","VALIDATOR_ADDRESS"]),
            x=df_heatmap.drop(columns=["cluster", "VALIDATOR","VALIDATOR_ADDRESS"]).columns,
			y=df_heatmap["VALIDATOR"]
            + " - "
            + "cluster "
            + df_heatmap["cluster"].astype(str),
            colorscale=dcolorscale,
            customdata=self.custom_data,
            hovertemplate="<b>%{y} </b><br>" + "Prop %{x}<br>" + "Vote: %{customdata} ",
            colorbar={"thickness": 10, "tickvals": tickvals, "ticktext": ticktext},
            name="",
        )
        fig = go.Figure(data=[heatmap]).update_layout(
            title="Osmosis validator voting heatmap",
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            height=900,
            yaxis={"dtick": 1},
        )
        self.df_heatmap = fig

    def df_pca_(self):
        fig = px.scatter_3d(
            self.analytics.pca_data,
            x=0,
            y=1,
            z=2,
            color=list(map(str, self.analytics.main_data.cluster)),
            hover_name="VALIDATOR",
            labels={"0": "X", "1": "Y", "2": "Z"},
            title="Predicted clusters",
        )
        fig.update_traces(marker_size=7)
        fig.update_layout(legend_title_text="Cluster")
        self.df_pca = fig

    def pca_bar_(self):
        self.axis_top_dict = collections.defaultdict(dict)
        self.pca_bar = collections.defaultdict(dict)
        for x, axis in enumerate(["X", "Y", "Z"], 0):
            top10 = pd.DataFrame(np.abs(self.analytics.pca_components[x]))
            top10["prop"] = pd.DataFrame(self.analytics.val_data.columns)
            top10 = top10.nlargest(10, 0, "all")
            toplist = ""
            for t, i in enumerate(top10["prop"].to_list(), 1):
                toplist += str(t) + ". Prop " + str(i) + " \n"
            self.axis_top_dict[axis][toplist] = toplist
            # self.display_df_bar['cluster'] = self.analytics.cluster

            self.bar_columns_1 = []
            options = {
                "Yes": "green",
                "No": "orange",
                "No With Veto": "red",
                "Abstain": "grey",
                "Did Not Vote": "black",
            }
            for prop in top10["prop"].to_list():
                grouping_df = self.custom_data.groupby("cluster")[prop].value_counts()
                grouping_df = grouping_df.to_frame("counts").reset_index()
                fig = go.Figure()
                for option in options:
                    fig.add_trace(
                        go.Bar(
                            y=grouping_df[grouping_df[prop] == option][
                                "cluster"
                            ].astype(str),
                            x=grouping_df[grouping_df[prop] == option]["counts"],
                            name=option,
                            orientation="h",
                            marker=dict(
                                color=options[option],
                            ),
                        )
                    )
                fig.update_layout(
                    barmode="stack",
                    title="Voting statistics of each cluster on Prop %d" % prop,
                    titlefont=dict(size=14),
                    height=300,
                )
                self.bar_columns_1.append(fig)
            self.bar_columns_2 = []
            for prop in top10["prop"].to_list()[5:]:
                grouping_df = self.custom_data.groupby("cluster")[prop].value_counts()
                grouping_df = grouping_df.to_frame("counts").reset_index()
                fig = go.Figure()
                for option in options:
                    fig.add_trace(
                        go.Bar(
                            y=grouping_df[grouping_df[prop] == option][
                                "cluster"
                            ].astype(str),
                            x=grouping_df[grouping_df[prop] == option]["counts"],
                            name=option,
                            orientation="h",
                            marker=dict(
                                color=options[option],
                            ),
                        )
                    )
                fig.update_layout(
                    barmode="stack",
                    title="Voting statistics of each cluster on Prop %d" % prop,
                    titlefont=dict(size=14),
                    height=300,
                )
                self.bar_columns_2.append(fig)
            self.pca_bar[axis][1] = self.bar_columns_1
            self.pca_bar[axis][2] = self.bar_columns_2

    def voting_power_(self):
        cluster_power = self.fsdata.vote_df.merge(
            self.analytics.main_data[["VALIDATOR_ADDRESS", "cluster"]],
            how="left",
            on="VALIDATOR_ADDRESS",
        )
        cluster_power["cluster"] = cluster_power["cluster"].astype("str")
        cluster_power = (
            cluster_power[["cluster", "PROPOSAL_ID", "VOTING_POWER"]]
            .groupby(["cluster", "PROPOSAL_ID"])
            .sum()
            .reset_index()
        )
        cluster_power = cluster_power.merge(
            cluster_power[
                [
                    "VOTING_POWER",
                    "PROPOSAL_ID",
                ]
            ]
            .groupby(["PROPOSAL_ID"])
            .sum()
            .reset_index(),
            on="PROPOSAL_ID",
        )
        cluster_power["ratio(%)"] = (
            cluster_power["VOTING_POWER_x"] / cluster_power["VOTING_POWER_y"] * 100
        )
        self.voting_power = px.bar(
            cluster_power,
            x="PROPOSAL_ID",
            y="ratio(%)",
            color="cluster",
            title="Estimated* voting power of clusters",
        ).update_yaxes(showgrid=False)
        self.voting_power.update_layout(
            xaxis_title="Proposal ID", yaxis_title="Voting Power (%)"
        )

    def display_keybert(self):
        self.keybert_chart_ = collections.defaultdict(dict)
        unique_clusters = sorted(self.analytics.main_data.cluster.unique())
        list = {1: ["yes", "green"], -1: ["no", "orange"], -2: ["no-with veto", "red"]}
        for cluster in unique_clusters:
            keybert_chart = []
            cluster_num = "Cluster " + str(cluster)
            for option in self.keybert.clusterkeyword[cluster]:
                df = pd.DataFrame(
                    self.keybert.clusterkeyword[cluster][option],
                    columns=["phrase", "relevance"],
                )
                fig = px.bar(
                    df,
                    x="relevance",
                    y="phrase",
                    orientation="h",
                    title=f"Keywords/phrases from {list[option][0]}-voted proposals",
                ).update_traces(marker_color=list[option][1])
                keybert_chart.append(fig)
                self.keybert_chart_[cluster_num] = keybert_chart


def render_svg(svg_file):
    html_svg = []
    for svg in svg_file:
        with open(svg, "r") as f:
            lines = f.readlines()
            svg_ = "".join(lines)
            b64 = base64.b64encode(svg_.encode("utf-8")).decode("utf-8")
            html_svg.append(
                r'<img src="data:image/svg+xml;base64,%s" width="100"/>' % b64
            )
    all_svg = "".join(html_svg)
    html = r'<div align="center">%s</div>' % all_svg
    return html

@st.cache_data
def fs():
    st.markdown(
        "```Querying data from Osmosis API and KeyBERT processing will take a moment```"
    )

    with st.spinner(text="Querying data from Flipside API"):
        data = fsdata()
    return data


@st.cache_data
def analytika(_data, index1, index2):
    with st.spinner(text="Analyzing Data"):
        analysis = analytics(_data, index1, index2)
    with st.spinner(
        text="Extracting keywords & phrases with KeyBERT"
    ):
        keybert_analysis = keybert(analysis,_data)
    with st.spinner(text="Generating visuals"):
        display_visuals = displays(_data, analysis, keybert_analysis)
    return analysis, keybert_analysis, display_visuals


if __name__ == "__main__":
    st.markdown(render_svg(["osmo.svg", "scientist.svg"]), unsafe_allow_html=True)
    st.title(
        "Osmosis validator onchain *polictics*"
    )
    """
		[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/JustinTzeJi/osmo-prop) 

	"""
    st.markdown("<br>", unsafe_allow_html=True)
    data = fs()
    with st.container():
        st.header("The Problem Statement")
        st.markdown(
            'With the ever growing amount and ever changing list of validators on the Osmosis chain, how does one evaluate and select validator with similar *on-chain "politics"/viewpoints* as oneself. It is a laborious task to monitor and keep up with validator socials, and voting activities on each proposal.'
        )
        st.header('The "Solution"?')
        st.markdown(
            "Using the on-chain validator voting habits, we can use clustering algorithms to attempt to aggregate validators with similar votes. We can also extract prominent/recurring phrases that occurs in proposal descriptions to observe how said cluster of validators vote on a very surface level."
        )
        st.header("Limitations and Caveats")
        st.markdown(
            """
		1. Osmosis proposals are not usually polarized (for a reference of polarized proposals, see Cosmos Hub), hence *politics* might not be observable onchain?
		2. A large percentage of proposal description includes URL to the complete proposals are removed, due to difficulty of parsing each and every type of website, document.
		3. This dashboard does not account for other various validator stats, such as commission rate, voting power and missed blocks (see [Smart Stake](https://osmosis.smartstake.io/))
		4. As the size of validators have increased multiple times, clustering might reflect newer vs older validators
		5. It's just a fun project for me, don't take it too seriously :D 
        
        UPDATE 25/12/2023:
        Flipside updated their database and Proposal and Governance data now only start from Proposal 394
		"""
        )

    list_of_prop = data.list_of_prop
    selected1, selected2 = st.select_slider(
        "Select the range of proposals",
        options=list_of_prop,
        value=(list_of_prop[0], list_of_prop[-1]),
    )

    analysis, keybert_analysis, display_visuals = analytika(
        data, list_of_prop.index(selected1), list_of_prop.index(selected2)
    )

    with st.expander("Preview of data"):
        st.dataframe(display_visuals.display_df)

    with st.container():
        st.header("Voting activity of Osmosis Validators")
        st.plotly_chart(display_visuals.df_heatmap, use_container_width=True)

    with st.container():
        st.header("Clustering of validators")

        st.markdown(
            """
		1. Voting data is dimensionaly reduced using *Principal Component Analysis* (PCA).
		2. Validators with similar voting patterns are clustered using *K-Means* algorithm (using a default of 3 cluster size).
		"""
        )

    st.plotly_chart(display_visuals.df_pca, use_container_width=True)
    with st.expander("Validators in each cluster"):
        a_data = analysis.main_data
        for column, cluster in zip(
            st.columns(len(keybert_analysis.unique_clusters)),
            keybert_analysis.unique_clusters,
        ):
            column.dataframe(
                a_data[a_data["cluster"] == cluster]
                .reset_index()
                .rename(
                    mapper={"VALIDATOR": "Validators in Cluster %s" % cluster},
                    axis="columns",
                )["Validators in Cluster %s" % cluster]
            )

    st.markdown(
        """
	### Understanding the axes:
	Each axis has its own weight for a set of Proposal. By looking at the top proposals with the heighest weigths in each axis, we can get a glimpse of the proposals that seperate clusters from one another. 
	"""
    )
    for axis in display_visuals.pca_bar:
        with st.expander("Top 10 Proposals for %s-axis:" % axis):
            for i, column in enumerate(st.columns(5)):
                column.plotly_chart(
                    display_visuals.pca_bar[axis][1][i], use_container_width=True
                )
            for i, column in enumerate(st.columns(5)):
                column.plotly_chart(
                    display_visuals.pca_bar[axis][2][i], use_container_width=True
                )

    with st.container():
        st.header("Voting power of each cluster")
        st.markdown(
            """
		The ratio of of voting power of clusters on each proposal
		"""
        )
        st.plotly_chart(display_visuals.voting_power, use_container_width=True)

    with st.container():
        st.header("Keywords from each Clusters")
        st.markdown(
            """
		### Methodology:
		1. Top 20 Proposal (with highest number of same votes) of each type of vote (`YES`, `NO`, `NO WITH VETO`) are selected
		2. The proposal descriptions are retreived from Osmosis LCD, URLs in the description are removed.
		3. Proposals description from each category and cluster is concatenated and keywords/phrases are extracted using `KeyBERT` algorithm, with `KeyphraseVectorizers()` and  `Maximal Margin Relevance`
		"""
        )
        for cluster in display_visuals.keybert_chart_:
            with st.expander(cluster):
                for chart in display_visuals.keybert_chart_[cluster]:
                    st.plotly_chart(chart, use_container_width=True)
