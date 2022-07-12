import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.cluster import KMeans
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go



st.set_page_config(page_title="Attempt to classify Osmosis validators based on voting activity", page_icon="⚗️", layout="wide", menu_items={'Report a Bug': "https://github.com/JustinTzeJi/osmo-prop/issues",'About': "An experiment"})

API_KEY = st.secrets["api_keys"]
SQL_QUERY1 = """
with b as (SELECT voter, proposal_id, vote_option
FROM osmosis.core.fact_governance_votes
WHERE tx_status = 'SUCCEEDED'
)

select '
select * 
from b 
  pivot (min(vote_option) for proposal_id in ('||listagg (distinct ''||proposal_id||'', ',') ||')) 
order by voter' 
from b;
"""

PIVOT_QUERY = """
with a as (SELECT raw_metadata[0]:account_address as address, label
FROM osmosis.core.dim_labels
WHERE label_subtype = 'validator'),

b as (SELECT voter, a.label, proposal_id, vote_option
FROM osmosis.core.fact_governance_votes b inner join a on b.voter = a.address
WHERE tx_status = 'SUCCEEDED')

--select * 
--from b
--  pivot (min(vote_option) for proposal_id in ('199','210','188','133','200','245','72','170','54','232','258','118','4','205','264','143','68','230','77','55','159','261','63','201','208','128','256','38','255','93','16','194','117','169','157','58','224','51','129','124','31','79','60','34','139','39','268','106','103','76','266','171','52','189','239','227','180','22','138','57','24','251','221','184','131','42','43','6','130','149','203','145','7','164','225','56','148','46','178','107','75','211','94','155','135','204','61','219','161','36','113','112','243','53','95','195','127','44','163','108','32','154','1','246','240','235','136','84','50','234','137','30','223','233','162','260','80','252','216','78','186','187','183','116','114','197','257','206','91','190','249','150','220','9','81','185','132','12','48','64','70','218','179','101','26','82','174','156','146','242','228','259','147','122','153','83','142','140','59','5','96','15','151','120','33','265','62','175','144','125','110','45','165','28','222','192','88','236','40','172','231','66','166','196','237','18','226','213','217','89','98','176','71','209','97','241','244','202','173','123','111','74','269','262','214','47','21','3','250','49','191','85','41','99','207','248','69','229','152','134','119','23','141','238','13','126','115','267','270','212','121','182','37','109','102','29','2')) 
--order by voter

"""
TTL_MINUTES = 15

def create_query(SQL_QUERY): ## posts a flipside query to flipside sdk
    r = requests.post(
        'https://node-api.flipsidecrypto.com/queries', 
        data=json.dumps({
            "sql": SQL_QUERY,
            "ttlMinutes": TTL_MINUTES
        }),
        headers={"Accept": "application/json", "Content-Type": "application/json", "x-api-key": API_KEY},
    )
    if r.status_code != 200:
        raise Exception("Error creating query, got response: " + r.text + "with status code: " + str(r.status_code))
    
    return json.loads(r.text)
    
def get_query_results(token): ##retrieves query results from flipside sdk
    r = requests.get(
        'https://node-api.flipsidecrypto.com/queries/' + token, 
        headers={"Accept": "application/json", "Content-Type": "application/json", "x-api-key": API_KEY}
    )
    if r.status_code != 200:
        raise Exception("Error getting query results, got response: " + r.text + "with status code: " + str(r.status_code))
    
    data = json.loads(r.text)
    if data['status'] == 'running':
        time.sleep(10)
        return get_query_results(token)
        
    return data
def run(SQL_QUERY): ##generates a pivot query that aggregates all proposal id
    query = create_query(SQL_QUERY)
    token = query.get('token')
    data = get_query_results(token)
    return data

@st.experimental_memo
def quer(SQL_QUERY): ##generates the final dataframe of validators and voting results on each prop
    pivot = run(SQL_QUERY)
    query = create_query(PIVOT_QUERY+pivot['results'][0][0]) # concats the final query header and the pivot query generated
    token2 = query.get('token')
    data = get_query_results(token2)
    return data

def ext_key(doc): ## using KeyBERT model to retreive key phrases/words, input string, output list of tuples of the result
  kw_model = KeyBERT()
  keywords = kw_model.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer(stop_words='english'), use_mmr=True)
  return keywords  

def get_prop_desc(id): #retrieves proposal description useing proposal id through Osmosis RPC
    r = requests.get(
        'https://osmosis.stakesystems.io/cosmos/gov/v1beta1/proposals/' + str(id), 
        headers={"Accept": "application/json", "Content-Type": "application/json"}
    )
    if r.status_code != 200:
        raise Exception("Error getting query results, got response: " + r.text + "with status code: " + str(r.status_code))
    
    data = json.loads(r.text)

    return pd.Series([data['proposal']['content']['description'], data['proposal']['content']['@type']]) ## only selecting the proposal descripton and proposal type

def cluster_df(cluster_num): ##generate counts of each type of vote for each proposal for the selected cluster
  cluster_0 = df[df['cluster']==cluster_num]
  cluster_0_res = cluster_0.apply(lambda x: x.value_counts()).T.stack().drop(labels=['VOTER','LABEL','cluster']).to_frame().reset_index()
  return cluster_0_res

def get_keywords(cluster_df): ## get keywords from top 20 propsal that were voted yes, no, no with veto separately
  import re
  yeslist = cluster_df[cluster_df['level_1']==1].sort_values(0, ascending = False).head(n=20)['level_0'].to_list() #select top 20 proposals, based on frequency of yes vote
  yesprop = re.sub(r"\S*https?:\S*", "",''.join(str(x) for x in prop_desc[prop_desc['id'].isin(yeslist)]['desc'].to_list())).strip().replace('\\n',' ')#remove URL and concat prop description into list
  key_yes= ext_key(yesprop)

  nolist = cluster_df[cluster_df['level_1']==3].sort_values(0, ascending = False).head(n=20)['level_0'].to_list() #select top 20 proposals, based on frequency of no vote
  noprop = re.sub(r"\S*https?:\S*", "", ''.join(str(x) for x in prop_desc[prop_desc['id'].isin(nolist)]['desc'].to_list())).strip().replace('\\n',' ')
  key_no= ext_key(noprop)

  nowvetolist = cluster_df[cluster_df['level_1']==4].sort_values(0, ascending = False).head(n=20)['level_0'].to_list() #select top 20 proposals, based on frequency of no with veto vote
  nowvprop = re.sub(r"\S*https?:\S*", "", ''.join(str(x) for x in prop_desc[prop_desc['id'].isin(nowvetolist)]['desc'].to_list())).strip().replace('\\n',' ')
  key_nowv= ext_key(nowvprop)
  return key_yes, key_no, key_nowv

def discrete_colorscale(bvals, colors): #for the legend of validator activity heatmap
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale    

@st.experimental_memo
def prop_descr(): #generate df of proposal description and type 
	prop_desc = pd.DataFrame(columns=['id','type','desc'])
	prop_desc['id'] = df.columns.values[:-3].tolist()
	prop_desc[['desc','type']] = prop_desc.apply(lambda x: get_prop_desc(x['id']), axis=1)
	return prop_desc

@st.experimental_memo
def keyword_data(): #retrieve keywords for each cluster
	yes, no, no_with_veto = get_keywords(cluster_df(0))
	yes1, no1, no_with_veto1 = get_keywords(cluster_df(1))
	yes2, no2, no_with_veto2 = get_keywords(cluster_df(2))
	yes3, no3, no_with_veto3 = get_keywords(cluster_df(3))
	return yes, no, no_with_veto, yes1, no1, no_with_veto1, yes2, no2, no_with_veto2, yes3, no3, no_with_veto3

st.title('Attempt to classify Osmosis validators based on voting activity')
'''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/JustinTzeJi/osmo-prop) 

'''
st.markdown("<br>",unsafe_allow_html=True)
st.markdown('```If you get a 504 error, reload the page```')
st.markdown('```Querying data from Osmosis API and KeyBERT processing will take a moment```')
with st.spinner(text="Querying data from Flipside Shroom SDK"):
	res = quer(SQL_QUERY1)

test=[] 
test = sorted(list(map(int, res['columnLabels'][2::])))
test.extend(res['columnLabels'][:2:])
df = pd.DataFrame(data=res['results'],columns=test)
df = df.replace({nan: 0})
X = df.drop(columns = ['VOTER','LABEL'])
pca = PCA()
Xt = pca.fit_transform(X)
km = KMeans(init="k-means++", n_clusters=4, n_init=4)
y_km = km.fit_predict(Xt)
df['cluster']=pd.Series(y_km)

with st.spinner(text="Querying Osmosis API for prop info..... This might take a while...."):
	prop_desc = prop_descr()
with st.spinner(text="Extracting keywords & phrases with KeyBERT"):
	yes, no, no_with_veto, yes1, no1, no_with_veto1, yes2, no2, no_with_veto2, yes3, no3, no_with_veto3 = keyword_data()

with st.container():
	st.header('The Problem Statement')
	st.markdown('With the ever growing amount and ever changing list of validators on the Osmosis chain, how does one evaluate and select validator with similar *on-chain "politics"/viewpoints* as oneself. It is a laborious task to monitor and keep up with validator socials, and voting activities on each proposal.')

with st.container():
	st.header('The "Solution"?')
	st.markdown('Using the on-chain validator voting habits, we can use clustering algorithms to attempt to aggregate validators with similar votes. We can also extract prominent/recurring phrases that occurs in proposal descriptions to observe how said cluster of validators vote on a very surface level.')

with st.container():
	st.header('Limitations and Caveats')
	st.markdown("""
	1. Osmosis proposals are not usually polarized (for a reference of polarized proposals, see Cosmos Hub), hence *politics* might not be observable onchain?
	2. A large percentage of proposal description includes URL to the complete proposals are removed, due to difficulty of parsing each and every type of website, document.
	3. This dashboard does not account for the various validator stats, such as commission rate, voting power and missed blocks (see [Smart Stake](https://osmosis.smartstake.io/))
	4. As the size of validators have increased multiple times, clustering might reflect newer vs older validators
	5. It's just a fun project for me, don't take it too seriously :D 
	""")


with st.expander('Preview of data'):
	st.markdown("Data is provided by Flipside Crypto's [Shroom SDK](https://sdk.flipsidecrypto.xyz/shroomdk).")
	display_df = df.replace({1.0: 'YES', 2.0: 'ABSTAIN', 3.0: 'NO', 4.0:'NO WITH VETO', 0: 'DID NOT VOTE'})
	display_df = display_df.set_index(['LABEL','VOTER'])
	st.dataframe(display_df)


bvals = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
colors = ['#000000' , '#228b22', '#676767', '#ff7a00', '#ff0000' ]
dcolorsc = discrete_colorscale(bvals, colors)

with st.container():
	st.header("Voting activity of Osmosis Validators")
	bvals = np.array(bvals)
	tickvals = [0,1,2,3,4]
	ticktext = ['DID NOT VOTE', 'YES', 'ABSTAIN', 'NO', 'NO WITH VETO']
	heatmap = go.Heatmap(z=df.drop(columns = ['VOTER','cluster']).set_index('LABEL'), y=df['LABEL'].to_numpy(), 
						colorscale = dcolorsc, 
						customdata= display_df,
						hovertemplate="<b>%{y} </b><br>" +  "Prop %{x}<br>" + "Vote: %{customdata} ", 
						colorbar = dict(thickness=10, 
										tickvals=tickvals, 
										ticktext=ticktext))
	fig = go.Figure(data=[heatmap]).update_layout(title='Osmosis validator voting heatmap', yaxis_zeroline=False, xaxis_zeroline=False)
	st.plotly_chart(fig, use_container_width=True)

with st.container():
	st.header("Clustering of validators")
	st.markdown("""
	### Methodology:
	1. Voting data is dimensionaly reduced using *Principal COmoponent Analysis* (PCA).
	2. PCA-ed data is clustered using *K-means clustering* algorithm, using `number of target cluster` of 4
	""")
	options = st.multiselect(label = 'Select a validator/validators you would like to focus', options = df['LABEL'], default=[])

	pca_fig = go.Figure().update_xaxes(showgrid=False,showticklabels=False,zeroline=False).update_yaxes(showgrid=False,showticklabels=False,zeroline=False).update_layout(title='"Clustering" of validators based on voting activity')

	for x in range(4):
		selectedpoints = df[y_km==x].reset_index(drop=True).index[df['LABEL'][y_km==x].reset_index(drop=True).isin(options)].tolist()
		pca_fig.add_trace(go.Scatter(x=Xt[y_km==x, 0], y=Xt[y_km==x, 1], selectedpoints=selectedpoints, name='Cluster '+str(x), mode='markers', hovertemplate=df['LABEL'][y_km==x], selected_marker_color = 'yellow', selected_marker_size=10))
	st.plotly_chart(pca_fig, use_container_width=True)
	
	with st.expander("Validators in each cluster"):	
		cluster1, cluster2, cluster3, cluster4 = st.columns(4)
		cluster1.dataframe(df[df['cluster']==0].reset_index().rename(mapper={'LABEL':'Validators in Cluster 0'}, axis='columns')['Validators in Cluster 0'])
		cluster2.dataframe(df[df['cluster']==1].reset_index().rename(mapper={'LABEL':'Validators in Cluster 1'}, axis='columns')['Validators in Cluster 1'])
		cluster3.dataframe(df[df['cluster']==2].reset_index().rename(mapper={'LABEL':'Validators in Cluster 2'}, axis='columns')['Validators in Cluster 2'])
		cluster4.dataframe(df[df['cluster']==3].reset_index().rename(mapper={'LABEL':'Validators in Cluster 3'}, axis='columns')['Validators in Cluster 3'])

	st.markdown("""
	### Understanding the axes:
	Each axis consist of voting results from all proposals but condensed in a way that each of the proposals will have different weights depending on the variance of voting results of the proposal.
	By figuring out the variance of each proposal in each axis, we can conclude the proposals that has most influnce on the clustering of validators. 
	""")
	with st.expander("Top 10 Proposals for X-axis:"):
		xtop10 = pd.DataFrame(np.abs(pca.components_[0]))
		xtop10['prop'] = pd.DataFrame(df.columns[:-3])
		xtop10 = xtop10.nlargest(10,0,'all')
		s = ''
		t = 1
		for i in xtop10['prop'].to_list():
			s += str(t)+". Prop " + str(i) + " \n"
			t += 1
		st.markdown(s)
		col1, col2, col3, col4, col5 = st.columns(5)
		col6, col7, col8, col9, col10 = st.columns(5)
		dict1 = {1: col1, 2: col2, 3: col3, 4:col4,5:col5,6:col6,7:col7,8:col8,9:col9,10:col10}
		aaa = display_df.reset_index()
		aaa['cluster']=pd.Series(y_km)
		t=1
		for i in xtop10['prop'].to_list():
			tester = aaa.groupby('cluster')[i].value_counts()
			tester = tester.to_frame('counts').reset_index()
			fig = go.Figure()
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='YES']['cluster'].astype(str),
				x=tester[tester[i]=='YES']["counts"],
				name='YES',
				orientation='h',
				marker=dict(
					color='green',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='NO']['cluster'].astype(str),
				x=tester[tester[i]=='NO']["counts"],
				name='NO',
				orientation='h',
				marker=dict(
					color='orange',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='NO WITH VETO']['cluster'].astype(str),
				x=tester[tester[i]=='NO WITH VETO']["counts"],
				name='NO WITH VETO',
				orientation='h',
				marker=dict(
					color='red',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='ABSTAIN']['cluster'].astype(str),
				x=tester[tester[i]=='ABSTAIN']["counts"],
				name='ABSTAIN',
				orientation='h',
				marker=dict(
					color='lightgrey',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='DID NOT VOTE']['cluster'].astype(str),
				x=tester[tester[i]=='DID NOT VOTE']["counts"],
				name='DID NOT VOTE',
				orientation='h',
				marker=dict(
					color='black',
				)
			))		
			fig.update_layout(barmode='stack',title='Voting statistics of each cluster on Prop %d'%i,titlefont=dict(size=14),height = 300)
			dict1[t].plotly_chart(fig, use_container_width=True)
			t+=1
	with st.expander("Top 10 Proposals for Y-axis:"):
		ytop10 = pd.DataFrame(np.abs(pca.components_[1]))
		ytop10['prop'] = pd.DataFrame(df.columns[:-3])
		ytop10 = ytop10.nlargest(10,0,'all')
		s = ''
		t = 1
		for i in ytop10['prop'].to_list():
			s += str(t)+". Prop " + str(i) + " \n"
			t += 1
		st.markdown(s)
		col11, col12, col13, col14, col15 = st.columns(5)
		col16, col17, col18, col19, col20 = st.columns(5)
		dict1 = {11: col11, 12: col12, 13: col13, 14:col14,15:col15,16:col16,17:col17,18:col18,19:col19,20:col20}
		aaa = display_df.reset_index()
		aaa['cluster']=pd.Series(y_km)
		for i in xtop10['prop'].to_list():
			tester = aaa.groupby('cluster')[i].value_counts()
			tester = tester.to_frame('counts').reset_index()
			print(tester)
			fig = go.Figure()
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='YES']['cluster'].astype(str),
				x=tester[tester[i]=='YES']["counts"],
				name='YES',
				orientation='h',
				marker=dict(
					color='green',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='NO']['cluster'].astype(str),
				x=tester[tester[i]=='NO']["counts"],
				name='NO',
				orientation='h',
				marker=dict(
					color='orange',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='NO WITH VETO']['cluster'].astype(str),
				x=tester[tester[i]=='NO WITH VETO']["counts"],
				name='NO WITH VETO',
				orientation='h',
				marker=dict(
					color='red',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='ABSTAIN']['cluster'].astype(str),
				x=tester[tester[i]=='ABSTAIN']["counts"],
				name='ABSTAIN',
				orientation='h',
				marker=dict(
					color='lightgrey',
				)
			))
			fig.add_trace(go.Bar(
				y=tester[tester[i]=='DID NOT VOTE']['cluster'].astype(str),
				x=tester[tester[i]=='DID NOT VOTE']["counts"],
				name='DID NOT VOTE',
				orientation='h',
				marker=dict(
					color='black',
				)
			))		
			fig.update_layout(barmode='stack',title='Voting statistics of each cluster on Prop %d'%i,titlefont=dict(size=14), height = 300)
			dict1[t].plotly_chart(fig, use_container_width=True,)
			t+=1

with st.container():
	st.header("Keywords from each Clusters")
	st.markdown("""
	### Methodology:
	1. Top 20 Proposal (with highest number of same votes) of each type of vote (`YES`, `NO`, `NO WITH VETO`) are selected
	2. The proposal descriptions are retreived from Osmosis LCD, URLs in the description are removed.
	3. Proposals description from each category and cluster is concatenated and keywords/phrases are extracted using `KeyBERT` algorithm, with `KeyphraseVectorizers()` and  `Maximal Margin Relevance`
	""")
	with st.expander("Cluster 0"):
		y_df = pd.DataFrame(yes, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposals")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposals")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposals")
		st.plotly_chart(nw_fig, use_container_width=True)

	with st.expander("Cluster 1"):
		y_df = pd.DataFrame(yes1, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposals")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no1, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposals")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto1, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposals")
		st.plotly_chart(nw_fig, use_container_width=True)

	with st.expander("Cluster 2"):
		y_df = pd.DataFrame(yes2, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposals")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no2, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposals")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto2, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposals")
		st.plotly_chart(nw_fig, use_container_width=True)

	with st.expander("Cluster 3"):
		y_df = pd.DataFrame(yes3, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposals")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no3, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposals")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto3, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposals")
		st.plotly_chart(nw_fig, use_container_width=True)