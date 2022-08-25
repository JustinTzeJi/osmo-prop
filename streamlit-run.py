import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from numpy import nan
from sklearn.cluster import KMeans
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from shroomdk import ShroomDK
import re
import collections

st.set_page_config(page_title="Attempt to classify Osmosis validators based on voting activity", page_icon="⚗️", layout="wide", menu_items={'Report a Bug': "https://github.com/JustinTzeJi/osmo-prop/issues",'About': "An experiment"})

class fsdata:
	def __init__(self):
		self.sdk = ShroomDK(st.secrets["api_keys"])
		self.act_val_quer = """SELECT address as node, raw_metadata[0]:account_address as address, label
FROM osmosis.core.dim_labels
WHERE label_subtype = 'validator'
AND raw_metadata[0]:status = 3"""
		self.prop_votes_quer = """with a as (SELECT address as node, raw_metadata[0]:account_address as address, label
FROM osmosis.core.dim_labels
WHERE label_subtype = 'validator'
AND raw_metadata[0]:status = 3),

max_time as (SELECT MAX(BLOCK_TIMESTAMP) as BLOCK_TIMESTAMP, a.address, node, a.label, proposal_id
FROM osmosis.core.fact_governance_votes b inner join a on b.voter = a.address
WHERE tx_status = 'SUCCEEDED'
Group by proposal_id, address, node, a.label)

SELECT c.voter as address, max_time.node, max_time.label, c.proposal_id, c.vote_option
FROM osmosis.core.fact_governance_votes c inner join max_time on 
  c.BLOCK_TIMESTAMP = max_time.BLOCK_TIMESTAMP AND 
  c.voter = max_time.address AND 
  c.proposal_id = max_time.proposal_id
WHERE tx_status = 'SUCCEEDED'"""
		self.daily_stake_quer = """
	-- superfluid staking amount are excluded
with 
  validators as (
    select address, label_subtype as role, label
    from osmosis.core.dim_labels
    where label_subtype = 'validator'
	and raw_metadata[0]:status = 3
  ),
  create_validator as (
    select block_timestamp, tx_id, action, iff(regexp_like(amount, '.+uosmo$') = 'TRUE', split_part(amount, 'uosmo', 1), amount) as amount,
      validator_address
    from (
        select distinct
          block_timestamp, 
          tx_id,
          'delegate' as action,
          max(iff(msg_type = 'create_validator' and attribute_key = 'amount', attribute_value, null)) as amount,
          max(iff(msg_type = 'create_validator' and attribute_key = 'validator', attribute_value, null)) as validator_address
          -- max(iff(msg_type = 'tx', split_part(attribute_value, '/', 1), null)) as creator
        from osmosis.core.fact_msg_attributes 
        where 0=0
          and ((msg_type = 'create_validator' and attribute_key in ('amount', 'validator')) or (msg_type = 'tx' and attribute_key = 'acc_seq')) 
          and tx_id in (select distinct tx_id from osmosis.core.fact_msg_attributes where msg_type = 'create_validator')
        group by 1,2
      )  order by block_timestamp asc, tx_id
  ),
  create_n_staking as (
  	-- self-bonded
    select 
      cv.block_timestamp,
      cv.tx_id,
      'delegate' as action,
      cv.amount,
      cv.validator_address,
      vld.label as validator_name
    from create_validator cv 
      left join validators vld on (vld.address = cv.validator_address)
    union all 
  	-- normal staking
    select 
      fs.block_timestamp,
      fs.tx_id,
      fs.action,
      case
        when fs.action in ('delegate', 'redelegate') and fs.validator_address = vld.address then amount
        when fs.action = 'redelegate' and fs.redelegate_source_validator_address = vld.address then amount*-1
        when fs.action = 'undelegate' and fs.validator_address = vld.address then amount*-1
      end as amount,
      vld.address as validator_address,
      vld.label as validator_name
    from osmosis.core.fact_staking fs
      left join validators vld on (vld.address = fs.validator_address or vld.address = fs.redelegate_source_validator_address)
  ),
  normal_staking as (
    select date(block_timestamp) as date, first_tx_timestamp, last_tx_timestamp, validator_address, validator_name, sum(amount) as real_amount
    from (
      select 
        *,
        min(block_timestamp) over (partition by validator_address) as first_tx_timestamp,
        max(block_timestamp) over (partition by validator_address) as last_tx_timestamp
      from create_n_staking
    ) group by 1,2,3,4,5
  ),
-- Generate every day + validator info table
  partition_ranges as (
    select 
      validator_address,
      validator_name,
      first_tx_timestamp,
      last_tx_timestamp,
      datediff('days', first_tx_timestamp, current_date) as span
    from normal_staking
  ), 
  huge_range as (
      select row_number() over(order by true)-1 as rn
      from table(generator(rowcount => 10000000))
  ), 
  in_fill as (
      select distinct 
        dateadd('day', hr.rn, pr.first_tx_timestamp) as date,
        pr.validator_address,
        pr.validator_name
      from partition_ranges as pr
      join huge_range as hr on pr.span >= hr.rn
  ),
--------------
  daily_staking_balance as (
    select 
      date_trunc('DAY', date) as date,
      nvl(last_tx_timestamp, lag(last_tx_timestamp) ignore nulls over (partition by validator_address order by date asc)) as last_tx_timestamp,
      validator_address,
      validator_name,
      amount/pow(10,6) as amount,
      sum(amount/pow(10,6)) over (partition by validator_name order by date asc) as daily_delegated_amount -- cumulative daily delegated amount
    from (
        select a.date, ns.last_tx_timestamp, a.validator_address, a.validator_name, nvl(ns.real_amount,0) as amount
        from in_fill a
          left join normal_staking ns on (date(a.date) = ns.date and a.validator_address = ns.validator_address)
        )
  ),
  total_staked as (
  	SELECT date, sum(daily_delegated_amount) as total_daily_staked
	from daily_staking_balance 
  	group by 1
  )

select ds.date, validator_name, validator_address, daily_delegated_amount, ts.total_daily_staked, daily_delegated_amount/ts.total_daily_staked*100 as voting_power
FROM daily_staking_balance ds LEFT JOIN total_staked ts
  ON ds.date = ts.date
order by date desc

"""
		self.query_prop()
		self.query_val()
		self.mergedata()
		self.daily_stake_f()
	
	def query_prop(self):
		data = self.sdk.query(self.prop_votes_quer)
		df = pd.DataFrame(data=data.rows,columns=data.columns).pivot_table(index = ['ADDRESS', 'LABEL', 'NODE'], columns='PROPOSAL_ID',values='VOTE_OPTION')
		df.columns = df.columns.astype(int)
		df = df.sort_index(axis=1).reset_index()
		self.prop_votes = df
		return self.prop_votes

	def query_val(self):
		validators = self.sdk.query(self.act_val_quer)
		self.act_val = pd.DataFrame(data=validators.rows,columns=validators.columns)
		return self.act_val

	def mergedata(self):
		self.act_val_prop_votes = pd.merge(self.act_val, self.prop_votes, how="left", on=['ADDRESS', 'LABEL', 'NODE']).replace({nan: 0})
		return self.act_val_prop_votes

	def daily_stake_f(self): ##generates the final dataframe of validators and voting results on each prop
		query_stake = self.sdk.query(self.daily_stake_quer)
		self.daily_stake = pd.DataFrame(data=query_stake.rows,columns=query_stake.columns)
		return self.daily_stake

class analytics:
	def __init__(self,fsdata):
		self.val_data = fsdata.act_val_prop_votes
		self.pca_fn()
		self.cluster_fn()
		self.mergeclusterdata()
	
	def pca_fn(self):
		X = self.val_data.drop(columns = ['NODE','ADDRESS', 'LABEL'])
		pca = PCA(n_components=3)
		self.Xt = pca.fit_transform(X)
		self.pca_data = pd.DataFrame(self.Xt).join(self.val_data.LABEL)
		self.pca_components = pca.components_
		return self.Xt, self.pca_data, self.pca_components
		
	def cluster_fn(self):
		km = KMeans(init="k-means++", n_clusters=4, n_init=4)
		self.cluster = km.fit_predict(self.Xt)
		return self.cluster
	
	def mergeclusterdata(self):
		self.main_data = self.val_data
		self.main_data['cluster']=pd.Series(self.cluster)
		return self.main_data

class keybert:
	def __init__(self,analytics):
		self.cluster_data = analytics.main_data
		self.get_prop_desc()
		self.cluster_dict()
		self.keybert_dict()

	def get_prop_desc(self):
		r = requests.get('https://osmosis.stakesystems.io/cosmos/gov/v1beta1/proposals?pagination.limit=1000000000', 
		headers={"Accept": "application/json", "Content-Type": "application/json"})
		if r.status_code != 200:
			raise Exception("Error getting query results, got response: " + r.text + "with status code: " + str(r.status_code))
    
		data_p = json.loads(r.text)
		df3 = pd.DataFrame(columns=['prop_id','title','description'])

		for props in data_p['proposals']:
			data=[{'prop_id':int(props['proposal_id']),'title':str(props['content']['title']),'description':str(props['content']['description'])}]
			df3 = pd.concat([df3, pd.DataFrame.from_records(data)])
		self.props = df3.reset_index(drop=True)
		return self.props
	
	def cluster_dict(self):
		self.cluster_count={}
		self.unique_clusters = sorted(self.cluster_data.cluster.unique())
		for cluster in self.unique_clusters:
			df = self.cluster_data[self.cluster_data['cluster']==cluster]
			countdata = df.apply(lambda x: x.value_counts()).T.stack().drop(labels=['NODE','ADDRESS','LABEL','cluster']).to_frame().reset_index().rename(columns={'level_0': 'proposal_id', 'level_1': 'vote',0:'count'})
			self.cluster_count[cluster] = countdata
		return self.cluster_count, self.unique_clusters
			
	def extract(self,doc):
		model = KeyBERT()
		keywords = model.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer(stop_words='english'), use_mmr=True)
		return keywords

	def keybert_dict(self):
		vote_option = [1.0,3.0,4.0]
		self.clusterkeyword=collections.defaultdict(dict)
		for cluster in self.cluster_count:
			for option in vote_option:
				df=self.cluster_count[cluster]
				top20prop = df[df['vote']==option].sort_values(by=['count'], ascending = False).head(n=20).proposal_id.to_list()
				top20prop_words = re.sub(r"\S*https?:\S*", "",''.join(str(x) for x in self.props[self.props.prop_id.isin(top20prop)]['description'].to_list())).strip().replace('\\n',' ')
				keywords = self.extract(top20prop_words)
				self.clusterkeyword[cluster][option]=keywords
		return self.clusterkeyword

class displays():
	def __init__(self,fsdata,analytics,keybert):
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
		df = self.analytics.val_data.replace({1.0: 'YES', 2.0: 'ABSTAIN', 3.0: 'NO', 4.0:'NO WITH VETO', 0: 'DID NOT VOTE'})
		df = df.drop(columns = ['NODE', 'cluster'])
		self.display_df_bar = df
		df = df.set_index(['LABEL','ADDRESS'])
		self.display_df = df

	def df_heatmap_(self):
		#---generate color and colorbar for heatmap
		bvals = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
		colors = ['#000000' , '#228b22', '#676767', '#ff7a00', '#ff0000' ]
		ticktext = ['DID NOT VOTE', 'YES', 'ABSTAIN', 'NO', 'NO WITH VETO']
		tickvals = [0,1,2,3,4]
		nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]
		dcolorscale = []
		for k in range(len(colors)):
			dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])

		#----generate heatmap
		heatmap = go.Heatmap(z=self.analytics.main_data.drop(columns = ['ADDRESS','cluster','NODE']).set_index('LABEL'), y=self.analytics.main_data['LABEL'].to_numpy(), 
							colorscale = dcolorscale, 
							customdata= self.display_df_bar.set_index(['LABEL','ADDRESS']),
							hovertemplate="<b>%{y} </b><br>" +  "Prop %{x}<br>" + "Vote: %{customdata} ", 
							colorbar = dict(thickness=10, 
											tickvals=tickvals, 
											ticktext=ticktext))
		fig = go.Figure(data=[heatmap]).update_layout(title='Osmosis validator voting heatmap', yaxis_zeroline=False, xaxis_zeroline=False)
		self.df_heatmap = fig

	def df_pca_(self):
		fig = px.scatter_3d(
			self.analytics.pca_data, x=0, y=1, z=2,
			color=list(map(str,self.analytics.main_data.cluster)), hover_name='LABEL',
			labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}, title='Visualization of predicted groups of formulations in data'
		)
		fig.update_traces(marker_size = 7)
		self.df_pca = fig

	def pca_bar_(self):
		self.axis_top_dict=collections.defaultdict(dict)
		self.pca_bar=collections.defaultdict(dict)
		# self.pca_bar_container = st.container()
		for x, axis in enumerate(['X','Y'],0):
			top10 = pd.DataFrame(np.abs(self.analytics.pca_components[x]))
			top10['prop'] = pd.DataFrame(self.analytics.val_data.columns[:-3])
			top10 = top10.nlargest(10,0,'all')
			toplist = ''
			for t, i in enumerate(top10['prop'].to_list(), 1):
				toplist += str(t)+". Prop " + str(i) + " \n"
			self.axis_top_dict[axis][toplist] = toplist
			self.display_df_bar['cluster'] = self.analytics.cluster
			
			self.bar_columns_1 = []
			options = {'YES':'green','NO':'orange','NO WITH VETO':'red','ABSTAIN':'lightgrey','DID NOT VOTE':'black'}
			for prop in top10['prop'].to_list():
				grouping_df = self.display_df_bar.groupby('cluster')[prop].value_counts()
				grouping_df = grouping_df.to_frame('counts').reset_index()
				fig = go.Figure()
				for option in options:
					fig.add_trace(go.Bar(
						y=grouping_df[grouping_df[prop]==option]['cluster'].astype(str),
						x=grouping_df[grouping_df[prop]==option]["counts"],
						name=option,
						orientation='h',
						marker=dict(
							color=options[option],
						)
					))
				fig.update_layout(barmode='stack',title='Voting statistics of each cluster on Prop %d'%prop,titlefont=dict(size=14),height = 300)
				self.bar_columns_1.append(fig)
			self.bar_columns_2 = []
			for prop in top10['prop'].to_list()[5:]:
				grouping_df = self.display_df_bar.groupby('cluster')[prop].value_counts()
				grouping_df = grouping_df.to_frame('counts').reset_index()
				fig = go.Figure()
				for option in options:
					fig.add_trace(go.Bar(
						y=grouping_df[grouping_df[prop]==option]['cluster'].astype(str),
						x=grouping_df[grouping_df[prop]==option]["counts"],
						name=option,
						orientation='h',
						marker=dict(
							color=options[option],
						)
					))
				fig.update_layout(barmode='stack',title='Voting statistics of each cluster on Prop %d'%prop,titlefont=dict(size=14),height = 300)
				self.bar_columns_2.append(fig)
			self.pca_bar[axis][1]=self.bar_columns_1
			self.pca_bar[axis][2]=self.bar_columns_2

	def voting_power_(self):
		cluster_power = self.fsdata.daily_stake.merge(self.analytics.val_data[['NODE','cluster']], left_on='VALIDATOR_ADDRESS', right_on='NODE')
		cluster_power = cluster_power [['DATE', 'VOTING_POWER', 'cluster']]
		cluster_power = cluster_power.groupby(['cluster', 'DATE']).sum().reset_index()
		voting_power = px.line(cluster_power, x="DATE", y="VOTING_POWER", color="cluster" , title="Estimated* voting power of clusters").update_yaxes(showgrid=False)
		voting_power.update_layout(xaxis_title=None, yaxis_title='Voting Power (%)')
		self.voting_power = voting_power
	
	def display_keybert(self):
		self.keybert_chart_=collections.defaultdict(dict)
		unique_clusters = sorted(self.analytics.main_data.cluster.unique())
		list = {1:['yes','green'],3:['no','orange'],4:['no-with veto','red']}
		for cluster in unique_clusters:
			keybert_chart = []
			cluster_num = 'Cluster ' + str(cluster)
			for option in self.keybert.clusterkeyword[cluster]:
				df = pd.DataFrame(self.keybert.clusterkeyword[cluster][option], columns=['phrase', 'relevance'])
				fig = px.bar(df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from %s-voted proposals"%list[option][0]).update_traces(marker_color=list[option][1])
				keybert_chart.append(fig)
				self.keybert_chart_[cluster_num]=keybert_chart

@st.cache(suppress_st_warning=True,show_spinner=False,allow_output_mutation=True)
def main():
	st.markdown('```Querying data from Osmosis API and KeyBERT processing will take a moment```')
	
	with st.spinner(text="Querying data from Flipside Shroom SDK"):
		data=fsdata()
	with st.spinner(text="Analyzing Data"):
		analysis=analytics(data)
	with st.spinner(text="Querying Osmosis API for prop info, Extracting keywords & phrases with KeyBERT"):
		keybert_analysis=keybert(analysis)
	with st.spinner(text="Generating visuals"):
		display_visuals=displays(data,analysis,keybert_analysis)
	return data,analysis,keybert_analysis,display_visuals

if __name__ == "__main__":
	st.title('Attempt to classify Osmosis validators based on voting activity')
	'''
		[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/JustinTzeJi/osmo-prop) 

	'''
	st.markdown("<br>",unsafe_allow_html=True)
	data,analysis,keybert_analysis,display_visuals = main()
	with st.container():
		st.header('The Problem Statement')
		st.markdown('With the ever growing amount and ever changing list of validators on the Osmosis chain, how does one evaluate and select validator with similar *on-chain "politics"/viewpoints* as oneself. It is a laborious task to monitor and keep up with validator socials, and voting activities on each proposal.')
		st.header('The "Solution"?')
		st.markdown('Using the on-chain validator voting habits, we can use clustering algorithms to attempt to aggregate validators with similar votes. We can also extract prominent/recurring phrases that occurs in proposal descriptions to observe how said cluster of validators vote on a very surface level.')
		st.header('Limitations and Caveats')
		st.markdown("""
		1. Osmosis proposals are not usually polarized (for a reference of polarized proposals, see Cosmos Hub), hence *politics* might not be observable onchain?
		2. A large percentage of proposal description includes URL to the complete proposals are removed, due to difficulty of parsing each and every type of website, document.
		3. This dashboard does not account for the various validator stats, such as commission rate, voting power and missed blocks (see [Smart Stake](https://osmosis.smartstake.io/))
		4. As the size of validators have increased multiple times, clustering might reflect newer vs older validators
		5. It's just a fun project for me, don't take it too seriously :D 
		""")

	with st.expander('Preview of data'):
		st.dataframe(display_visuals.display_df)

	with st.container():
		st.header("Voting activity of Osmosis Validators")
		st.plotly_chart(display_visuals.df_heatmap, use_container_width=True)

	with st.container():
		st.header("Clustering of validators")
		st.markdown("""
		### Methodology:
		1. Voting data is dimensionaly reduced using *Principal COmoponent Analysis* (PCA).
		2. PCA-ed data is clustered using *K-means clustering* algorithm, using `number of target cluster` of 4
		""")	
	st.plotly_chart(display_visuals.df_pca, use_container_width=True)	
	with st.expander("Validators in each cluster"):
		a_data = analysis.main_data
		for column, cluster in zip(st.columns(len(keybert_analysis.unique_clusters)), keybert_analysis.unique_clusters):
			column.dataframe(a_data[a_data['cluster']==cluster].reset_index().rename(mapper={'LABEL':'Validators in Cluster %s'%cluster}, axis='columns')['Validators in Cluster %s'%cluster])
	
	st.markdown("""
	### Understanding the axes:
	Each axis consist of voting results from all proposals but condensed in a way that each of the proposals will have different weights depending on the variance of voting results of the proposal.
	By figuring out the variance of each proposal in each axis, we can conclude the proposals that has most influnce on the clustering of validators. 
	""")
	for axis in display_visuals.pca_bar:
		with st.expander("Top 10 Proposals for %s-axis:"%axis):
			for i, column in enumerate(st.columns(5)):
				column.plotly_chart(display_visuals.pca_bar[axis][1][i], use_container_width=True) 
			for i, column in enumerate(st.columns(5)):
				column.plotly_chart(display_visuals.pca_bar[axis][2][i], use_container_width=True) 

with st.container():
	st.header("Voting power of each cluster")
	st.markdown("""
	`The voting power is estimated as tokens staked through superfluid is not included in this measurement.`
	Credit to [@whaen](https://github.com/whaen) for the query!
	""")
	st.plotly_chart(display_visuals.voting_power, use_container_width=True)

with st.container():
	st.header("Keywords from each Clusters")
	st.markdown("""
	### Methodology:
	1. Top 20 Proposal (with highest number of same votes) of each type of vote (`YES`, `NO`, `NO WITH VETO`) are selected
	2. The proposal descriptions are retreived from Osmosis LCD, URLs in the description are removed.
	3. Proposals description from each category and cluster is concatenated and keywords/phrases are extracted using `KeyBERT` algorithm, with `KeyphraseVectorizers()` and  `Maximal Margin Relevance`
	""")
	for cluster in display_visuals.keybert_chart_:
		with st.expander(cluster):
			for chart in display_visuals.keybert_chart_[cluster]:
				st.plotly_chart(chart, use_container_width=True)