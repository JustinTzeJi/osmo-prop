import streamlit as st
import requests
import json
import time
import pandas as pd
from numpy import nan
from sklearn.cluster import KMeans
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.decomposition import PCA
import plotly.express as px

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

def create_query(SQL_QUERY):
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
    
def get_query_results(token):
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
def run(SQL_QUERY):
    query = create_query(SQL_QUERY)
    token = query.get('token')
    data = get_query_results(token)
    return data

def quer(SQL_QUERY):
    pivot = run(SQL_QUERY)
    query = create_query(PIVOT_QUERY+pivot['results'][0][0])
    token2 = query.get('token')
    data = get_query_results(token2)
    return data

def ext_key(doc):
  kw_model = KeyBERT()
  keywords = kw_model.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer(stop_words='english'), use_mmr=True)
  return keywords  
# keywords = kw_model.extract_keywords(doc)

def get_prop_desc(id):
    r = requests.get(
        'https://osmosis.stakesystems.io/cosmos/gov/v1beta1/proposals/' + str(id), 
        headers={"Accept": "application/json", "Content-Type": "application/json"}
    )
    if r.status_code != 200:
        raise Exception("Error getting query results, got response: " + r.text + "with status code: " + str(r.status_code))
    
    data = json.loads(r.text)

    return pd.Series([data['proposal']['content']['description'], data['proposal']['content']['@type']])

def cluster_df(cluster_num):
  cluster_0 = df[df['cluster']==cluster_num]
  cluster_0_res = cluster_0.apply(lambda x: x.value_counts()).T.stack().drop(labels=['VOTER','LABEL','cluster']).to_frame().reset_index()
  return cluster_0_res

def get_keywords(cluster_df):
  import re
  yeslist = cluster_df[cluster_df['level_1']==20].sort_values(0, ascending = False).head(n=20)['level_0'].to_list()
  yesprop = re.sub(r"\S*https?:\S*", "",''.join(str(x) for x in prop_desc[prop_desc['id'].isin(yeslist)]['desc'].to_list())).strip().replace('\\n',' ')
  key_yes= ext_key(yesprop)

  nolist = cluster_df[cluster_df['level_1']==-15].sort_values(0, ascending = False).head(n=20)['level_0'].to_list()
  noprop = re.sub(r"\S*https?:\S*", "", ''.join(str(x) for x in prop_desc[prop_desc['id'].isin(nolist)]['desc'].to_list())).strip().replace('\\n',' ')
  key_no= ext_key(noprop)

  nowvetolist = cluster_df[cluster_df['level_1']==-20].sort_values(0, ascending = False).head(n=20)['level_0'].to_list()
  nowvprop = re.sub(r"\S*https?:\S*", "", ''.join(str(x) for x in prop_desc[prop_desc['id'].isin(nowvetolist)]['desc'].to_list())).strip().replace('\\n',' ')
  key_nowv= ext_key(nowvprop)
  return key_yes, key_no, key_nowv

st.title('Attempt to classify Osmosis validators based on voting activity')

with st.spinner(text="Querying data from Flipside SDK"):
	res = quer(SQL_QUERY1)

test=[] 
test = sorted(list(map(int, res['columnLabels'][2::])))
test.extend(res['columnLabels'][:2:])
df = pd.DataFrame(data=res['results'],columns=test)
df = df.replace({2.0: 0, nan: -1, 1.0: 20, 3.0: -15, 4.0:-20})

with st.container():
	st.header("Preview of data")
	display_df = df.replace({20: 'YES', 0.0: 'ABSTAIN', -15: 'NO', -20:'NO WITH VETO', -1: 'DID NOT VOTE'})
	display_df = display_df.set_index(['LABEL','VOTER'])
	st.dataframe(display_df)

X = df.drop(columns = ['VOTER','LABEL'])
pca = PCA()
Xt = pca.fit_transform(X)
km = KMeans(init="k-means++", n_clusters=4, n_init=4)
y_km = km.fit_predict(Xt)
df['cluster']=pd.Series(y_km)
ykm2 = [str(numeric_string) for numeric_string in y_km]

with st.container():
	st.header("Voting activity of Osmosis Validators")
	fig = px.imshow(df.drop(columns = ['VOTER','cluster']).set_index('LABEL'))
	st.plotly_chart(fig, use_container_width=True)

with st.container():
	st.header("K-means grouping of validators")
	scatter = px.scatter(Xt, x= Xt[:,0], y=Xt[:,1], color=ykm2, symbol=ykm2)
	st.plotly_chart(scatter, use_container_width=True)

prop_desc = pd.DataFrame(columns=['id','type','desc'])
prop_desc['id'] = df.columns.values[:-3].tolist()
with st.spinner(text="Querying Osmosis API for prop info"):
	prop_desc[['desc','type']] = prop_desc.apply(lambda x: get_prop_desc(x['id']), axis=1)

with st.container():
	st.subheader("Keywords from Clusters")
	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Cluster 0")
		with st.spinner(text="Extracting keywords & phrases with KeyBERT"):
			yes, no, no_with_veto = get_keywords(cluster_df(0))
		y_df = pd.DataFrame(yes, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposal")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposal")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposal")
		st.plotly_chart(nw_fig, use_container_width=True)

	with col2:
		st.subheader("Cluster 1")
		with st.spinner(text="Extracting keywords & phrases with KeyBERT"):
			yes, no, no_with_veto = get_keywords(cluster_df(1))
		y_df = pd.DataFrame(yes, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposal")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposal")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposal")
		st.plotly_chart(nw_fig, use_container_width=True)

	col3, col4 = st.columns(2)
	with col3:
		st.subheader("Cluster 2")
		with st.spinner(text="Extracting keywords & phrases with KeyBERT"):
			yes, no, no_with_veto = get_keywords(cluster_df(2))
		y_df = pd.DataFrame(yes, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposal")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposal")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposal")
		st.plotly_chart(nw_fig, use_container_width=True)

	with col4:
		st.subheader("Cluster 3")
		with st.spinner(text="Extracting keywords & phrases with KeyBERT"):
			yes, no, no_with_veto = get_keywords(cluster_df(3))
		y_df = pd.DataFrame(yes, columns=['phrase', 'relevance'])
		y_fig = px.bar(y_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from yes-voted proposal")
		st.plotly_chart(y_fig, use_container_width=True)
		n_df = pd.DataFrame(no, columns=['phrase', 'relevance'])
		n_fig = px.bar(n_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-voted proposal")
		st.plotly_chart(n_fig, use_container_width=True)
		nw_df = pd.DataFrame(no_with_veto, columns=['phrase', 'relevance'])
		nw_fig = px.bar(nw_df, x="relevance", y="phrase", orientation='h', title="Keywords/phrases from no-with veto-voted proposal")
		st.plotly_chart(nw_fig, use_container_width=True)