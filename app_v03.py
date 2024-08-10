import streamlit as st
import pickle
import pandas as pd
import numpy as np
import configparser
#import os
#import pyarrow.dataset as ds
import gravis as gv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit.components.v1 as components
import pyarrow.dataset as ds
import shap
from pathlib import Path

def filter_dataset(dataset, min_d, max_d = None , cols=None):
    if max_d == None:
        return dataset.to_table(filter=(ds.field("d") == min_d), columns=cols).to_pandas()
    else:
        return dataset.to_table(filter=((ds.field("d") >= min_d) & (ds.field("d") <= max_d)), columns=cols).to_pandas()

def create_label(i_node):
    label_tmp = df_day.loc[i_node, groupby_levels]
    label = str()
    for i_level in groupby_levels: 
        if pd.isna(label_tmp[i_level]):
            continue
        else: 
            if label == '':
                label = i_level + '=' + str(label_tmp[i_level])
            else:
                label =label + ' & '+ i_level + '=' + str(label_tmp[i_level])
    if label == '':
        label = 'All'
    return label

def create_filter_df_shap(filter_key):
    level_list = filter_key.index    
    flt_str_and = str()
    initial = ''
    for i_level in level_list:
        if flt_str_and != '':
            initial = ' & '
        val = filter_key.loc[i_level]
        if pd.isnull(val):
            val = i_level+'.isnull()'
        else:
            val = i_level + '==' + "\'" + filter_key[i_level] + "\'" 
        flt_str_and = flt_str_and + initial + val
    flt_str = flt_str_and
    return(flt_str)

def aggregate_shap(start_node, groupby_levels, shap_df, reported_nodes, features):
    #This finction calculates the sum of shap values for the start_node from shap_df
    #input: start_node - Index to be analyzed 
    #       shap_df - SHAP values for all unaggregated nodes
    #       reported_nodes - DataFrame with outliers of the given day. (The key of the start node is extracted from reported_nodes)
    #       groupby_levels - List of columns that define the key of the start_node
    #output: sum_shap_df_filtered_np - SHAP values aggregated at the group by level of the given start_node 
    #Requires create_filter_df_shap function
    filter_key = reported_nodes.loc[start_node, groupby_levels]
    filter_key = filter_key.dropna()
    filter = create_filter_df_shap(filter_key)
    if filter == '': #This is the case when the start_node is the top node
        shap_df_filtered = shap_df 
    else: #all other cases
        #filter shap_df to get the shap values of the start_node
        shap_df_filtered = shap_df.query(filter)
    #rowsums shap
    sum_shap_df_filtered = shap_df_filtered.loc[:,features].sum(axis=0)
    #convert to numpy
    sum_shap_df_filtered_np = sum_shap_df_filtered.to_numpy()
    return sum_shap_df_filtered_np

def display_description_maxshap(max_display, feature_description, sum_shap_df_filtered_np, features):
    #Find index of max_display highest absolue values in sum_shap_df_filtered_np
    top_idx = np.abs(sum_shap_df_filtered_np).argsort()[-max_display:]#[::-1]
    #Generate list of feature names of 7 highest absolue values in sum_shap_df_filtered_np
    top_features = [features[i] for i in top_idx]

    #Display feature description of top 7 features
    print(feature_description[feature_description['Feature name'].isin(top_features)])

def list_files(directory):
    p = Path(directory)
    files = [file.name for file in p.glob('*') if file.is_file()]
    return files


#tbd: move and read groupby levels from a separate file to spare some space from the main file df_day 

########SELECT DAY########
#d_day = 1540#1556
#########################


##Config
config_file_path = 'config_app.conf'
root_prefix = ''
#config_file_path = '/content/gdrive/MyDrive/signalligence/config.conf'
#root_prefix = '/content/gdrive/MyDrive/signalligence/'
config_read = configparser.ConfigParser()
config_read.read(config_file_path)
#preprocessing_package_file = root_prefix+config_read.get("Paths", "preprocessing_package_file")
training_outlier_data_file = root_prefix+config_read.get("Paths", "training_outlier_data_file")
dim_level_structure_pth = root_prefix+config_read.get("Paths", "dim_level_structure_pth")
df_day_pth = root_prefix+config_read.get("Paths", "df_day_pth")
edges_pth = root_prefix+config_read.get("Paths", "edges_pth")
shap_pth = root_prefix+config_read.get("Paths", "shap_pth")
explainer_file = root_prefix+config_read.get("Paths", "explainer_file")
feature_description_pth = root_prefix+config_read.get("Paths", "feature_description_pth")


_, groupby_levels, _ = pd.read_pickle(dim_level_structure_pth)#.replace('/','\\'))

measure = config_read.get("MainSettings", "measure") #the response
measure1 = config_read.get("MainSettings", "measure1") #the response
preds = config_read.get("MainSettings", "preds") #the response
preds1 = config_read.get("MainSettings", "preds1") #the response

#list of files in dim_level_structure_pth
#dim_files = os.listdir(dayly_data_pth) #os is not allowed in streamlit cloud
dim_files = list_files(shap_pth)
#extract numbers from the filenames after 'before '_'
days_in_dir = [int(i.split('_')[0]) for i in dim_files]
#days_in_dir =list(range(1547,1561)) 

#d_day = 1556

st.title('Signalligence')

st.header('Choose a day to analyze')
# Add a selection box to the application to select the 'id' with defoult value 1556
st.write(f'You can select a DAY. For demonstration purposis, some extreme artificial contamination was placed in the data. Contaminated days are: 1549, 1556, 1616, 1706')
d_day = st.selectbox('Select day to analyze', days_in_dir)

reported_idx = []

#from parquet
dataset = ds.dataset(training_outlier_data_file.replace('/','\\'), 
                    format="parquet",
                    partitioning="hive")
df_day = filter_dataset(dataset, d_day, d_day, cols=None)
#as github does not allow data longer than 25MB, we need to cut the data into days and save it as pkl.
#load outlier data for a given day
#from pickle 
#df_day = pd.read_pickle((df_day_pth+str(d_day)+'_df_day.pkl'))#.replace('/','\\'))

#for d_day in days_in_dir:
#    df_day = filter_dataset(dataset, d_day, d_day, cols=None)
#    #export to pickle
#    df_day.to_pickle((df_day_pth+str(d_day)+'_df_day.pkl').replace('/','\\'))

#Load SHAP values for a given day
try:
    shap_df = pd.read_pickle((shap_pth+str(d_day)+'_shap.pkl'))#.replace('/','\\'))
except FileNotFoundError:
    st.error(f"SHAP file not found.")
    st.stop()
#Load explainer
try:
    explainer = pickle.load(open(explainer_file, 'rb'))
except FileNotFoundError:
    st.error(f"Explainer file not found.")
    st.stop()   
#Load feature description
try:
    feature_description = pd.read_excel(feature_description_pth)
except FileNotFoundError:
    st.error(f"Feature description file not found.")
    st.stop()   

#Load edges and nodes
try:
    edges_df = pd.read_pickle((edges_pth+'edges_df_'+str(d_day)+'.pkl'))#.replace('/','\\'))
except FileNotFoundError:
    st.error(f"Edges file not found.")
    st.stop()

#Extract features from shap_df. Features are columns of shap_df_filtered without groupby_levels
features = list(shap_df.columns.difference(groupby_levels))


thr_entropy = 0.5#3.5e-5
thr_ecdf = 0.95
thr_turnover = 15000
reported_nodes = df_day.loc[(df_day['ecdf'] > thr_ecdf) & (df_day['turnover'] > thr_turnover), ['node_key_id']+groupby_levels+['ecdf','turnover']]
#reported_nodes sort by turnover
reported_nodes = reported_nodes.sort_values(by='turnover', ascending=False)
#conver turnover to integer
reported_nodes['turnover'] = reported_nodes['turnover'].astype(int)
#Display reported_nodes in a table
st.table(reported_nodes)
reported_idx = reported_nodes.index

if len(reported_idx) == 0:
    st.write('No outliers found for the given day. Please select another day.')
else:
    st.header('Choose a node to analyze and press the Analyze button')
    # Add a selection box to the application to select the node index
    start_node = st.selectbox('Select node to analyze', reported_idx, on_change=None, key=None)
    #start_node =13201792 # 13198576 #13201792
    #Describe value and prediction for the given day and node
    st.write('Node description:')
    node_desc = df_day.loc[start_node, groupby_levels].to_frame().T
    #replace NaN with 'All'
    node_desc = node_desc.fillna('All')
    st.write(node_desc, index = False)
    st.write('Node '+ measure1 + ':')
    st.write(df_day.loc[start_node, measure1])#move the multiplication to aggregation function, df_day.loc[start_node, measure1]
    st.write('Predicted '+ measure1)
    st.write(df_day.loc[start_node, preds1])
    st.write('Node '+ measure + ':')
    st.write(df_day.loc[start_node, measure])
    st.write('Predicted '+ measure)
    st.write(df_day.loc[start_node, preds])

    if st.button('Explain prediction'):
        #Load SHAP values for a given day
        hi_or_low = 'high' if df_day.loc[start_node, measure] > df_day.loc[start_node, preds] else 'low'
        st.write('Why is the predicted '+ measure + ' '+ hi_or_low + ' for the node?')
        #Barplot to compare measure and prediction
        fig, ax = plt.subplots()
        ax.bar(['measure', 'prediction'], [df_day.loc[start_node, measure], df_day.loc[start_node, preds]])
        ax.set(xlabel='', ylabel=measure,
                title='Comparison of measure and prediction')
        #add value labels
        for i in ax.patches:
            ax.text(i.get_x() + i.get_width()/2, i.get_height() + 0.1, str(round(i.get_height(),2)), ha = 'center', va = 'bottom')
        ax.grid()
        st.pyplot(fig)
        #Aggregate SHAP values for the start_node
        st.write('SHAP values for the node:')
        sum_shap_df_filtered_np = aggregate_shap(start_node, groupby_levels, shap_df, reported_nodes, features)
        #shap.initjs()
        max_display = 7
        # Create a new matplotlib figure
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=sum_shap_df_filtered_np, base_values=explainer.expected_value, feature_names = features), max_display=max_display, show=True)
        st.pyplot(fig)
        st.write('This plot shows the contribution of each feature to the difference between the measure and the prediction. The base value is the expected value of the prediction. The sum of the SHAP values and the base value is the prediction.')
        #Display feature description of max_display-1 features (-1 as watterfall plot shows max_display features -1 + "others")
        display_description_maxshap(max_display-1, feature_description, sum_shap_df_filtered_np, features)
    if st.button('View node stats'):    
        #---Time series plots and outlier analysis for the given node ---
        #Load time series for a given node
        node_key_id = reported_nodes.loc[start_node,'node_key_id']
        ts = dataset.to_table(filter=(ds.field("node_key_id") == node_key_id ), columns=None).to_pandas()
        #Plot the time series and mark the value of d_day with red
        fig, ax = plt.subplots()
        ax.plot(ts['d'], ts[measure])
        ax.set(xlabel='day', ylabel=measure,
                title='Time series of the measure')
        #Mark the d_day point with red, linewidth 0.5, type --
        ax.axvline(x=d_day, color='r', linewidth=0.5, linestyle='-.')
        ax.grid()
        st.pyplot(fig)
        #plot the r
        fig, ax = plt.subplots()
        ax.plot(ts['d'], ts['r'])
        ax.set(xlabel='day', ylabel='residual',
                    title='Time series of the residuals')
        #Mark the d_day point with red, linewidth 0.5, type --
        ax.axvline(x=d_day, color='r', linewidth=0.75, linestyle='-.')
        ax.grid()
        st.pyplot(fig)
        #Historgram of r
        fig, ax = plt.subplots()
        ax.hist(ts['r'], bins='auto') #len(ts)//10
        ax.set(xlabel='residual_bin', ylabel='Frequency',
                    title='Histogram of residuals')
        #Mark the value of the d_day with red, linewidth 0.5, type --
        ax.axvline(x=ts.loc[ts['d'] == d_day, 'r'].values[0], color='r', linewidth=0.5, linestyle='-.')
        ax.grid()
        st.pyplot(fig)


    if st.button('Analyze'):
        #Instantiate trajectory
        trajectory_cols = ['src', 'dst', 'entropy', 'ecdf_src', 'ecdf_dst', 'edge_label', 'direction']
        trajectory = pd.DataFrame(columns=trajectory_cols)
        trajectory_tmp = pd.DataFrame(columns=trajectory_cols)

        ###Drill down###
        neighbor_buffer = [start_node]

        while len(neighbor_buffer)>0: #Until there are no more neighbors to connect with edges
            #Update current src node
            src = neighbor_buffer[0] 
            #Find outlying low-entropy neighbors of the start_node
            edges_tmp = edges_df[(edges_df['src']==src) & (edges_df['entropy']<thr_entropy)& (edges_df['ecdf_dst']>thr_ecdf)].copy()
            #Select only the highest entropy dst node as several dimensions neighbors may fulfill the condition - 
            if len(edges_tmp) > 0: #If there are any outlying low-entropy neighbors for src
                edges_tmp = edges_tmp[edges_tmp['entropy'] == edges_tmp['entropy'].max()]
                #Create trajectory_step
                trajectory_tmp.loc[0,'src'] = src
                trajectory_tmp.loc[0,'dst'] = edges_tmp['dst'].tolist()
                trajectory_tmp.loc[0,'entropy'] = edges_tmp['entropy'].iloc[0] #all same
                trajectory_tmp.loc[0,'ecdf_src'] = edges_tmp['ecdf_src'].iloc[0] #all same
                trajectory_tmp.loc[0,'ecdf_dst'] = edges_tmp['ecdf_dst'].tolist()
                trajectory_tmp.loc[0,'edge_label'] = edges_tmp['edge_label'].iloc[0]
                #Update trajectory
                trajectory = pd.concat([trajectory, trajectory_tmp], ignore_index=True)
                #Update nodes  buffer
                #add dst_nodes to the neighbor_buffer
                neighbor_buffer = neighbor_buffer + edges_tmp['dst'].tolist()
                #drop duplicates
                neighbor_buffer = list(set(neighbor_buffer))
            #Remove src from the neighbor_buffer as it was already connected
            neighbor_buffer.remove(src)
        trajectory['direction'] = 'down'  
        trajectory.reset_index(drop=True, inplace=True)
        #display(trajectory)

        ####Roll up####
        dst = start_node
        #reset trajectory loop temporary 
        trajectory_tmp = trajectory_tmp = pd.DataFrame(columns=trajectory_cols)


        while True:
            edges_tmp = edges_df.loc[(edges_df['dst']==dst) & (edges_df['entropy']<thr_entropy)& (edges_df['ecdf_dst']>thr_ecdf),:].copy()
            #Select only the highest entropy dst node
            if len(edges_tmp) > 0: #If there are any outlying low-entropy neighbors for src
                edges_tmp = edges_tmp[edges_tmp['entropy'] == edges_tmp['entropy'].max()]
                #This can only be one line (unlike in drill down where there can be multiple dst nodes)
                trajectory_tmp.loc[0,'src'] = int(edges_tmp['src'].iloc[0])
                trajectory_tmp.loc[0,'dst'] = edges_tmp['dst'].tolist()
                trajectory_tmp.loc[0,'entropy'] = edges_tmp['entropy'].iloc[0] #all same
                trajectory_tmp.loc[0,'ecdf_src'] = edges_tmp['ecdf_src'].iloc[0] #all same
                trajectory_tmp.loc[0,'ecdf_dst'] = edges_tmp['ecdf_dst'].tolist()
                trajectory_tmp.loc[0,'edge_label'] = edges_tmp['edge_label'].iloc[0]
                trajectory = pd.concat([trajectory, trajectory_tmp], ignore_index=True)
                #Update src
                dst = edges_tmp['src'].iloc[0] #allway only one node in the buffer
            else:
                break
        #add direction 'up' to the trajectory where it is missing
        trajectory['direction'] = trajectory['direction'].fillna('up')
        
        
        #Write real contamination
        st.header('Real contamination:')
        if d_day == 1549:
            st.write('Contaminations placed at:')
            st.write(pd.DataFrame({'cont_day': 1549, 'dim_name':['product', 'geo'], 'level_name':['item_id', 'store_id'], 'cat_name':['FOODS_3_827','CA_3'], 'level':[0,0], 'cont_value': 2500}))  
        if d_day == 1556:
            st.write('Contaminations placed at:')
            st.write(pd.DataFrame({'cont_day': 1556, 'dim_name':['product', 'geo'], 'level_name':['dept_id', 'store_id'], 'cat_name':['HOUSEHOLD_1','WI_3'], 'level':[1,0], 'cont_value': 13}))
        if d_day == 1616:
            st.write('Contaminations placed at:')
            st.write(pd.DataFrame({'cont_day': 1616, 'dim_name':['geo'], 'level_name':['store_id'], 'cat_name':['TX_1'], 'level':[0], 'cont_value': 2}))
        if d_day == 1706:
            st.write('Contaminations placed at:')
            st.write(pd.DataFrame({'cont_day': 1706, 'dim_name':['product'], 'level_name':['dept_id'], 'cat_name':['HOBBIES_2'], 'level':[0], 'cont_value': 3}))
        if d_day not in [1549, 1556, 1616, 1706]:
            st.write('No contamination placed at this day')


        #-----------------------------
        st.header('Trajectory:')
        st.write('src->dst') 
        st.write('  *ecdf = outlyingness ([0,1] the higher the more outlying ') 
        st.write('  *entropy is information vallue (the lower the better)')
        st.write(trajectory)
        
        #Streamlit write heading 'graph'
        st.header('Graph')
        #Streamlit write text
        st.write('Nodes: ')
        st.write('  *Color to correspond with ecdf ')
        st.write('  *Size with turnover')
        st.write('  *Label with the node id')
        st.write('  *Border color with yellow for the start node')
        #Streamlit write text
        st.write('Edges: ')
        st.write('  *Color of the edges corresponds with the entropy')
        st.write('  *Label with the dimension level that is added to the node in the drill down direction')
        #-----------------------------

        #List of unique nodes appearing in the trajectory (src or dst)
        list_nodes = list(trajectory['dst'])
        #flatten the list
        list_nodes = [item for sublist in list_nodes for item in sublist]
        #add the src nodes
        list_nodes = list_nodes + trajectory['src'].to_list()
        #unique
        list_nodes = list(set(list_nodes))


        #This is to map the colors to the entropy values and ecdf values. We use a different one as the scale (normalization) may be different
        # Create a color map from red to blue (entropy)
        cmap_r = plt.get_cmap('coolwarm_r')
        # Create a color map from blue to red (ecdf)
        cmap = plt.get_cmap('coolwarm')

        # Create a Normalize object for mapping entropy values to the range [0, 1]
        norm_node = mcolors.Normalize(vmin=thr_ecdf, vmax=1)
        norm_edge = mcolors.Normalize(vmin=thr_entropy, vmax=1)

        #instantiate nodes and edges
        nodes = dict()
        edges = []

        #-----add to nodes------
        for i_node in list_nodes:
            color_value = cmap(norm_node(df_day.loc[i_node,'ecdf']))
            color_value = mcolors.to_hex(color_value)
            nodes[i_node] = {'metadata':  {'color':color_value, 
                                        'size': int(df_day.loc[i_node,'turnover']), 
                                        'label': create_label(i_node), 
                                        'border_color': 'yellow' if i_node == start_node else color_value,
                                        'border_size': 2 if i_node == start_node else 0
                                        }}


        #-----add to edges------
        for i_src in range(len(trajectory)):
            for i_dst in range(len(trajectory.loc[i_src,'dst'])):

                # Map the entropy value to a color
                color_value = cmap_r(norm_edge(trajectory.loc[i_src,'entropy']))
                color_value = mcolors.to_hex(color_value)       
                edge_tmp = {'source': trajectory.loc[i_src,'src'], 
                            'target': trajectory.loc[i_src,'dst'][i_dst],
                            'metadata': {'color':color_value, 
                                        'edge_label': trajectory.loc[i_src,'edge_label']
                                        } #get metadata from edges_df (add turnover to edges_df too)
                            }
                edges.append(edge_tmp)

        #Create the graph
        graph1 = {
            'graph':{
                'directed': True,
                'metadata': {
                    'arrow_size': 5,
                    'background_color': 'black',
                    'edge_size': 3,
                    'edge_label_size': 14,
                    'edge_label_color': 'white',
                    'node_label_size': 10,
                    'node_label_color': 'yellow'

                },
                'nodes': nodes,
                'edges': edges,
            }
        }
        
        gv_fig = gv.vis(graph1, 
        show_node_label=True, 
        show_edge_label=True, 
        edge_label_data_source='edge_label', 
        node_label_data_source='label', 
        node_size_factor=1/1000,
        gravitational_constant=-13000,
        avoid_overlap=0.35
        )

        components.html(gv_fig.to_html(), height=800)
        #https://stackoverflow.com/questions/77730137/how-to-integrate-gravis-visualization-inside-of-streamlit

    
 

 

#import streamlit.components.v1 as components

#components.html(fig.to_html(), height=600) 





#and a button "analyze"

#If the button is clicked, print the integer 


#radio_buttons_df = pd.DataFrame({' ': ['<input type="radio" name="option">' for _ in range(len(reported_nodes))]}, index=reported_nodes.index)
#radio_buttons_df = pd.DataFrame({'Select': ['<input type="radio" name="option_'+str(i)+'">' for i in reported_nodes.index]}, index=reported_nodes.index)


# Concatenate the DataFrames
#full_df = pd.concat([radio_buttons_df, reported_nodes], axis=1)

# Convert the DataFrame to HTML
#full_html = full_df.to_html(escape=False, index=True)

# Display the HTML in Streamlit
#st.markdown(full_html, unsafe_allow_html=True)

