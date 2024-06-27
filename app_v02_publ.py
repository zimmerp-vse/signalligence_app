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


########SELECT DAY########
#d_day = 1556
#########################


##Config
config_file_path = 'config_app.conf'
root_prefix = ''
#config_file_path = '/content/gdrive/MyDrive/signalligence/config.conf'
#root_prefix = '/content/gdrive/MyDrive/signalligence/'
config_read = configparser.ConfigParser()
config_read.read(config_file_path)
#preprocessing_package_file = root_prefix+config_read.get("Paths", "preprocessing_package_file")
##training_outlier_data_file = root_prefix+config_read.get("Paths", "training_outlier_data_file")
dim_level_structure_pth = root_prefix+config_read.get("Paths", "dim_level_structure_pth")
dayly_data_pth = root_prefix+config_read.get("Paths", "df_day_pth")
edges_pth = root_prefix+config_read.get("Paths", "edges_pth")


_, groupby_levels, _ = pd.read_pickle(dim_level_structure_pth.replace('/','\\'))

#list of files in dim_level_structure_pth
#dim_files = os.listdir(dayly_data_pth)
#extract numbers from the filenames after 'before '_'
#days_in_dir = [int(i.split('_')[0]) for i in dim_files]
days_in_dir = [1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560]



#for d_day in days_in_dir:
#    df_day = filter_dataset(dataset, d_day, d_day, cols=None)
#    #export to pickle
#    df_day.to_pickle((dayly_data_pth+str(d_day)+'_df_day.pkl').replace('/','\\'))
    

st.title('Signalligence')

st.header('Choose a day to analyze')
# Add a selection box to the application to select the 'id'
st.write(f'You can selected DAY. Contaminated days are: 1549, 1556, 1616, 1706')
d_day = st.selectbox('Select DAY', days_in_dir)


reported_idx = []

#load pickle
df_day = pd.read_pickle((dayly_data_pth+str(d_day)+'_df_day.pkl').replace('/','\\'))

#Load edges and nodes
try:
    edges_df = pd.read_pickle((edges_pth+str(d_day)+'_edges_df.pkl').replace('/','\\'))
except FileNotFoundError:
    st.error(f"File not found.")
    #stop st
    st.stop()

thr_entropy = 0.5#3.5e-5
thr_ecdf = 0.95
thr_turnover = 15000
reported_nodes = df_day.loc[(df_day['ecdf'] > thr_ecdf) & (df_day['turnover'] > thr_turnover), groupby_levels+['ecdf','turnover']]
#reported_nodes sort by turnover
reported_nodes = reported_nodes.sort_values(by='turnover', ascending=False)
#conver turnover to integer
reported_nodes['turnover'] = reported_nodes['turnover'].astype(int)
#Display reported_nodes in a table
st.table(reported_nodes)
reported_idx = reported_nodes.index


st.header('Choose a node to analyze and press the Analyze button')
# Add a selection box to the application to select the node index
start_node = st.selectbox('Select node to analyze', reported_idx, on_change=None, key=None)

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

