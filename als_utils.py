# Creates a dictionary to map user/item id to sparse_matrix index
def create_id_to_index_dic(ids_list):
    count = 0
    ids_dic = dict()
    for iden in ids_list:
        ids_dic[iden] = count
        count += 1
    return ids_dic
# Create a dictionary to map sparse_matrix index to user/item id
def create_index_to_id_dic(ids_list):
    count = 0
    ids_dic = dict()
    for iden in ids_list:
        ids_dic[count] = iden
        count += 1
    return ids_dic

def create_confidence_matrix(train_df, user_to_item_matrix, user_id_to_index_dic, item_id_to_index_dic):
    """
        Creates a confidence matrix based on weighted frequency of user events
    """
    # Create the confidence matrix
    action_weights = [1,2,3]
    for row in train_df.itertuples():

        user_id = row[2]
        item_id = row[4]
        value = 0
        if row.event == 'view':
            value = action_weights[0]
        elif row.event == 'addtocart':
            value = action_weights[1]       
        elif row.event == 'transaction':
            value = action_weights[2]

        previous_value = user_to_item_matrix[user_id_to_index_dic[user_id], item_id_to_index_dic[item_id]]
        user_to_item_matrix[user_id_to_index_dic[user_id], item_id_to_index_dic[item_id]] = previous_value + value
        
    return user_to_item_matrix

def create_confidence_matrix2(train_df, user_to_item_matrix, user_id_to_index_dic, item_id_to_index_dic):
    """
        Creates a confidence matrix based on the highest weighted event action taken by the user
    """
    # Create the confidence matrix
    action_weights = [1,2,3]
    for row in train_df.itertuples():

        user_id = row[2]
        item_id = row[4]
        value = 0
        if row.event == 'view':
            value = action_weights[0]
        elif row.event == 'addtocart':
            value = action_weights[1]       
        elif row.event == 'transaction':
            value = action_weights[2]

        previous_value = user_to_item_matrix[user_id_to_index_dic[user_id], item_id_to_index_dic[item_id]]
        if value > previous_value:
            user_to_item_matrix[user_id_to_index_dic[user_id], item_id_to_index_dic[item_id]] = value
        
    return user_to_item_matrix

def find_sparsity(user_to_item_matrix):
    sparsity = float(len(user_to_item_matrix.nonzero()[0]))
    sparsity /= (user_to_item_matrix.shape[0] * user_to_item_matrix.shape[1])
    sparsity = 1 - sparsity
    sparsity *= 100
    print (f"Sparsity = {sparsity}")
    
def get_user_activity_count(df):
    user_activity_count = dict()
    for row in df.itertuples():
        if row.visitorid not in user_activity_count:
            user_activity_count[row.visitorid] = {'view':0 , 'addtocart':0, 'transaction':0};
        if row.event == 'addtocart':
            user_activity_count[row.visitorid]['addtocart'] += 1 
        elif row.event == 'transaction':
            user_activity_count[row.visitorid]['transaction'] += 1
        elif row.event == 'view':
            user_activity_count[row.visitorid]['view'] += 1
            
    return user_activity_count

def find_total_user_activities(activities):
    total = 0
   
    for key in activities.keys():
        total += activities[key]
            
    return total

def find_hit_ratio(model, user_to_item_trained, test_users_activities, test_df, user_id_to_index_dic, index_to_item_id_dic,filter_already_liked_items=True, N=100):
    hits = dict()
    print(f"Find hit-ratio with flag filter_already_liked_items = {filter_already_liked_items}")
    test_set_userids = set(test_df['visitorid'].unique())
    train_set_userids = set(user_id_to_index_dic.keys())
    matching_users = train_set_userids.intersection(test_set_userids)
    print(f"Total # of common userIds in TrainSet and TestSet = {len(matching_users)}")
    # Iterate through the test set
    for user_id in list(matching_users):
        # Find all the items user actually performed view/add/transact
        item_ids = set(test_df[(test_df.visitorid == int(user_id))]['itemid'].tolist())
        if user_id in user_id_to_index_dic.keys():
            # Find the top 100 recommendations
            recommendations = model.recommend(user_id_to_index_dic[user_id], user_to_item_trained, N=N, filter_already_liked_items=filter_already_liked_items)
            # convert sparse_matrix_indices to item_id
            rec_item_ids = [ index_to_item_id_dic[i[0]] for i in recommendations if i[0] in index_to_item_id_dic.keys()]
            # Check if there there is any hit between user operations and recommendations
            hit = list(item_ids.intersection(set(rec_item_ids)))
            if hit:
                hit_ratio = len(hit) * 100 / find_total_user_activities(test_users_activities[user_id])
                #print(f"Hit Ratio for user_id: {user_id} =  {hit_ratio:.3f}")
                hits[user_id] = hit_ratio
    print(f"Total # of userIds for successful Recommendation = {len(hits)}")
    print(f"Total Coverage of Test dataset = {len(hits)/len(matching_users) * 100}")
    return hits

def train_test_split(df, num_days=1):
    """
        Splits the input dataset based on the num_days parameter passed by the user. test_df = #num_days data
        Default num_days=1
        Returns train_df, test_df
    """
    print(f"Spliting the dataframe with test data = {num_days} day(s)")
    last_day = max(df['date'])
    if num_days == 1:
        test_df = df[(df.date == last_day)]
        train_df = df[(df.date != last_day)]
    elif num_days > 1:
        test_df = df[(df.date <= last_day) & (df.date > last_day + datetime.timedelta(-num_days))]
        train_df = df[(df.date <= last_day + datetime.timedelta(-num_days))]
        
    print(f"Training set length = {len(train_df)}")
    print(f"Test set length = {len(test_df)}")
    
    test_df.reset_index(drop=True, inplace=True)
    train_df.sort_values('date',inplace=True)
    
    return train_df, test_df

def filter_data_by_events_count(df, min_events_count=2):
    """
     This method will delete all the records for users whose total_events_count < min_events_count
    """
    print(f"Total Unique users in original df = {len(df['visitorid'].unique())}")
    grouped_df = df.groupby('visitorid').count()
    ids_to_delete = list(grouped_df[(grouped_df.event < min_events_count)].index)
    
    df.set_index('visitorid', drop=False, inplace=True)
    df.drop(ids_to_delete, inplace=True)
    df.reset_index(drop=True, inplace=True)
          
    print(f"Total Unique users in filtered df where # of user transactions >= {min_events_count} = {len(df['visitorid'].unique())}")
    return df

def filter_data_by_items_count(df, min_items_count=2):
    """
     This method will delete all the records for users whose total_item_count < min_items_count
    """
    print(f"Total Unique users in original df = {len(df['visitorid'].unique())}")
    grouped_df = df.groupby('itemid').count()
    ids_to_delete = list(grouped_df[(grouped_df.visitorid < min_items_count)].index)
    
    df.set_index('itemid', drop=False, inplace=True)
    df.drop(ids_to_delete, inplace=True)
    df.reset_index(drop=True, inplace=True)
          
    print(f"Total Unique users in filtered df where # of items >= {min_items_count} = {len(df['visitorid'].unique())}")
    return df    