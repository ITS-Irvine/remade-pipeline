import pandas as pd
import json
from IPython.display import display 

dataarr=[]
# each tmp.json is the download of the response data from the google /search? query
# the first is the initial response, the rest are the subsequent pages of results
# pulled when you scroll down the query results pane
for f in ['tmp.json','tmp2.json','tmp3.json','tmp4.json','tmp5.json']:
    with open(f,'r') as file:
        dataarr.append(json.load(file))

list_of_all_relevant_data=[]
for addr_data in [json.loads(data['d'][5:]) for data in dataarr]:
    list_of_relevant_data = (

        [                       # for each element in the list created by the for statement at the below
                                # create a simpler list containing...
            [x[14][11],         # Name is the 12th entry of the 15th entry (python lists are zero indexed)
            '; '.join(x[14][2]) # Address is a list of elements (street addr, city) stored in the 2nd entry 
                                # of the 15th entry, I concat them with a '; '
            ]
            +x[14][9][2:]       # lat/lon are the 3rd and 4th elements (2:) of a list stored in the 10th entry of
                                # the 15th entry. since this is a list, I concatenate it using the + operator
                                # to the [name, addr] list

                                # at this point, we have constructed a 4 element list with [name, addr, lat, lon]

            # all the stuff above is applied to list item x from the main data; x is the iterator value for the
            # elements of the list in addr_data[0][1] (2nd element of the first element of addr_data)

            for x in addr_data[0][1] if x is not None          # we filter out bad list elements...
                                        and len(x)>14          # and those that don't have 15 elements
                                        and x[14] is not None  # and those that have a null 15th element
                                        and type(x[14])==list  # and those where the 15th element isn't a list
        ]
    )
    list_of_all_relevant_data += list_of_relevant_data

import pandas as pd
df=(
    pd.DataFrame(list_of_all_relevant_data)                 # make our data a data frame
    .rename(columns={0:'name',1:'address',2:'lat',3:'lon'}) # give the columns names
)
display(df)
df.to_csv('stores.csv')