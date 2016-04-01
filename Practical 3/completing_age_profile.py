import numpy as np
import csv
import pandas as pd

# Create helper function to fill in missing ages based on country and gender (if available)
def fill_ages(indf):
    '''This helper function takes in a cleaned dataframe (with columns: user, age, sex and country)
    where invalid ages have been replaced with null values. This function infers a user's age based
    on their country and gender. If one or both of those is missing, the inference is performed on 
    the most informative subset of the data we have (at worst this is the entire dataset).'''
    # First copy dataframe for safety's sake
    outdf = indf.copy()
    
    # Loop through the entries in the data frame that have null valued age
    for index, (user_id,gender,age,country) in indf[indf.age.isnull()].iterrows():
        
        # Build up conditions over which we're calculating distribution of ages for this particular user
        #nans won't equal themselves
        if gender != gender:       # gender is not present
            if country != country: # country isn't present either
                
                condition = (~indf.age.isnull())
            else:
                
                condition = (~indf.age.isnull()) & (~indf.sex.isnull()) & (indf.country == country)
        elif country != country:   # country isn't present
            
            condition = (~indf.age.isnull()) & (indf.sex == gender)
        else:                      # both are present
            
            condition = (~indf.age.isnull()) & (indf.sex == gender) & (indf.country == country)
        
        # Extract array of ages relevant to the current user, based on country and gender (if available)
        relevant_ages = np.array(indf.age[condition]).astype('int64')
        relevant_ages_aggregated = np.bincount(relevant_ages)
        # Normalize to create a distribution (assuming that it's continuous between low and high (no need for pseudocounts))
        normed_ages = relevant_ages_aggregated/float(np.sum(relevant_ages_aggregated))
        
        # Found that there are some edge cases where there are few users from nation x that all have no age... 
        # Recalculate distribution
        if len(normed_ages) == 0: 
            if gender != gender:
                condition = (~indf.age.isnull()) & (~indf.sex.isnull())
            else:
                condition = (~indf.age.isnull()) & (indf.sex == gender)
            relevant_ages = np.array(indf.age[condition]).astype('int64')
            relevant_ages_aggregated = np.bincount(relevant_ages)
            normed_ages = relevant_ages_aggregated/float(np.sum(relevant_ages_aggregated))
            
        # Randomly assign user age based on distribution
        new_age = np.random.choice(range(len(normed_ages)),size=1,replace=True,p=normed_ages)[0]
        outdf.loc[index,'age'] = new_age
    
    # Return filled dataframe
    return outdf


if __name__ == '__main__':
	# Load user profiles
	user_databasedf = pd.read_csv("profiles.csv")

	# Create copy to remove spurrious ages (anything less than 10 and above 65)
	fixed_age_databasedf = user_databasedf.copy()
	fixed_age_databasedf.loc[(user_databasedf.age <= 8) | (user_databasedf.age > 65),'age'] = None

	# Fill values of missing age
	completed_age_databasedf = fill_ages(fixed_age_databasedf)
	completed_age_databasedf.to_csv("complete_age_profiles.csv")

