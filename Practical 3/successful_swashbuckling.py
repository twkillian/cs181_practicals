import numpy as np
import csv
import matplotlib.pyplot as plt
# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'

# Load the training data.
train_data = {}
# key = user, value = number of artists that the user listened to
num_diff_artists = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
            num_diff_artists[user] = 0.0
        
        train_data[user][artist] = int(plays)
        num_diff_artists[user] += 1

# Plot histogram of number of artists that a user listened to
plt.hist(num_diff_artists.values,bins=100)
plt.show()

# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]


            if user in user_medians:
                n = num_diff_artists[user]
                prediction  = 1.0/n * global_median + (1-1.0/n) * user_medians[user]
                soln_csv.writerow([id, prediction])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                