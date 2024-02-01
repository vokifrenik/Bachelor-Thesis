import matplotlib.pyplot as plt

# Initialize lists to store data
score_history = []
death_history = []
total_scores_100 = []
total_deaths_100 = []

# Read the data from the text file
# "C:\Users\vikpl\OneDrive\Desktop\results.txt"
with open('C:/Users/vikpl/OneDrive/Desktop/results_no_ensemble.txt', 'r') as file:
    content = file.readlines()

# Flags to identify which section of the file is being read
reading_scores = False
reading_deaths = False
reading_total_scores = False
reading_total_deaths = False

# Process each line in the file
for line in content:
    if 'Scores per Episode:' in line:
        reading_scores = True
        continue
    elif 'Deaths per Episode:' in line:
        reading_scores = False
        reading_deaths = True
        continue
    elif 'Total Scores per 100 Episodes:' in line:
        reading_deaths = False
        reading_total_scores = True
        continue
    elif 'Total Deaths per 100 Episodes:' in line:
        reading_total_scores = False
        reading_total_deaths = True
        continue
    
    if reading_scores and line.strip():
        # take onl even lines
        score_history.append(float(line.strip()))
    elif reading_deaths and line.strip():
        death_history.append(float(line.strip()))
    elif reading_total_scores and line.strip():
        total_scores_100.append(float(line.strip()))
    #elif reading_total_deaths and line.strip():
        #total_deaths_100.append(int(line.strip()))

# count total score
total_score = 0
count = 0
for score in score_history:
    if count != 10000:
        total_score += score
        count += 1

# count total deaths
total_deaths = 0
count = 0
for death in death_history:
    if count != 10000:
        total_deaths += death
        count += 1
    

print("total score: ", total_score)
print("total deaths: ", total_deaths)


histogram = [ 558.0, 70.0, 13.0]

plt.bar(['Base', 'Ensemble', 'Safe'], histogram, color=['green', 'red', 'blue'])
plt.title('Total Deaths (Excluding Timeouts)')
plt.xlabel('Model Type')
plt.ylabel('Value')
plt.grid(True)
plt.show()

'''
adj_score_history = []
count = 0
## Plot the scores per episode by skipping every second score because they are repeated
for i in range(0, len(score_history), 2):
    if count != 10000:
        adj_score_history.append(score_history[i])
        count += 1



plt.plot(adj_score_history)
plt.title('Scores per Episode (Safe)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()
    '''

'''
# Plotting
plt.figure(figsize=(12, 8))

# Scores per episode
plt.subplot(2, 2, 1)
plt.plot(adj_score_history, label='Score per Episode')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Scores per Episode')
plt.grid(True)
plt.legend()


# Deaths per episode
plt.subplot(2, 2, 2)
plt.plot(death_history, label='Deaths per Episode', color='red')
plt.xlabel('Episodes')
plt.ylabel('Deaths')
plt.title('Deaths per Episode')
plt.grid(True)
plt.legend()

# Total scores per 100 episodes
plt.subplot(2, 2, 3)
plt.plot(total_scores, label='Total Score', color='green')
plt.xlabel('Episodes (x100)')
plt.ylabel('Total Score')
plt.title('Total Scores per 100 Episodes')
plt.grid(True)
plt.legend()

# Total deaths per 100 episodes
plt.subplot(2, 2, 4)
plt.plot(summed_deaths, label='Total Deaths per 100 Episodes', color='purple')
plt.xlabel('Episodes (x100)')
plt.ylabel('Total Deaths')
plt.title('Total Deaths per 100 Episodes')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
        
'''



'''#p#/lt.scatter(range(len(adj_score_history)), adj_score_history, alpha=0.6)
plt.plot(summed_deaths, label='Deaths per 100 Episodes (Safe)', color='red')
plt.title('Deaths Accumulated per 100 Episodes (Base)')
plt.xlabel('Episodes (100x)')
plt.ylabel('Deaths')
#plt.yticks([0, 1], ['No Death', 'Death'])
plt.grid(True)
plt.show()'''


'''
histogram = [7868, 1814]

plt.bar(['Base', 'Safe'], histogram, color=['green', 'red'])
plt.title('Total Deaths per Training')
plt.xlabel('Model Type')
plt.ylabel('Deaths')
plt.grid(True)
plt.show()

'''

