import matplotlib.pyplot as plt

def parse_scores_per_episode(file_path):
    scores_per_episode = {}

    with open(file_path, 'r') as file:
        for line in file:
            if 'episode' in line and 'score' in line:
                parts = line.split()
                episode = int(parts[1])
                score = float(parts[3])
                scores_per_episode[episode] = score

    return scores_per_episode


# Usage example
#"C:\Users\vikpl\OneDrive\Desktop\job-7001079.log"
file_path = 'C:/Users/vikpl/OneDrive/Desktop/job-7001079.log'
scores = parse_scores_per_episode(file_path)
print(scores)


# # Plot the scores per episode and smoothen the curve
plt.plot(scores.keys(), scores.values())
plt.title('Scores per Episode (Ensemble)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()





