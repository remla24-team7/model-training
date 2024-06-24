import matplotlib.pyplot as plt

with open('pylint_report.txt', 'r') as file:
    for line in file:
        if 'Your code has been rated at' in line:
            score = float(line.split('/')[0].split()[-1])

plt.bar(['Pylint Score'], [score])
plt.ylim(0, 10)
plt.ylabel('Score')
plt.title('Pylint Score Overview')
plt.savefig('pylint_score.png')
