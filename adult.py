

def processLine(line):
   values = line.strip().split(', ')
   (age, employer, _, _, education, marital, occupation, _, race, sex,
      capital_gain, capital_loss, hr_per_week, country, income) = values

   return [(int(age), employer, int(education), marital, occupation,
         race, sex, int(capital_gain), int(capital_loss), int(hr_per_week),
         country), (1 if income == '>50K' else -1)]


if __name__ == '__main__':
   with open('data/adult.data', 'r') as infile:
      trainingData = [processLine(line) for line in infile]

   with open('data/adult.test', 'r') as infile:
      testData = [processLine(line) for line in infile]



