with open("training_log.txt", "r") as log_file:
    logs = log_file.readlines()

games_count = 0

for line in logs:
    number =int(line.split("|")[-1].split(" ")[-1] )
    games_count += number
    
print(games_count) 