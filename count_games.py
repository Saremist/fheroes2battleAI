with open("training_log.txt", "r") as log_file:
    logs = log_file.readlines()

games_count = 0
total_time = 0

for line in logs:
    number =int(line.split("|")[-1].split(" ")[-1] )
    time = int(line.split("|")[1].split(" ")[-2].split(":")[-1])
    games_count += number
    total_time += time
    
total_hours = total_time // 3600

print(games_count) 
print(total_hours)