import json

with open("reddit_jokes.json", "r") as f:
    jokes = json.load(f)

plaintext = []
counter = 0
for joke in jokes:
    text = joke["title"].replace("\n", " ") + " " + joke["body"].replace("\n", " ") + "\n"
    plaintext.append(text)
    
with open("reddit_jokes.txt", "w") as f:
    for line in plaintext:
        f.write(line.encode('utf8'))
        
