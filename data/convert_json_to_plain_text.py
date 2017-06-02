with open("reddit_jokes.json") as f:
	i = 0
	for line in f:
		if i > 50: break
		print line
		i += 1
